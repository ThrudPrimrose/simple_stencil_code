#include <vector>
#include <functional>
#include <cstddef> // just in case for size_t
#include <algorithm>
#include <iostream>
#include <sstream>
#include <mpi.h>
#include <cassert>

constexpr float threshold = 0.1;
constexpr float viscosity_factor = 4.0;
constexpr size_t default_dam_offset = 20;
constexpr size_t default_domain_size = 100;
constexpr size_t print_frequency = 200;

std::function<float(size_t, size_t)> symmetric_dambreak = [](size_t y, size_t x) -> float
{
    if ((x < default_dam_offset) || (x > default_domain_size - default_dam_offset))
    {
        return 20.0;
    }
    return 10.0;
};

enum class Border : size_t
{
    Top,
    Bottom,
    Left,
    Right
};

class Domain
{
private:
    // Size of 1 dimension, in the end we have a square of _d x _d
    size_t _dimension;

    // Matrix to hold cell data
    std::vector<std::vector<float>> _domain;

    // Updates array, separate to make code easier and prevent data races
    std::vector<std::vector<float>> _net_updates;

    // Amount of patch updates computed
    unsigned long long _patch_updates;

    // Ranks to be ordered from smallest to the biggest, information needed for
    // Halo exchange
    std::vector<std::pair<int, Border>> _neighbors;

    int rank_me; // Local rank in comm world
    int rank_n;  // Total amount of mpi ranks in comm world

    // In order Top, Bottom, Left, Right (same as static casting the enum class to size_t)
    // Then we will know who ranks we need to send our data to and recv from
    std::array<std::vector<float>, 4> ghost_layers;

    // Used as a boolean but easier to have int with mpi imho
    int converged;

public:
    Domain(size_t dimension, std::function<float(size_t, size_t)> initial_water_height,
           std::vector<std::pair<int, Border>> &&neighbors,
           int rank, int size)
        : _dimension(dimension),
          _domain(_dimension, std::vector<float>(_dimension, 0.0)),
          _net_updates(_dimension, std::vector<float>(_dimension, 0.0)), _patch_updates(0), _neighbors(std::move(neighbors)),
          rank_me(rank), rank_n(size), converged(0)
    {
        // Initialize the domain depending on the initializer function
#pragma omp parallel for schedule(static)
        for (size_t y = 0; y < _dimension; y++)
        {
            for (size_t x = 0; x < _dimension; x++)
            {
                _domain[y][x] = initial_water_height(y, x);
            }
        }

        // Initialize the ghost layers that will be used
        for (auto &pr : _neighbors)
        {
            ghost_layers[static_cast<size_t>(pr.second)] = std::vector<float>(_dimension, 0.0);
        }
    }

    // Default constructors imagines a domain that will run on a single rank without remote communication
    Domain() : _dimension(default_domain_size), _domain(_dimension, std::vector<float>(_dimension, 0.0)),
               _net_updates(_dimension, std::vector<float>(_dimension, 0.0)), _patch_updates(0),
               _neighbors{}, converged(0)
    {
#pragma omp parallel for schedule(static)
        for (size_t y = 0; y < _dimension - 0; y++)
        {
            for (size_t x = 0; x < _dimension - 0; x++)
            {
                _domain[y][x] = symmetric_dambreak(x, y);
            }
        }
    }

    unsigned long long simulate()
    {
        // Volatile to make sure that the variable is always loaded
        volatile bool terminate_criteria_met = 0;

        while (!terminate_criteria_met)
        {

            /*
            if (_patch_updates % 200 == 0)
            {
                print();
            }
            */

            // Exchange ghost layer data
            exchange_halo();

            // Computes the wave propogation depending on the *minecraft* like formula
            // The influx to b from a (a,b) is the same as the outflux from a to b (b,a)
            // We are computing more updates than minimally necessary which could be improved, but
            // since the cost of the update scheme is so low improving that part of the code
            // probably wont help
            compute_stencil();

            // Updates the water height, having separate update and compute phases
            // Makes it much easier to code, as there are no data races
            apply_influx();

            // Check globally if the termination criteria is fulfilled
            terminate_criteria_met = termination_criteria_fulfilled();

            _patch_updates += 1;
        }

        return _patch_updates;
    }

    // Prints the 2D domain of water heigh, the precision doesnt totally work
    void print()
    {
        std::vector<float> row_max;
        row_max.reserve(_dimension);
        for (const auto &row : _domain)
        {
            row_max.push_back(*std::max_element(row.begin(), row.end()));
        }

        float glob_max = *std::max_element(row_max.begin(), row_max.end());

        size_t digits = std::to_string(glob_max).size();

        std::stringstream ss;
        ss.precision(2);

        ss << "{" << std::endl;
        for (size_t i = 0; i < _dimension; i++)
        {
            ss << "  {";
            for (size_t j = 0; j < _domain[i].size() - 1; j++)
            {
                float el = _domain[i][j];
                size_t local_digits = std::to_string(el).size();

                size_t padding = digits - local_digits;

                for (size_t k = 0; k < padding + 1; k++)
                {
                    ss << " ";
                }

                ss << el;
                ss << ",";
            }

            float el = _domain[i][_domain[i].size() - 1];
            size_t local_digits = std::to_string(el).size();

            size_t padding = digits - local_digits;

            for (size_t k = 0; k < padding + 1; k++)
            {
                ss << " ";
            }

            ss << el;
            ss << "}" << std::endl;
        }
        ss << "}";
        std::cout << ss.str() << std::endl;
    }

private:
    /*
        This function could be avoided by creating an MPI datatype that is strided, specific mpi implementatione
        can implement the packing better than the user level code, but this is very rarely the case and needs specific
        optimized mpi implementations, could be test too
    */
    std::pair<std::vector<float>, std::vector<float>> pack_strided_halo_to_continous()
    {
        std::pair<std::vector<float>, std::vector<float>> pr;
        std::vector<float> &left = pr.first;
        std::vector<float> &right = pr.second;
        left.reserve(_dimension);
        right.reserve(_dimension);

        for (size_t y = 0; y < _dimension; y++)
        {
            left.push_back(_domain[y][0]);
            right.push_back(_domain[y][_dimension - 1]);
        }

        return pr;
    }

    void exchange_halo()
    {
        // We saved data row major, so column (left and right boundary) messages are and not continous
        // best to pack it before sending them to remote ranks
        std::pair<std::vector<float>, std::vector<float>> l_and_r = pack_strided_halo_to_continous();

        // Send the edge row and columns' data to remote ranks, also receive from the neighboring ranks at the same time
        for (const auto &pr : _neighbors)
        {
            const int neighbor_rank = pr.first;
            const Border location = pr.second;
            MPI_Status status;
            int retcode = -1;

            /*
                For rows we dont need to copy and take the domain's line as buffer, we use our layer's Border type as a tag,
                and use their layer's location as tag. If we send our TOP layer then we receive their BOTTOM layer. This is taken
                from the list of neighbors
            */

            switch (location)
            {
            case Border::Top:
            {
                retcode = MPI_Sendrecv(_domain[0].data(), _dimension, MPI_FLOAT, neighbor_rank, static_cast<size_t>(Border::Top),
                                       ghost_layers[static_cast<size_t>(Border::Top)].data(), _dimension, MPI_FLOAT, neighbor_rank, static_cast<size_t>(Border::Bottom),
                                       MPI_COMM_WORLD, &status);
                break;
            }
            case Border::Bottom:
            {
                retcode = MPI_Sendrecv(_domain[_dimension - 1].data(), _dimension, MPI_FLOAT, neighbor_rank, static_cast<size_t>(Border::Bottom),
                                       ghost_layers[static_cast<size_t>(Border::Bottom)].data(), _dimension, MPI_FLOAT, neighbor_rank, static_cast<size_t>(Border::Top),
                                       MPI_COMM_WORLD, &status);
                break;
            }
            case Border::Left:
            {
                retcode = MPI_Sendrecv(l_and_r.first.data(), _dimension, MPI_FLOAT, neighbor_rank, static_cast<size_t>(Border::Left),
                                       ghost_layers[static_cast<size_t>(Border::Left)].data(), _dimension, MPI_FLOAT, neighbor_rank, static_cast<size_t>(Border::Right),
                                       MPI_COMM_WORLD, &status);
                break;
            }
            case Border::Right:
            {
                retcode = MPI_Sendrecv(l_and_r.second.data(), _dimension, MPI_FLOAT, neighbor_rank, static_cast<size_t>(Border::Right),
                                       ghost_layers[static_cast<size_t>(Border::Right)].data(), _dimension, MPI_FLOAT, neighbor_rank, static_cast<size_t>(Border::Left),
                                       MPI_COMM_WORLD, &status);
                break;
            }
            default:
            {
                throw std::runtime_error("Unhandled case");
            }
            }

            // Just in case
            if (retcode != MPI_SUCCESS)
            {
                throw std::runtime_error(std::to_string(retcode));
            }
        }
    }

    // Update water heights
    void apply_influx()
    {
#pragma omp parallel for schedule(static)
        for (size_t y = 0; y < _dimension; y++)
        {
            for (size_t x = 0; x < _dimension; x++)
            {
                _domain[y][x] += _net_updates[y][x];
            }
        }
    }

    void compute_stencil()
    {
        // For evert cell with coordinates (y,x) compute the influx from neighbor cells
        // Apply reflecting boundary conditions

        // The function computes more updates than necessary so it could be improved but the computation
        // cost of the stencil is low, therefore it will be cheaper to re-compute than the memory access

#pragma omp parallel for schedule(static)
        for (size_t y = 0; y < _dimension; y++)
        {
            for (size_t x = 0; x < _dimension; x++)
            {
                // This is inline so vector allocation here should not cause a problem
                std::vector<std::pair<size_t, size_t>> neighbors = get_neighbors(y, x);

                float cell_water = _domain[y][x];
                float update = 0.0;
                float difference = 0.0;

                for (auto &coordinates : neighbors)
                {
                    difference = _domain[coordinates.first][coordinates.second] - cell_water;

                    // If the difference is positive influx is positive otherwise negative
                    update += difference / viscosity_factor;
                }

                // If we are not in a true corner, we need to check the ghost layers and ask them for water height
                if (x == 0 && !ghost_layers[static_cast<size_t>(Border::Left)].empty())
                {
                    difference = ghost_layers[static_cast<size_t>(Border::Left)][y] - cell_water;
                    update += difference / viscosity_factor;
                }
                if (x == _dimension - 1 && !ghost_layers[static_cast<size_t>(Border::Right)].empty())
                {
                    difference = ghost_layers[static_cast<size_t>(Border::Right)][y] - cell_water;
                    update += difference / viscosity_factor;
                }
                if (y == _dimension - 1 && !ghost_layers[static_cast<size_t>(Border::Bottom)].empty())
                {
                    difference = ghost_layers[static_cast<size_t>(Border::Bottom)][x] - cell_water;
                    update += difference / viscosity_factor;
                }
                if (y == 0 && !ghost_layers[static_cast<size_t>(Border::Top)].empty())
                {
                    difference = ghost_layers[static_cast<size_t>(Border::Top)][x] - cell_water;
                    update += difference / viscosity_factor;
                }

                _net_updates[y][x] = update;
            }
        }
    }

    // Terminate of the max and min height throughout the GLOBAL domain is under a certain threshold
    // Reducing of the matrix can be split row wise to parallelize the code, then the local maxima and minima
    // How to be globally reduced
    bool termination_criteria_fulfilled() const
    {
        std::vector<float> max_elems(_dimension, 0.0);
        std::vector<float> min_elems(_dimension, 0.0);

#pragma omp parallel for
        for (size_t y = 0; y < _dimension; y++)
        {
            const std::vector<float> &row = _domain[y];
            auto mm = std::minmax_element(std::begin(row), std::end(row));
            min_elems[y] = *(mm.first);
            max_elems[y] = *(mm.second);
        }

        float max = *std::max_element(std::begin(max_elems), std::end(max_elems));
        float min = *std::min_element(std::begin(min_elems), std::end(min_elems));

        float glob_max = max;
        float glob_min = min;

        MPI_Allreduce(&max, &glob_max, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(&min, &glob_min, 1, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);

        if ((glob_max - glob_min) < threshold)
        {
            return true;
        }

        return false;
    }

    std::vector<std::pair<size_t, size_t>> get_neighbors(size_t y, size_t x) const
    {
        // Most common case in an inner cell with 4 neighbors
        std::vector<std::pair<size_t, size_t>> neighbors;
        neighbors.reserve(4);

        // Add left neighbor
        if (x != 0)
        {
            neighbors.emplace_back(y, x - 1);
        }

        // Add right neighbor
        if (x != _dimension - 1)
        {
            neighbors.emplace_back(y, x + 1);
        }

        // Add upper neighbor
        if (y != 0)
        {
            neighbors.emplace_back(y - 1, x);
        }

        // Add lower neighbor
        if (y != _dimension - 1)
        {
            neighbors.emplace_back(y + 1, x);
        }

        return neighbors;
    }
};

int main(int argc, char *argv[])
{
    // Init MPI
    MPI_Init(&argc, &argv);

    // Get the rank and size
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Divide to 4 ranks
    // 0 | 1
    // 2 | 3
    std::vector<std::pair<int, Border>> neighbors;
    std::function<float(size_t, size_t)> local_symmetric_dambreak;
    size_t domain_size = 100;
    size_t dambreak_offset = 20;

    switch (rank)
    {
    case 0:
    {
        neighbors = std::vector<std::pair<int, Border>>{{1, Border::Right}, {2, Border::Bottom}};
        local_symmetric_dambreak = [domain_size, dambreak_offset](size_t y, size_t x) -> float
        {
            if ((x < dambreak_offset) || (x > domain_size - dambreak_offset))
            {
                return 20.0;
            }
            return 10.0;
        };

        break;
    }
    case 1:
    {
        neighbors = std::vector<std::pair<int, Border>>{{0, Border::Left}, {3, Border::Bottom}};
        local_symmetric_dambreak = [domain_size, dambreak_offset](size_t y, size_t x) -> float
        {
            if ((x + 51 < dambreak_offset) || (x + 51 > domain_size - dambreak_offset))
            {
                return 20.0;
            }
            return 10.0;
        };

        break;
    }
    case 2:
    {
        neighbors = std::vector<std::pair<int, Border>>{{0, Border::Top}, {3, Border::Right}};
        local_symmetric_dambreak = [domain_size, dambreak_offset](size_t y, size_t x) -> float
        {
            if ((x < dambreak_offset) || (x > domain_size - dambreak_offset))
            {
                return 20.0;
            }
            return 10.0;
        };

        break;
    }
    case 3:
    {
        neighbors = std::vector<std::pair<int, Border>>{{2, Border::Left}, {1, Border::Top}};

        local_symmetric_dambreak = [domain_size, dambreak_offset](size_t y, size_t x) -> float
        {
            if ((x + 51 < dambreak_offset) || (x + 51 > domain_size - dambreak_offset))
            {
                return 20.0;
            }
            return 10.0;
        };

        break;
    }
    default:
    {
        throw std::runtime_error("Only with 4 MPI ranks please!");
    }
    }

    Domain d(domain_size / 2, local_symmetric_dambreak, std::move(neighbors), rank, size);
    unsigned long long p_up = d.simulate();

    if (rank == 0)
    {
        std::cout << rank << " required " << p_up << " updates until convergence" << std::endl;
    }

    MPI_Finalize();

    return 0;
}