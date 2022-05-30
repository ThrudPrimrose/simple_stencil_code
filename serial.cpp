#include <vector>
#include <functional>
#include <cstddef> // just in case for size_t
#include <algorithm>
#include <iostream>
#include <sstream>

constexpr float threshold = 0.1;
constexpr float viscosity_factor = 4.0;
constexpr size_t dam_offset = 20;
constexpr size_t default_domain_size = 100;
constexpr size_t print_frequency = 200;

std::function<float(size_t, size_t)> symmetric_dambreak = [](size_t y, size_t x) -> float
{
    if ((x < dam_offset) || (x + 1 > default_domain_size - dam_offset))
    {
        return 20.0;
    }
    return 10.0;
};

class Domain
{
private:
    size_t _dimension;
    std::vector<std::vector<float>> _domain;
    // Separate net updates with compute and update phases prevents data race conditions
    std::vector<std::vector<float>> _net_updates;
    unsigned long long _patch_updates;

public:
    Domain(size_t dimension, std::function<float(size_t, size_t)> initial_water_height)
        : _dimension(dimension), _domain(_dimension, std::vector<float>(_dimension, 0.0)),
          _net_updates(_dimension, std::vector<float>(_dimension, 0.0)), _patch_updates(0)
    {
        for (size_t y = 0; y < _dimension; y++)
        {
            for (size_t x = 0; x < _dimension; x++)
            {
                _domain[y][x] = initial_water_height(y, x);
            }
        }
    }

    Domain() : _dimension(default_domain_size), _domain(_dimension, std::vector<float>(_dimension, 0.0)),
               _net_updates(_dimension, std::vector<float>(_dimension, 0.0)), _patch_updates(0)
    {
        for (size_t y = 0; y < _dimension; y++)
        {
            for (size_t x = 0; x < _dimension; x++)
            {
                _domain[y][x] = symmetric_dambreak(x, y);
            }
        }
    }

    unsigned long long simulate()
    {
        volatile bool terminate_criteria_met = false;

        while (!terminate_criteria_met)
        {
            /*
            if (_patch_updates % 200 == 0)
            {
                print();
            }
            */

            compute_stencil();
            apply_influx();
            terminate_criteria_met = termination_criteria_fulfilled();
            _patch_updates += 1;
        }

        return _patch_updates;
    }

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
        std::cerr << ss.str() << std::endl;
    }

private:
    void apply_influx()
    {
        for (size_t y = 0; y < _dimension; y++)
        {
            for (size_t x = 0; x < _dimension; x++)
            {
                _domain[y][x] += _net_updates[y][x];
                _net_updates[y][x] = 0;
            }
        }
    }

    void compute_stencil()
    {
        // For evert cell with coordinates (y,x) compute the influx from neighbor cells
        // Apply reflecting boundary conditions
        for (size_t y = 0; y < _dimension; y++)
        {
            for (size_t x = 0; x < _dimension; x++)
            {
                std::vector<std::pair<size_t, size_t>> neighbors = get_neighbors(y, x);

                float cell_water = _domain[y][x];
                float update = 0.0;

                for (auto &coordinates : neighbors)
                {
                    float difference = _domain[coordinates.first][coordinates.second] - cell_water;

                    // If the difference is positive influx is positive otherwise negative
                    update += difference / viscosity_factor;
                }

                _net_updates[y][x] = update;
            }
        }
    }

    bool termination_criteria_fulfilled() const
    {
        std::vector<float> max_elems(_dimension, 0.0);
        std::vector<float> min_elems(_dimension, 0.0);

        for (size_t y = 0; y < _dimension; y++)
        {
            const std::vector<float> &row = _domain[y];
            auto mm = std::minmax_element(std::begin(row), std::end(row));
            min_elems[y] = *(mm.first);
            max_elems[y] = *(mm.second);
        }

        float max = *std::max_element(std::begin(max_elems), std::end(max_elems));
        float min = *std::min_element(std::begin(min_elems), std::end(min_elems));

        if ((max - min) < threshold)
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

        // Add lower neighbor
        if (y != 0)
        {
            neighbors.emplace_back(y - 1, x);
        }

        // Add upper neighbor
        if (y != _dimension - 1)
        {
            neighbors.emplace_back(y + 1, x);
        }

        return neighbors;
    }
};

int main()
{
    // std::vector<float> u(1000000000, 0.0);

    Domain d;
    unsigned long long p_up = d.simulate();

    std::cout << "Required " << p_up << " updated until convergence" << std::endl;

    return 0;
}