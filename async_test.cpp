#include <mpi.h>
#include <vector>
#include <iostream>

int main(int argc, char *argv[])
{
    // Init MPI
    MPI_Init(&argc, &argv);

    // Get the rank and size
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    constexpr size_t vsize = 1000;
    std::vector<float> v(vsize, 1.0);
    std::vector<float> rv(vsize, 0.0);

    if (rank == 0)
    {
        MPI_Request sendreq;
        MPI_Isend(v.data(), vsize, MPI_FLOAT, 1, 0, MPI_COMM_WORLD, &sendreq);
    }
    else
    {
        MPI_Request recvreq;
        MPI_Status recvstat;
        volatile int flag = 0;

        /*
        while (flag == 0)
        {
            int internal_flag = 0;
            // MPI_Test(&recvreq, &internal_flag, &recvstat);
            MPI_Iprobe(0, 0, MPI_COMM_WORLD, &internal_flag, NULL);
            flag = internal_flag;

            if (flag)
            {
                MPI_Recv(rv.data(), vsize, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, NULL);
                std::cout << rv[0] << std::endl;
            }

        }
        */

        MPI_Request waitreq;
        MPI_Irecv(rv.data(), vsize, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &waitreq);
        MPI_Wait(&waitreq, NULL);
        std::cout << rv[0] << std::endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
}