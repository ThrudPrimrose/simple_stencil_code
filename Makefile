CXX = c++
MPICXX = mpicxx
CXX_FLAGS = -std=c++17 -Wall -Wextra -Wno-unused -march=native -g -O2 -fopenmp

all:  serial omp hybrid	async

serial: serial.cpp
	$(CXX) $(CXX_FLAGS) serial.cpp -o serial

omp: omppar.cpp
	$(CXX) $(CXX_FLAGS) omppar.cpp -o omppar

hybrid: hybrid.cpp
	$(MPICXX) $(CXX_FLAGS) hybrid.cpp -o hybrid

async: hybrid_async_communication.cpp
	$(MPICXX) $(CXX_FLAGS) hybrid_async_communication.cpp -o async

clean:
	rm -f serial hybrid omppar async