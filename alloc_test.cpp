#include <vector>
#include <cstddef>
#include <iostream>
#include <thread>

int main()
{
    constexpr size_t alloc_size = 1000000000;
    std::vector<unsigned int> v(alloc_size, 0);
    size_t allocsize = sizeof(unsigned int) * alloc_size;
    std::cout << "Allocated roughly ~"
              << (allocsize / 1000000)
              << " mb, random entry:"
              //<< static_cast<unsigned int>(v[alloc_size / 2])
              << std::endl;
    using namespace std::chrono_literals;
    std::this_thread::sleep_for(2000ms);
    return 0;
}