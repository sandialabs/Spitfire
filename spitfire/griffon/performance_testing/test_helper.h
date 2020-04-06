#include <iostream>
#include <stdexcept>
#include <sstream>
#include <chrono>
#include "combustion_kernels.h"

namespace griffon
{
struct Timer
{
    std::chrono::high_resolution_clock::time_point tic_time;
    std::chrono::high_resolution_clock::time_point toc_time;

    void
    tic();

    double
    toc(const double multiplier = 1.e6, const int repeats = 1);
};

void print_result(const int ns, const int nz, const int rp, const std::string name, const double dcput);
void print_result(const int ns, const int nr, const int nz, const int rp, const std::string name, const double dcput);

griffon::CombustionKernels make_fake_mechanism(const int ns, const int nr = 0);

}