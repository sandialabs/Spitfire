#include "test_helper.h"
#include <cstdlib>

int main(int argc, char *argv[])
{

    int nz = atoi(argv[1]); // number of grid points
    int ns = atoi(argv[2]); // number of species
    int nr = atoi(argv[3]); // number of reactions
    int rp = atoi(argv[4]); // number of repeats

    griffon::CombustionKernels m = griffon::make_fake_mechanism(ns, nr);

    double state[ns * nz];
    double oxystate[ns];
    double fuelstate[ns];
    double cmajor[nz];
    double csub[nz];
    double csup[nz];
    double mcoeff[nz];
    double ncoeff[nz];
    double chi[nz];
    double rhs[ns * nz];

    const double pressure = 101325.;

    const bool adiabatic = true;
    const bool include_enthalpy_flux = true;
    const bool include_variable_cp = true;
    const bool use_scaled_heat_loss = true;

    {
        griffon::Timer timer;
        timer.tic();
        for (int r = 0; r < rp; ++r)
        {
            m.flamelet_rhs(state, pressure, oxystate, fuelstate,
                           adiabatic, nullptr, nullptr,
                           nullptr, nullptr, nz, cmajor,
                           csub, csup, mcoeff, ncoeff,
                           chi, include_enthalpy_flux, include_variable_cp,
                           use_scaled_heat_loss, rhs);
        }
        std::cout << "rhs[0] = " << rhs[0] << '\n';
        auto dt = timer.toc(1.e6, rp);
        griffon::print_result(ns, nr, nz, rp, "flamelet_rhs", dt);
    }
    {
        griffon::Timer timer;
        timer.tic();
        for (int r = 0; r < rp; ++r)
        {
            m.flamelet_rhs_test1(state, pressure, oxystate, fuelstate,
                                 adiabatic, nullptr, nullptr,
                                 nullptr, nullptr, nz, cmajor,
                                 csub, csup, mcoeff, ncoeff,
                                 chi, include_enthalpy_flux, include_variable_cp,
                                 use_scaled_heat_loss, rhs);
        }
        std::cout << "rhs[0] = " << rhs[0] << '\n';
        auto dt = timer.toc(1.e6, rp);
        griffon::print_result(ns, nr, nz, rp, "flamelet_rhs_test1", dt);
    }
}