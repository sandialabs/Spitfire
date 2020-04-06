/*
 * mmw_timer.cpp
 *
 *  Created on: Apr 5, 2020
 *      Author: mahanse
 */

#include <iostream>
#include <chrono>
#include "combustion_kernels.h"

struct Timer
{
  std::chrono::high_resolution_clock::time_point tic_time;
  std::chrono::high_resolution_clock::time_point toc_time;

  void
  tic ()
  {
    tic_time = std::chrono::high_resolution_clock::now ();
  }

  double
  toc (const double multiplier = 1.e6, const int repeats = 1)
  {
    toc_time = std::chrono::high_resolution_clock::now ();
    return std::chrono::duration_cast<std::chrono::duration<double>> (toc_time - tic_time).count () * multiplier
        / ((double) repeats);
  }
};

void
print (const int ns, const int nz, const int rp, const std::string name, const double dcput)
{
  std::cout << rp << " repeats of " << name << " with nspec = " << ns << " and ngrid = " << nz
      << " completed, average of " << dcput << " us each\n";
}

using AtomMap = std::map<std::string, double>;

int
main ()
{

  const int ns = 8;     // number of species
  const int nz = 256;   // number of grid points
  const int rp = 1000;  // number of repeats

  griffon::CombustionKernels m;

  m.mechanism_add_element ("H");
  m.mechanism_add_element ("O");

  m.mechanism_set_ref_pressure(101325.);
  m.mechanism_set_ref_temperature(300.);

  m.mechanism_add_species("H", AtomMap({{"H", 1.}}));
  m.mechanism_add_species("O", AtomMap({{"O", 1.}}));
  m.mechanism_add_species("H2", AtomMap({{"H", 2.}}));
  m.mechanism_add_species("O2", AtomMap({{"O", 2.}}));
  m.mechanism_add_species("HO", AtomMap({{"H", 1.}, {"O", 1.}}));
  m.mechanism_add_species("HO2", AtomMap({{"H", 1.}, {"O", 2.}}));
  m.mechanism_add_species("H2O", AtomMap({{"H", 2.}, {"O", 1.}}));
  m.mechanism_add_species("H2O2", AtomMap({{"H", 2.}, {"O", 2.}}));


  std::string name1 = "mmw-1";

  Timer timer;

  timer.tic ();

  auto dt = timer.toc (1.e6, rp);

  print (ns, nz, rp, name1, dt);
}

