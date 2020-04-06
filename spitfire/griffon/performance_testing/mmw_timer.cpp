/*
 * mmw_timer.cpp
 *
 *  Created on: Apr 5, 2020
 *      Author: mahanse
 */

#include <iostream>
#include <chrono>
#include <cstdlib>
#include "combustion_kernels.h"

struct Timer
{
  std::chrono::high_resolution_clock::time_point tic_time;
  std::chrono::high_resolution_clock::time_point toc_time;

  void
  tic()
  {
    tic_time = std::chrono::high_resolution_clock::now();
  }

  double
  toc(const double multiplier = 1.e6, const int repeats = 1)
  {
    toc_time = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::duration<double>>(toc_time - tic_time).count() * multiplier / ((double)repeats);
  }
};

void print(const int ns, const int nz, const int rp, const std::string name, const double dcput)
{
  std::cout << rp << " repeats of \"" << name << "\" with nspec = " << ns << " and ngrid = " << nz
            << " completed, average of " << dcput << " us each\n";
}

using AtomMap = std::map<std::string, double>;

int main(int argc, char *argv[])
{

  int nz = atoi(argv[1]);
  int ns = atoi(argv[2]);
  int rp = atoi(argv[3]);

  //  const int nz = 256;   // number of grid points
  //  const int ns = 8;     // number of species
  //  const int rp = 1000;  // number of repeats

  griffon::CombustionKernels m;

  m.mechanism_add_element("H");
  m.mechanism_add_element("O");
  m.mechanism_add_element("C");
  m.mechanism_add_element("N");

  m.mechanism_set_ref_pressure(101325.);
  m.mechanism_set_ref_temperature(300.);

  std::vector<std::pair<std::string, AtomMap>> available_species;

  std::vector<std::string> elements = {{"H", "O", "C", "N"}};
  std::vector<std::string> species_names;
  for (const auto &e : elements)
  {
    const auto species_name = e;
    if (std::find(species_names.begin(), species_names.end(), species_name) == species_names.end())
    {
      species_names.push_back(species_name);
      available_species.push_back(std::make_pair(species_name, AtomMap({{e, 1.}})));
    }
  }
  for (const auto &e1 : elements)
  {
    for (const auto &e2 : elements)
    {
      {
        const auto species_name = e1 + e2;
        if (std::find(species_names.begin(), species_names.end(), species_name) == species_names.end())
        {
          species_names.push_back(species_name);
          available_species.push_back(std::make_pair(species_name, AtomMap({{e1, 1.}, {e2, 1.}})));
        }
      }
      {
        const auto species_name = e1 + "2" + e2;
        if (std::find(species_names.begin(), species_names.end(), species_name) == species_names.end())
        {
          species_names.push_back(species_name);
          available_species.push_back(std::make_pair(species_name, AtomMap({{e1, 2.}, {e2, 1.}})));
        }
      }
      {
        const auto species_name = e1 + e2 + "2";
        if (std::find(species_names.begin(), species_names.end(), species_name) == species_names.end())
        {
          species_names.push_back(species_name);
          available_species.push_back(std::make_pair(species_name, AtomMap({{e1, 1.}, {e2, 2.}})));
        }
      }
      {
        const auto species_name = e1 + "2" + e2 + "2";
        if (std::find(species_names.begin(), species_names.end(), species_name) == species_names.end())
        {
          species_names.push_back(species_name);
          available_species.push_back(std::make_pair(species_name, AtomMap({{e1, 2.}, {e2, 2.}})));
        }
      }
      {
        const auto species_name = e1 + "3" + e2;
        if (std::find(species_names.begin(), species_names.end(), species_name) == species_names.end())
        {
          species_names.push_back(species_name);
          available_species.push_back(std::make_pair(species_name, AtomMap({{e1, 3.}, {e2, 1.}})));
        }
      }
      {
        const auto species_name = e1 + "3" + e2 + "2";
        if (std::find(species_names.begin(), species_names.end(), species_name) == species_names.end())
        {
          species_names.push_back(species_name);
          available_species.push_back(std::make_pair(species_name, AtomMap({{e1, 3.}, {e2, 2.}})));
        }
      }
      {
        const auto species_name = e1 + e2 + "3";
        if (std::find(species_names.begin(), species_names.end(), species_name) == species_names.end())
        {
          species_names.push_back(species_name);
          available_species.push_back(std::make_pair(species_name, AtomMap({{e1, 1.}, {e2, 3.}})));
        }
      }
      {
        const auto species_name = e1 + "2" + e2 + "3";
        if (std::find(species_names.begin(), species_names.end(), species_name) == species_names.end())
        {
          species_names.push_back(species_name);
          available_species.push_back(std::make_pair(species_name, AtomMap({{e1, 2.}, {e2, 3.}})));
        }
      }
      {
        const auto species_name = e1 + "3" + e2 + "3";
        if (std::find(species_names.begin(), species_names.end(), species_name) == species_names.end())
        {
          species_names.push_back(species_name);
          available_species.push_back(std::make_pair(species_name, AtomMap({{e1, 3.}, {e2, 3.}})));
        }
      }
    }
  }
  for (const auto &e1 : elements)
  {
    for (const auto &e2 : elements)
    {
      for (const auto &e3 : elements)
      {
        {
          const auto species_name = e1 + e2 + e3;
          if (std::find(species_names.begin(), species_names.end(), species_name) == species_names.end())
          {
            species_names.push_back(species_name);
            available_species.push_back(std::make_pair(species_name, AtomMap({{e1, 1.}, {e2, 1.}, {e3, 1.}})));
          }
        }
        {
          const auto species_name = "2" + e1 + e2 + e3;
          if (std::find(species_names.begin(), species_names.end(), species_name) == species_names.end())
          {
            species_names.push_back(species_name);
            available_species.push_back(std::make_pair(species_name, AtomMap({{e1, 2.}, {e2, 1.}, {e3, 1.}})));
          }
        }
        {
          const auto species_name = e1 + "2" + e2 + e3;
          if (std::find(species_names.begin(), species_names.end(), species_name) == species_names.end())
          {
            species_names.push_back(species_name);
            available_species.push_back(std::make_pair(species_name, AtomMap({{e1, 1.}, {e2, 2.}, {e3, 1.}})));
          }
        }
        {
          const auto species_name = e1 + e2 + "2" + e3;
          if (std::find(species_names.begin(), species_names.end(), species_name) == species_names.end())
          {
            species_names.push_back(species_name);
            available_species.push_back(std::make_pair(species_name, AtomMap({{e1, 1.}, {e2, 1.}, {e3, 2.}})));
          }
        }
      }
    }
  }

  std::cout << species_names.size() << " species available\n";

  for (int is = 0; is < ns; ++is)
  {
    m.mechanism_add_species(available_species[is].first, available_species[is].second);
  }
  const double *minv = m.get_mechanism_data().phaseData.inverseMolecularWeights.data();

  // ----

  {
    double TY[ns * nz];
    for (int i = 0; i < ns * nz; ++i)
    {
      TY[i] = 1.;
    }
    double mmw[nz];
    double sum = 0.;

    Timer timer;
    timer.tic();
    for (int r = 0; r < rp; ++r)
    {
      for (int i = 0; i < nz; ++i)
      {
        double y[ns];
        m.extract_y(&TY[i * ns + 1], ns, y);
        mmw[i] = m.mixture_molecular_weight(y);
      }
      sum += mmw[0];
    }
    std::cout << "sum = " << sum << '\n';

    auto dt = timer.toc(1.e6, rp);
    print(ns, nz, rp, "zloop(extract_y, mechfxn)", dt);
  }

  {
    double TY[ns * nz];
    for (int i = 0; i < ns * nz; ++i)
    {
      TY[i] = 1.;
    }
    double mmw[nz];
    double sum = 0.;

    Timer timer;
    timer.tic();
    for (int r = 0; r < rp; ++r)
    {
      for (int i = 0; i < nz; ++i)
      {
        double y[ns];
        m.extract_y(&TY[i * ns + 1], ns, y);
        double value = 0.;
        for (int k = 0; k < ns; ++k)
        {
          value += minv[k] * y[k];
        }
        mmw[i] = 1. / value;
      }
      sum += mmw[0];
    }
    std::cout << "sum = " << sum << '\n';

    auto dt = timer.toc(1.e6, rp);
    print(ns, nz, rp, "zloop(extract_y,sloop)", dt);
  }

  {
    double TY[ns * nz];
    for (int i = 0; i < ns * nz; ++i)
    {
      TY[i] = 1.;
    }
    double mmw[nz];
    double sum = 0.;

    Timer timer;
    timer.tic();
    for (int r = 0; r < rp; ++r)
    {
      for (int i = 0; i < nz; ++i)
      {
        double value = 0.;
        double yn = 1.;
        const int offset = i * ns + 1;
        for (int k = 0; k < ns - 1; ++k)
        {
          const double yk = TY[offset + k];
          yn -= yk;
          value += minv[k] * yk;
        }
        mmw[i] = 1. / (value + minv[ns - 1] * yn);
      }
      sum += mmw[0];
    }
    std::cout << "sum = " << sum << '\n';

    auto dt = timer.toc(1.e6, rp);
    print(ns, nz, rp, "zloop(directTY,sloop)", dt);
  }

  {
    double TY[ns * nz];
    for (int i = 0; i < ns * nz; ++i)
    {
      TY[i] = 1.;
    }
    double yn[nz];
    double mmw[nz];
    double sum = 0.;

    Timer timer;
    timer.tic();
    for (int r = 0; r < rp; ++r)
    {

      for (int i = 0; i < nz; ++i)
      {
        yn[i] = 1.;
      }

      for (int k = 0; k < ns - 1; ++k)
      {
        const double minvk = minv[k];
        for (int i = 0; i < nz; ++i)
        {
          mmw[i] += minvk * TY[i * ns + 1 + k];
          yn[i] -= TY[i * ns + 1 + k];
        }
      }
      for (int i = 0; i < nz; ++i)
      {
        mmw[i] = 1. / (mmw[i] + yn[i] * minv[ns - 1]);
      }
      sum += mmw[0];
    }
    std::cout << "sum = " << sum << '\n';

    auto dt = timer.toc(1.e6, rp);
    print(ns, nz, rp, "sloop(zloop,directTY)", dt);
  }

  {
    double TY[ns * nz];
    for (int i = 0; i < ns * nz; ++i)
    {
      TY[i] = 1.;
    }
    double y[nz * ns];
    for (int i = 0; i < nz; ++i)
    {
      const int offsetTY = i * ns + 1;
      const int offsetY = i;
      y[offsetY + (ns - 1) * nz] = 1.;
      for (int k = 0; k < ns - 1; ++k)
      {
        y[offsetY + k * nz] = TY[offsetTY + k];
        y[offsetY + (ns - 1) * nz] -= y[offsetY + k * nz];
      }
    }
    double mmw[nz];
    double sum = 0.;

    Timer timer;
    timer.tic();
    for (int r = 0; r < rp; ++r)
    {

      for (int k = 0; k < ns; ++k)
      {
        const double minvk = minv[k];
        const int offset = k * nz;
        for (int i = 0; i < nz; ++i)
        {
          mmw[i] += minvk * y[offset + i];
        }
      }
      for (int i = 0; i < nz; ++i)
      {
        mmw[i] = 1. / mmw[i];
      }
      sum += mmw[0];
    }
    std::cout << "sum = " << sum << '\n';

    auto dt = timer.toc(1.e6, rp);
    print(ns, nz, rp, "sloop(zloop,transposedY) no transpose", dt);
  }

  {
    double TY[ns * nz];
    for (int i = 0; i < ns * nz; ++i)
    {
      TY[i] = 1.;
    }
    double y[nz * ns];
    double mmw[nz];
    double sum = 0.;

    Timer timer;
    timer.tic();
    for (int r = 0; r < rp; ++r)
    {
      for (int i = 0; i < nz; ++i)
      {
        const int offsetTY = i * ns + 1;
        const int offsetY = i;
        y[offsetY + (ns - 1) * nz] = 1.;
        for (int k = 0; k < ns - 1; ++k)
        {
          y[offsetY + k * nz] = TY[offsetTY + k];
          y[offsetY + (ns - 1) * nz] -= y[offsetY + k * nz];
        }
      }
      sum += y[0];
    }
    std::cout << "sum = " << sum << '\n';

    auto dt = timer.toc(1.e6, rp);
    print(ns, nz, rp, "TY transpose", dt);
  }

  {
    double TY[ns * nz];
    for (int i = 0; i < ns * nz; ++i)
    {
      TY[i] = 1.;
    }
    double y[nz * ns];
    double mmw[nz];
    double sum = 0.;

    Timer timer;
    timer.tic();
    for (int r = 0; r < rp; ++r)
    {
      for (int i = 0; i < nz; ++i)
      {
        const int offsetTY = i * ns + 1;
        const int offsetY = i;
        y[offsetY + (ns - 1) * nz] = 1.;
        for (int k = 0; k < ns - 1; ++k)
        {
          y[offsetY + k * nz] = TY[offsetTY + k];
          y[offsetY + (ns - 1) * nz] -= y[offsetY + k * nz];
        }
      }
      for (int k = 0; k < ns; ++k)
      {
        const double minvk = minv[k];
        const int offset = k * nz;
        for (int i = 0; i < nz; ++i)
        {
          mmw[i] += minvk * y[offset + i];
        }
      }
      for (int i = 0; i < nz; ++i)
      {
        mmw[i] = 1. / mmw[i];
      }
      sum += mmw[0];
    }
    std::cout << "sum = " << sum << '\n';

    auto dt = timer.toc(1.e6, rp);
    print(ns, nz, rp, "sloop(zloop,transposedY) FULL", dt);
  }
}
