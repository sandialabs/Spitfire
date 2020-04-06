#include "test_helper.h"
#include <cstdlib>

int main(int argc, char *argv[])
{

  int nz = atoi(argv[1]); // number of grid points
  int ns = atoi(argv[2]); // number of species
  int rp = atoi(argv[3]); // number of repeats

  griffon::CombustionKernels m = griffon::make_fake_mechanism(ns);
  const auto minv = m.get_mechanism_data().phaseData.inverseMolecularWeights.data();

  // ----

  {
    double TY[ns * nz];
    for (int i = 0; i < ns * nz; ++i)
    {
      TY[i] = 1.;
    }
    double mmw[nz];
    double sum = 0.;

    griffon::Timer timer;
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
    griffon::print_result(ns, nz, rp, "zloop(extract_y, mechfxn)", dt);
  }

  {
    double TY[ns * nz];
    for (int i = 0; i < ns * nz; ++i)
    {
      TY[i] = 1.;
    }
    double mmw[nz];
    double sum = 0.;

    griffon::Timer timer;
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
    griffon::print_result(ns, nz, rp, "zloop(extract_y,sloop)", dt);
  }

  {
    double TY[ns * nz];
    for (int i = 0; i < ns * nz; ++i)
    {
      TY[i] = 1.;
    }
    double mmw[nz];
    double sum = 0.;

    griffon::Timer timer;
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
    griffon::print_result(ns, nz, rp, "zloop(directTY,sloop)", dt);
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

    griffon::Timer timer;
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
    griffon::print_result(ns, nz, rp, "sloop(zloop,directTY)", dt);
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

    griffon::Timer timer;
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
    griffon::print_result(ns, nz, rp, "sloop(zloop,transposedY) no transpose", dt);
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

    griffon::Timer timer;
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
    griffon::print_result(ns, nz, rp, "TY transpose", dt);
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

    griffon::Timer timer;
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
    griffon::print_result(ns, nz, rp, "sloop(zloop,transposedY) FULL", dt);
  }
}
