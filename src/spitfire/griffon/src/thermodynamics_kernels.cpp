/* 
 * Spitfire - a Python-C++ library for building tabulated chemistry models and solving differential equations                    
 * Copyright 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS). 
 *                      
 * You should have received a copy of the 3-clause BSD License                                        
 * along with this program.  If not, see <https://opensource.org/licenses/BSD-3-Clause>.   
 *                   
 * Questions? Contact Mike Hansen (mahanse@sandia.gov)    
 */

#include "combustion_kernels.h"
#include "blas_lapack_kernels.h"
#include <cmath>

namespace griffon
{

void CombustionKernels::mole_fractions(const double *y, double *x) const
{
  const int nSpec = mechanismData.phaseData.nSpecies;
  const auto invMolecularWeights = mechanismData.phaseData.inverseMolecularWeights.data();
  const double mmw = mixture_molecular_weight(y);
  for (int i = 0; i < nSpec; ++i)
  {
    x[i] = y[i] * mmw * invMolecularWeights[i];
  }
}

double
CombustionKernels::ideal_gas_density(const double &pressure, const double &temperature, const double *y) const
{
  double rho;
  ideal_gas_density(pressure, temperature, mixture_molecular_weight(y), &rho);
  return rho;
}

double
CombustionKernels::ideal_gas_pressure(const double &density, const double &temperature, const double *y) const
{
  double p;
  ideal_gas_pressure(density, temperature, mixture_molecular_weight(y), &p);
  return p;
}

void CombustionKernels::cp_mix_and_species(const double &temperature, const double *y, double *out_cpmix,
                                           double *out_cpspecies) const
{
  const int nSpec = mechanismData.phaseData.nSpecies;
  const auto invMolecularWeights = mechanismData.phaseData.inverseMolecularWeights.data();
  const double t = temperature;
  *out_cpmix = 0.;
  for (int i = 0; i < nSpec; ++i)
  {
    const auto polyType = mechanismData.heatCapacityData.types[i];
    const auto &c = mechanismData.heatCapacityData.coefficients[i];
    const auto &c9 = mechanismData.heatCapacityData.nasa9Coefficients[i];
    const double maxT = mechanismData.heatCapacityData.maxTemperatures[i];
    const double minT = mechanismData.heatCapacityData.minTemperatures[i];

    switch (polyType)
    {
    case CpType::CONST:
      out_cpspecies[i] = invMolecularWeights[i] * c[3];
      *out_cpmix += y[i] * out_cpspecies[i];
      break;
    case CpType::NASA7:
      if (t <= c[0] && t >= minT)
      {
        out_cpspecies[i] = invMolecularWeights[i] * (c[8] + t * (2. * c[9] + t * (6. * c[10] + t * (12. * c[11] + 20. * t * c[12]))));
        *out_cpmix += y[i] * out_cpspecies[i];
        break;
      }
      else if (t > c[0] && t <= maxT)
      {
        out_cpspecies[i] = invMolecularWeights[i] * (c[1] + t * (2. * c[2] + t * (6. * c[3] + t * (12. * c[4] + 20. * t * c[5]))));
        *out_cpmix += y[i] * out_cpspecies[i];
        break;
      }
      else if (t < minT)
      {
        out_cpspecies[i] = invMolecularWeights[i] * (c[8] + minT * (2. * c[9] + minT * (6. * c[10] + minT * (12. * c[11] + 20. * minT * c[12]))));
        *out_cpmix += y[i] * out_cpspecies[i];
        break;
      }
      else
      {
        out_cpspecies[i] = invMolecularWeights[i] * (c[1] + maxT * (2. * c[2] + maxT * (6. * c[3] + maxT * (12. * c[4] + 20. * maxT * c[5]))));
        *out_cpmix += y[i] * out_cpspecies[i];
        break;
      }
    case CpType::NASA9:
    {
      const auto nregions = static_cast<int>(c9[0]);
      if(t < minT)
      {
        const double invminT = 1. / minT;
        const auto *a = &(c9[3]);
        out_cpspecies[i] = invMolecularWeights[i] * (invminT * (a[1] + invminT * a[0]) + a[2] + minT * (a[3] + minT * (a[4] + minT * (a[5] + minT * a[6]))));
        *out_cpmix += y[i] * out_cpspecies[i];
      }
      else if(t > maxT)
      {
        const double invmaxT = 1. / maxT;
        const auto *a = &(c9[1 + (nregions - 1) * 11 + 2]);
        out_cpspecies[i] = invMolecularWeights[i] * (invmaxT * (a[1] + invmaxT * a[0]) + a[2] + maxT * (a[3] + maxT * (a[4] + maxT * (a[5] + maxT * a[6]))));
        *out_cpmix += y[i] * out_cpspecies[i];
      }
      else
      {  
        const double invT = 1. / t;
        for(int k=0; k<nregions; ++k)
        {
          const int region_start_idx = 1 + k * 11;
          if(t >= c9[region_start_idx] && t < c9[region_start_idx + 1])
          {
            const auto *a = &(c9[region_start_idx + 2]);
            out_cpspecies[i] = invMolecularWeights[i] * (invT * (a[1] + invT * a[0]) + a[2] + t * (a[3] + t * (a[4] + t * (a[5] + t * a[6]))));
            *out_cpmix += y[i] * out_cpspecies[i];
            break;
          }
        }
      }
      break;
    }
    default:
    {
      break;
    }
    }
  }
}

double
CombustionKernels::cp_mix(const double &temperature, const double *y) const
{
  const int nSpec = mechanismData.phaseData.nSpecies;
  double cpmix = 0.;
  double cpspecies[nSpec];
  cp_mix_and_species(temperature, y, &cpmix, cpspecies);
  return cpmix;
}

void CombustionKernels::species_cp(const double &temperature, double *out_cpspecies) const
{
  double garbage;
  cp_mix_and_species(temperature, out_cpspecies, &garbage, out_cpspecies);
}

double
CombustionKernels::cv_mix(const double &temperature, const double *y) const
{
  const double gasConstant = mechanismData.phaseData.Ru;
  return cp_mix(temperature, y) - gasConstant / mixture_molecular_weight(y);
}

void CombustionKernels::species_cv(const double &temperature, double *out_cvspecies) const
{
  const int nSpec = mechanismData.phaseData.nSpecies;
  const auto invMolecularWeights = mechanismData.phaseData.inverseMolecularWeights.data();
  const double gasConstant = mechanismData.phaseData.Ru;
  double null;
  cp_mix_and_species(temperature, out_cvspecies, &null, out_cvspecies);
  for (int i = 0; i < nSpec; ++i)
  {
    out_cvspecies[i] -= gasConstant * invMolecularWeights[i];
  }
}

void CombustionKernels::cv_mix_and_species(const double &temperature, const double *y, const double &mmw,
                                           double *out_cvmix, double *out_cvspecies) const
{
  const int nSpec = mechanismData.phaseData.nSpecies;
  const auto invMolecularWeights = mechanismData.phaseData.inverseMolecularWeights.data();
  const double gasConstant = mechanismData.phaseData.Ru;
  cp_mix_and_species(temperature, y, out_cvmix, out_cvspecies);
  *out_cvmix -= gasConstant / mmw;
  for (int i = 0; i < nSpec; ++i)
  {
    out_cvspecies[i] -= gasConstant * invMolecularWeights[i];
  }
}

void CombustionKernels::cp_sens_T(const double &temperature, const double *y, double *out_cpmixsens,
                                  double *out_cpspeciessens) const
{
  const int nSpec = mechanismData.phaseData.nSpecies;
  const auto invMolecularWeights = mechanismData.phaseData.inverseMolecularWeights.data();
  const double t = temperature;
  *out_cpmixsens = 0.;
  for (int i = 0; i < nSpec; ++i)
  {
    const auto polyType = mechanismData.heatCapacityData.types[i];
    const auto &c = mechanismData.heatCapacityData.coefficients[i];
    const auto &c9 = mechanismData.heatCapacityData.nasa9Coefficients[i];
    const double minT = mechanismData.heatCapacityData.minTemperatures[i];
    const double maxT = mechanismData.heatCapacityData.maxTemperatures[i];

    switch (polyType)
    {
    case CpType::CONST:
      out_cpspeciessens[i] = 0.;
      *out_cpmixsens = 0.;
      break;
    case CpType::NASA7:
      if (t <= c[0] && t >= minT)
      {
        out_cpspeciessens[i] = invMolecularWeights[i] * ((2. * c[9] + t * (12. * c[10] + t * (36. * c[11] + 80. * t * c[12]))));
        *out_cpmixsens += y[i] * out_cpspeciessens[i];
        break;
      }
      else if (t > c[0] && t <= maxT)
      {
        out_cpspeciessens[i] = invMolecularWeights[i] * ((2. * c[2] + t * (12. * c[3] + t * (36. * c[4] + 80. * t * c[5]))));
        *out_cpmixsens += y[i] * out_cpspeciessens[i];
        break;
      }
      else if (t < minT)
      {
        out_cpspeciessens[i] = 0.;
        *out_cpmixsens += 0.;
        break;
      }
      else
      {
        out_cpspeciessens[i] = 0.;
        *out_cpmixsens += 0.;
        break;
      }
    case CpType::NASA9:
    {
      const auto nregions = static_cast<int>(c9[0]);
      if(t < minT || t > maxT)
      {
        out_cpspeciessens[i] = 0.;
        *out_cpmixsens += 0.;
      }
      else
      {
        const double invT = 1. / t;
        for(int k=0; k<nregions; ++k)
        {
          const int region_start_idx = 1 + k * 11;
          if(t >= c9[region_start_idx] && t < c9[region_start_idx + 1])
          {
            const auto *a = &(c9[region_start_idx + 2]);
            out_cpspeciessens[i] = invMolecularWeights[i] * (-invT * invT * (a[1] + invT * 2.0 * a[0]) + a[3] + t * (2.0 * a[4] + t * (3.0 * a[5] + t * 4.0 * a[6])));
            *out_cpmixsens += y[i] * out_cpspeciessens[i];
            break;
          }
        }
      }
      break;
    }
    default:
    {
      break;
    }
    }
  }
}

void CombustionKernels::species_enthalpies(const double &temperature, double *out_enthalpies) const
{
  const int nSpec = mechanismData.phaseData.nSpecies;
  const auto invMolecularWeights = mechanismData.phaseData.inverseMolecularWeights.data();
  const double temp = temperature;

  for (int i = 0; i < nSpec; ++i)
  {
    const auto polyType = mechanismData.heatCapacityData.types[i];
    const auto &c = mechanismData.heatCapacityData.coefficients[i];
    const auto &c9 = mechanismData.heatCapacityData.nasa9Coefficients[i];
    const double minT = mechanismData.heatCapacityData.minTemperatures[i];
    const double maxT = mechanismData.heatCapacityData.maxTemperatures[i];

    switch (polyType)
    {
    case CpType::CONST:
      out_enthalpies[i] = invMolecularWeights[i] * (c[1] + c[3] * (temp - c[0]));
      break;
    case CpType::NASA7:
      if (temp <= c[0] && temp >= minT)
      {
        out_enthalpies[i] =
            invMolecularWeights[i] * (c[13] + temp * (c[8] + temp * (c[9] + temp * (2. * c[10] + temp * (3. * c[11] + temp * 4. * c[12])))));
        break;
      }
      else if (temp > c[0] && temp <= maxT)
      {
        out_enthalpies[i] = invMolecularWeights[i] * (c[6] + temp * (c[1] + temp * (c[2] + temp * (2. * c[3] + temp * (3. * c[4] + temp * 4. * c[5])))));
        break;
      }
      else if (temp < minT)
      {
        out_enthalpies[i] =
            invMolecularWeights[i] * (c[13] + c[8] * temp + minT * (2. * c[9] * temp + minT * (3. * 2. * c[10] * temp - c[9] + minT * (4. * 3. * c[11] * temp - 2. * 2. * c[10] + minT * (5. * 4. * c[12] * temp - 3. * 3. * c[11] + minT * -4. * 4. * c[12])))));
        break;
      }
      else
      {
        out_enthalpies[i] =
            invMolecularWeights[i] * (c[6] + c[1] * temp + maxT * (2. * c[2] * temp + maxT * (3. * 2. * c[3] * temp - c[2] + maxT * (4. * 3. * c[4] * temp - 2. * 2. * c[3] + maxT * (5. * 4. * c[5] * temp - 3. * 3. * c[4] + maxT * -4. * 4. * c[5])))));
        break;
      }
    case CpType::NASA9:
    {
      const auto nregions = static_cast<int>(c9[0]);
      if(temp < minT)
      {
        const double invT = 1. / minT;
        const double logT = std::log(minT);
        const int region_start_idx = 1;
        const auto *a = &(c9[region_start_idx + 2]);
        const double hmin = invMolecularWeights[i] * minT * (invT * (a[7] + logT * a[1] - a[0] * invT) + a[2] + minT * (0.5 * a[3] + minT * (0.3333333333333333 * a[4] + minT * (0.25 * a[5] + minT * 0.2 * a[6]))));
        const double cpmin = invMolecularWeights[i] * (invT * (a[1] + invT * a[0]) + a[2] + minT * (a[3] + minT * (a[4] + minT * (a[5] + minT * a[6]))));
        out_enthalpies[i] = hmin + cpmin * (temp - minT);
      }
      else if(temp > maxT)
      {
        const double invT = 1. / maxT;
        const double logT = std::log(maxT);
        const int region_start_idx = 1 + (nregions - 1) * 11;
        const auto *a = &(c9[region_start_idx + 2]);
        const double hmax = invMolecularWeights[i] * maxT * (invT * (a[7] + logT * a[1] - a[0] * invT) + a[2] + maxT * (0.5 * a[3] + maxT * (0.3333333333333333 * a[4] + maxT * (0.25 * a[5] + maxT * 0.2 * a[6]))));
        const double cpmax = invMolecularWeights[i] * (invT * (a[1] + invT * a[0]) + a[2] + maxT * (a[3] + maxT * (a[4] + maxT * (a[5] + maxT * a[6]))));
        out_enthalpies[i] = hmax + cpmax * (temp - maxT);
      }
      else
      {
        const double invT = 1. / temp;
        const double logT = std::log(temp);
        for(int k=0; k<nregions; ++k)
        {
          const int region_start_idx = 1 + k * 11;
          if(temp >= c9[region_start_idx] && temp < c9[region_start_idx + 1])
          {
            const auto *a = &(c9[region_start_idx + 2]);
            out_enthalpies[i] = invMolecularWeights[i] * temp * (invT * (a[7] + logT * a[1] - a[0] * invT) + a[2] + temp * (0.5 * a[3] + temp * (0.3333333333333333 * a[4] + temp * (0.25 * a[5] + temp * 0.2 * a[6]))));
            break;
          }
        }
      }
      break;
    }
    default:
    {
      break;
    }
    }
  }
}

void CombustionKernels::species_energies(const double &temperature, double *out_energies) const
{
  const int nSpec = mechanismData.phaseData.nSpecies;
  const auto invMolecularWeights = mechanismData.phaseData.inverseMolecularWeights.data();
  const double gasConstant = mechanismData.phaseData.Ru;
  species_enthalpies(temperature, out_energies);
  const double RT = gasConstant * temperature;
  for (int i = 0; i < nSpec; ++i)
  {
    out_energies[i] -= RT * invMolecularWeights[i];
  }
}

double
CombustionKernels::enthalpy_mix(const double &temperature, const double *y) const
{
  const int nSpec = mechanismData.phaseData.nSpecies;
  double hi[nSpec];
  species_enthalpies(temperature, hi);
  return blas::inner_product(nSpec, hi, y);
}

double
CombustionKernels::energy_mix(const double &temperature, const double *y) const
{
  const int nSpec = mechanismData.phaseData.nSpecies;
  double ei[nSpec];
  species_energies(temperature, ei);
  return blas::inner_product(nSpec, ei, y);
}
} // namespace griffon
