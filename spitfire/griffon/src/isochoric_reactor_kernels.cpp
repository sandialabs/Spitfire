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
#include <numeric>

namespace griffon {

void CombustionKernels::chem_rhs_isochoric(const double &rho, const double &cv, const double *e, const double *w, double *out_rhs) const {
  const int nSpec = mechanismData.phaseData.nSpecies;
  out_rhs[0] = 0.;
  out_rhs[1] = -griffon::blas::inner_product(nSpec, w, e) / (rho * cv);
  griffon::blas::copy_vector(nSpec - 1, &out_rhs[2], w);
  griffon::blas::scale_vector(nSpec - 1, &out_rhs[2], 1. / rho);
}

void CombustionKernels::heat_rhs_isochoric(const double &temperature, const double &rho, const double &cv, const double &fluidTemperature,
    const double &surfTemperature, const double &hConv, const double &epsRad, const double &surfaceAreaOverVolume,
    double *out_heatTransferRate) const {
  *out_heatTransferRate = surfaceAreaOverVolume / (rho * cv)
      * (hConv * (fluidTemperature - temperature)
          + epsRad * 5.67e-8
              * (surfTemperature * surfTemperature * surfTemperature * surfTemperature
                  - temperature * temperature * temperature * temperature));
}

void CombustionKernels::mass_rhs_isochoric(const double *y, const double *energies, const double *inflowEnergies, const double &rho,
    const double &inflowRho, const double &pressure, const double &inflowPressure, const double &cv, const double *inflowY,
    const double &tau, double *out_rhs) const {
  const int nSpec = mechanismData.phaseData.nSpecies;
  out_rhs[0] = inflowRho - rho;

  double einme[nSpec];
  griffon::blas::copy_vector(nSpec, einme, inflowEnergies);
  griffon::blas::vector_plus_scaled_vector(nSpec, einme, -1., energies);
  out_rhs[1] = (griffon::blas::inner_product(nSpec, einme, inflowY) * inflowRho) / (rho * cv);

  griffon::blas::copy_vector(nSpec - 1, &out_rhs[2], inflowY);
  griffon::blas::vector_plus_scaled_vector(nSpec - 1, &out_rhs[2], -1., y);
  griffon::blas::scale_vector(nSpec - 1, &out_rhs[2], inflowRho / rho);

  griffon::blas::scale_vector(nSpec + 1, out_rhs, 1. / tau);
}

void CombustionKernels::chem_jacexactdense_isochoric(const double &temperature, const double *y, const double &rho, const double &cv,
    const double *cvi, const double &cvsensT, const double *e, const double *w, const double *wsens, double *out_rhs,
    double *out_jac) const {
  const int nSpec = mechanismData.phaseData.nSpecies;
  // out_primJac is an (n+1) x (n+1) column-major array
  // row i col j is i + j * (n+1)

  chem_rhs_isochoric(rho, cv, e, w, out_rhs);

  const double invRho = 1. / rho;
  const double invCv = 1. / cv;
  const double invRhoCv = 1. / (rho * cv);

  // rho column
  out_jac[0] = 0.;
  out_jac[1] = -invRhoCv * griffon::blas::inner_product(nSpec, wsens, e) - invRho * out_rhs[1];
  griffon::blas::copy_vector(nSpec - 1, &out_jac[2], wsens);
  griffon::blas::vector_plus_scaled_vector(nSpec - 1, &out_jac[2], -invRho, w);
  griffon::blas::scale_vector(nSpec - 1, &out_jac[2], invRho);

  // T column
  const std::size_t firstRow = nSpec + 1;
  out_jac[firstRow] = 0.;
  out_jac[firstRow + 1] = -invCv
      * (invRho * (griffon::blas::inner_product(nSpec, &wsens[nSpec + 1], e) + griffon::blas::inner_product(nSpec, w, cvi))
          + out_rhs[1] * cvsensT);
  griffon::blas::copy_vector(nSpec - 1, &out_jac[firstRow + 2], &wsens[nSpec + 1]);
  griffon::blas::scale_vector(nSpec - 1, &out_jac[firstRow + 2], invRho);

  // Yk column
  const double cvn = cvi[nSpec - 1];
  for (std::size_t k = 0; k < nSpec - 1; ++k) {
    const std::size_t firstRowSpec = (2 + k) * (nSpec + 1);
    out_jac[firstRowSpec] = 0.;
    out_jac[firstRowSpec + 1] = -invRhoCv * griffon::blas::inner_product(nSpec, &wsens[(2 + k) * (nSpec + 1)], e)
        - out_rhs[1] * (cvi[k] - cvn) * invCv;
    griffon::blas::copy_vector(nSpec - 1, &out_jac[firstRowSpec + 2], &wsens[(2 + k) * (nSpec + 1)]);
    griffon::blas::scale_vector(nSpec - 1, &out_jac[firstRowSpec + 2], invRho);
  }
}

void CombustionKernels::mass_jacexactdense_isochoric(const double &pressure, const double &inflowPressure, const double &temperature,
    const double *y, const double &rho, const double &inflowRho, const double &cv, const double &cvsensT, const double *cvi,
    const double *energies, const double *inflowEnergies, const double &inflowTemperature, const double *inflowY, const double &tau,
    double *out_rhs, double *out_jac) const {
  const int nSpec = mechanismData.phaseData.nSpecies;
  // out_primJac is an (n+1) x (n+1) column-major array
  // row i col j is i + j * (n+1)

  mass_rhs_isochoric(y, energies, inflowEnergies, rho, inflowRho, pressure, inflowPressure, cv, inflowY, tau, out_rhs);

  const double invRho = 1. / rho;
  const double invCv = 1. / cv;
  const double invTau = 1. / tau;

  // rho column
  out_jac[0] = -invTau;
  griffon::blas::copy_vector(nSpec, &out_jac[1], &out_rhs[1]);
  griffon::blas::scale_vector(nSpec, &out_jac[1], -invRho);

  // T column
  const std::size_t firstRow = nSpec + 1;
  out_jac[firstRow] = 0.;
  out_jac[firstRow + 1] = -invCv
      * (invTau * invRho * (inflowRho * griffon::blas::inner_product(nSpec, inflowY, cvi)) + cvsensT * out_rhs[1]);
  double *Tcolp1 = &out_jac[firstRow + 2];
  for (std::size_t i = 0; i < nSpec - 1; ++i) {
    Tcolp1[i] = 0.;
  }

  // Yk column
  const double cvn = cvi[nSpec - 1];
  for (std::size_t k = 0; k < nSpec - 1; ++k) {
    const std::size_t firstRowSpec = (2 + k) * (nSpec + 1);
    out_jac[firstRowSpec] = 0.;
    out_jac[firstRowSpec + 1] = -invCv * out_rhs[1] * (cvi[k] - cvn);
    for (std::size_t i = 0; i < nSpec - 1; ++i) {
      out_jac[firstRowSpec + 2 + i] = 0.;
    }
    out_jac[firstRowSpec + 2 + k] = -invTau * inflowRho / rho;
  }
}

void CombustionKernels::heat_jacexactdense_isochoric(const double &temperature, const double &rho, const double &cv, const double &cvsensT,
    const double *cvi, const double &convectionTemperature, const double &radiationTemperature, const double &convectionCoefficient,
    const double &radiativeEmissivity, const double &surfaceAreaOverVolume, double *out_heatTransferRate,
    double *out_heatTransferRateJac) const {
  const int nSpec = mechanismData.phaseData.nSpecies;
  heat_rhs_isochoric(temperature, rho, cv, convectionTemperature, radiationTemperature, convectionCoefficient, radiativeEmissivity,
      surfaceAreaOverVolume, out_heatTransferRate);

  const double invRhoCv = 1. / (rho * cv);
  const double invCv = 1. / cv;

  // rho column
  out_heatTransferRateJac[0] = -*out_heatTransferRate / rho;

  // T column
  out_heatTransferRateJac[1] = -invCv * cvsensT * *out_heatTransferRate
      - surfaceAreaOverVolume * invRhoCv
          * (convectionCoefficient + 4. * radiativeEmissivity * 5.67e-8 * temperature * temperature * temperature);

  // Yk column
  double *Yksens = &out_heatTransferRateJac[2];
  const double cvn = cvi[nSpec - 1];
  for (std::size_t k = 0; k < nSpec - 1; ++k) {
    Yksens[k] = invCv * *out_heatTransferRate * (cvn - cvi[k]);
  }

}

void CombustionKernels::reactor_rhs_isochoric(const double *state, const double &inflowDensity, const double &inflowTemperature,
    const double *inflowY, const double &tau, const double &fluidTemperature, const double &surfTemperature, const double &hConv,
    const double &epsRad, const double &surfaceAreaOverVolume, const int heatTransferOption, const bool open, double *out_rhs) const {
  const int nSpec = mechanismData.phaseData.nSpecies;
  double p;
  double cv;
  double cvi[nSpec];
  double energies[nSpec];
  double w[nSpec];
  double out_heatTransferRate;
  double massRhs[nSpec];
  double inflowP;
  double inflowEnergies[nSpec];
  const double inflowMmw = mixture_molecular_weight(inflowY);

  const double density = state[0];
  const double temperature = state[1];
  double y[nSpec];
  griffon::blas::copy_vector(nSpec - 1, y, &state[2]);
  y[nSpec - 1] = 1. - std::accumulate(y, y + nSpec - 1, 0.);

  const double mmw = mixture_molecular_weight(y);

  ideal_gas_pressure(density, temperature, mmw, &p);
  cv_mix_and_species(temperature, y, mmw, &cv, cvi);
  species_energies(temperature, energies);
  production_rates(temperature, density, mmw, y, w);

  chem_rhs_isochoric(density, cv, energies, w, out_rhs);

  if (open) {
    species_energies(inflowTemperature, inflowEnergies);
    ideal_gas_pressure(inflowDensity, inflowTemperature, inflowMmw, &inflowP);

    mass_rhs_isochoric(y, energies, inflowEnergies, density, inflowDensity, p, inflowP, cv, inflowY, tau, massRhs);
    griffon::blas::vector_plus_scaled_vector(nSpec + 1, out_rhs, 1., massRhs);
  }

  switch (heatTransferOption) {
  case 0:  // adiabatic
    break;
  case 1:  // isothermal
    out_rhs[1] = 0.;
    break;
  case 2:  // diathermal
    heat_rhs_isochoric(temperature, density, cv, fluidTemperature, surfTemperature, hConv, epsRad, surfaceAreaOverVolume,
        &out_heatTransferRate);
    out_rhs[1] += out_heatTransferRate;
    break;
  }
}

void CombustionKernels::reactor_jac_isochoric(const double *state, const double &inflowDensity, const double &inflowTemperature,
    const double *inflowY, const double &tau, const double &fluidTemperature, const double &surfTemperature, const double &hConv,
    const double &epsRad, const double &surfaceAreaOverVolume, const int heatTransferOption, const bool open,
    const int rates_sensitivity_option, double *out_rhs, double *out_jac) const {
  const int nSpec = mechanismData.phaseData.nSpecies;
  double p;
  double cv;
  double cvsensT;
  double cvi[nSpec];
  double cvisensT[nSpec];
  double energies[nSpec];
  double w[nSpec];
  double wsens[(nSpec + 1) * (nSpec + 1)];
  double heatTransferRate;
  double heatTransferRateJac[nSpec];

  double inflowP;
  double inflowEnergies[nSpec];
  double massRhs[nSpec];
  double massJac[(nSpec + 1) * (nSpec + 1)];
  const double inflowMmw = mixture_molecular_weight(inflowY);

  const double density = state[0];
  const double temperature = state[1];
  double y[nSpec];
  griffon::blas::copy_vector(nSpec - 1, y, &state[2]);
  y[nSpec - 1] = 1. - std::accumulate(y, y + nSpec - 1, 0.);

  const double mmw = mixture_molecular_weight(y);
  ideal_gas_pressure(density, temperature, mmw, &p);
  cv_mix_and_species(temperature, y, mmw, &cv, cvi);
  cv_sens_T(temperature, y, &cvsensT, cvisensT);
  species_energies(temperature, energies);
  prod_rates_sens_exact(temperature, density, mmw, y, w, wsens);

  chem_jacexactdense_isochoric(temperature, y, density, cv, cvi, cvsensT, energies, w, wsens, out_rhs, out_jac);

  if (open) {
    ideal_gas_pressure(inflowDensity, inflowTemperature, inflowMmw, &inflowP);
    species_energies(inflowTemperature, inflowEnergies);
    mass_jacexactdense_isochoric(p, inflowP, temperature, y, density, inflowDensity, cv, cvsensT, cvi, energies, inflowEnergies,
        inflowTemperature, inflowY, tau, massRhs, massJac);
    griffon::blas::vector_plus_scaled_vector(nSpec + 1, out_rhs, 1., massRhs);
    griffon::blas::vector_plus_scaled_vector((nSpec + 1) * (nSpec + 1), out_jac, 1., massJac);
  }

  switch (heatTransferOption) {
  case 0:  // adiabatic
    break;
  case 1:  // isothermal
    for (std::size_t k = 0; k < nSpec + 1; ++k) {
      out_jac[k * (nSpec + 1) + 1] = 0.;
    }
    out_rhs[1] = 0.;
    break;
  case 2:  // diathermal
    heat_jacexactdense_isochoric(temperature, density, cv, cvsensT, cvi, fluidTemperature, surfTemperature, hConv, epsRad,
        surfaceAreaOverVolume, &heatTransferRate, heatTransferRateJac);
    out_rhs[1] += heatTransferRate;
    for (std::size_t k = 0; k < nSpec + 1; ++k) {
      out_jac[k * (nSpec + 1) + 1] += heatTransferRateJac[k];
    }
    break;
  }
}

}
