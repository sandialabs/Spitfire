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
#include <numeric>

namespace griffon {

void CombustionKernels::chem_rhs_isobaric(const double &rho, const double &cp, const double *h, const double *w, double *out_rhs) const {
  const int nSpec = mechanismData.phaseData.nSpecies;
  out_rhs[0] = -griffon::blas::inner_product(nSpec, w, h) / (rho * cp);
  griffon::blas::copy_vector(nSpec - 1, &out_rhs[1], w);
  griffon::blas::scale_vector(nSpec - 1, &out_rhs[1], 1. / rho);
}

void CombustionKernels::heat_rhs_isobaric(const double &temperature, const double &rho, const double &cp, const double &fluidTemperature,
    const double &surfTemperature, const double &hConv, const double &epsRad, const double &surfaceAreaOverVolume,
    double *out_heatTransferRate) const {
  *out_heatTransferRate = surfaceAreaOverVolume / (rho * cp)
      * (hConv * (fluidTemperature - temperature)
          + epsRad * 5.67e-8
              * (surfTemperature * surfTemperature * surfTemperature * surfTemperature
                  - temperature * temperature * temperature * temperature));
}

void CombustionKernels::mass_rhs_isobaric(const double *y, const double *enthalpies, const double *inflowEnthalpies, const double &rho,
    const double &cp, const double *inflowY, const double &tau, double *out_rhs) const {
  const int nSpec = mechanismData.phaseData.nSpecies;
  double hmhin[nSpec];
  griffon::blas::copy_vector(nSpec, hmhin, inflowEnthalpies);
  griffon::blas::vector_plus_scaled_vector(nSpec, hmhin, -1., enthalpies);
  out_rhs[0] = griffon::blas::inner_product(nSpec, hmhin, inflowY) / cp;

  griffon::blas::copy_vector(nSpec - 1, &out_rhs[1], inflowY);
  griffon::blas::vector_plus_scaled_vector(nSpec - 1, &out_rhs[1], -1., y);

  griffon::blas::scale_vector(nSpec, out_rhs, 1. / tau);
}

void CombustionKernels::chem_jacexactdense_isobaric(const double &pressure, const double &temperature, const double *y, const double &mmw,
    const double &rho, const double &cp, const double *cpi, const double &cpsensT, const double *h, const double *w, const double *wsens,
    double *out_rhs, double *out_primJac) const {
  const int nSpec = mechanismData.phaseData.nSpecies;
  // out_primJac is an n x (n+1) column-major array
  // row i col j is i + j * n

  chem_rhs_isobaric(rho, cp, h, w, out_rhs);

  const double invRhoCp = 1. / (rho * cp);
  const double invRho = 1. / rho;
  const double invCp = 1. / cp;

  // rho column
  out_primJac[0] = -invRhoCp * griffon::blas::inner_product(nSpec, wsens, h) - invRho * out_rhs[0];
  griffon::blas::copy_vector(nSpec - 1, &out_primJac[1], wsens);
  griffon::blas::vector_plus_scaled_vector(nSpec - 1, &out_primJac[1], -invRho, w);
  griffon::blas::scale_vector(nSpec - 1, &out_primJac[1], invRho);

  // T column
  out_primJac[nSpec] = -invRhoCp * (griffon::blas::inner_product(nSpec, &wsens[nSpec + 1], h) + griffon::blas::inner_product(nSpec, w, cpi))
      - out_rhs[0] * cpsensT * invCp;
  griffon::blas::copy_vector(nSpec - 1, &out_primJac[nSpec + 1], &wsens[nSpec + 1]);
  griffon::blas::scale_vector(nSpec - 1, &out_primJac[nSpec + 1], invRho);

  // Yk column
  const double cpn = cpi[nSpec - 1];
  for (std::size_t k = 0; k < nSpec - 1; ++k) {
    const std::size_t firstRow = (2 + k) * nSpec;
    out_primJac[firstRow] = -invRhoCp * griffon::blas::inner_product(nSpec, &wsens[(2 + k) * (nSpec + 1)], h)
        - out_rhs[0] * (cpi[k] - cpn) * invCp;
    griffon::blas::copy_vector(nSpec - 1, &out_primJac[firstRow + 1], &wsens[(2 + k) * (nSpec + 1)]);
    griffon::blas::scale_vector(nSpec - 1, &out_primJac[firstRow + 1], invRho);
  }

}

void CombustionKernels::mass_jacexactdense_isobaric(const double &pressure, const double &temperature, const double *y, const double &rho,
    const double &cp, const double &cpsensT, const double *cpi, const double *enthalpies, const double *inflowEnthalpies,
    const double &inflowTemperature, const double *inflowY, const double &tau, double *out_rhs, double *out_primJac) const {
  const int nSpec = mechanismData.phaseData.nSpecies;
  // out_primJac is an n x (n+1) column-major array
  // row i col j is i + j * n

  mass_rhs_isobaric(y, enthalpies, inflowEnthalpies, rho, cp, inflowY, tau, out_rhs);

  const double invCp = 1. / cp;
  const double invTau = 1. / tau;

  // rho column
  for (std::size_t i = 0; i < nSpec; ++i) {
    out_primJac[i] = 0.;
  }

  // T column
  out_primJac[nSpec] = -invCp * (cpsensT * out_rhs[0] + invTau * griffon::blas::inner_product(nSpec, inflowY, cpi));
  double *Tcolp1 = &out_primJac[nSpec + 1];
  for (std::size_t i = 0; i < nSpec - 1; ++i) {
    Tcolp1[i] = 0.;
  }

  // Yk column
  for (std::size_t k = 0; k < nSpec - 1; ++k) {
    out_primJac[(2 + k) * nSpec] = -out_rhs[0] * (cpi[k] - cpi[nSpec - 1]) * invCp;
    double *Ykcolp1 = &out_primJac[(2 + k) * nSpec + 1];
    for (std::size_t i = 0; i < nSpec - 1; ++i) {
      Ykcolp1[i] = 0.;
    }
    out_primJac[(2 + k) * nSpec + 1 + k] = -invTau;
  }
}

void CombustionKernels::heat_jacexactdense_isobaric(const double &temperature, const double &rho, const double &cp, const double &cpsensT,
    const double *cpi, const double &convectionTemperature, const double &radiationTemperature, const double &convectionCoefficient,
    const double &radiativeEmissivity, const double &surfaceAreaOverVolume, double *out_heatTransferRate,
    double *out_heatTransferRatePrimJac) const {
  const int nSpec = mechanismData.phaseData.nSpecies;
  heat_rhs_isobaric(temperature, rho, cp, convectionTemperature, radiationTemperature, convectionCoefficient, radiativeEmissivity,
      surfaceAreaOverVolume, out_heatTransferRate);

  const double invRhoCp = 1. / (rho * cp);
  const double invCp = 1. / cp;

  // rho column
  out_heatTransferRatePrimJac[0] = -*out_heatTransferRate / rho;

  // T column
  out_heatTransferRatePrimJac[1] = -invCp * cpsensT * *out_heatTransferRate
      - surfaceAreaOverVolume * invRhoCp
          * (convectionCoefficient + 4. * radiativeEmissivity * 5.67e-8 * temperature * temperature * temperature);

  // Yk column
  double *Yksens = &out_heatTransferRatePrimJac[2];
  const double cpn = cpi[nSpec - 1];
  for (std::size_t k = 0; k < nSpec - 1; ++k) {
    Yksens[k] = invCp * *out_heatTransferRate * (cpn - cpi[k]);
  }
}

void CombustionKernels::reactor_rhs_isobaric(const double *state, const double &pressure, const double &inflowTemperature,
    const double *inflowY, const double &tau, const double &fluidTemperature, const double &surfTemperature, const double &hConv,
    const double &epsRad, const double &surfaceAreaOverVolume, const int heatTransferOption, const bool open, double *out_rhs) const {
  const int nSpec = mechanismData.phaseData.nSpecies;
  double rho;
  double heatTransferRate;
  double enthalpies[nSpec];
  double w[nSpec];

  const double temperature = state[0];
  double y[nSpec];
  griffon::blas::copy_vector(nSpec - 1, y, &state[1]);
  y[nSpec - 1] = 1. - std::accumulate(y, y + nSpec - 1, 0.);

  const double mmw = mixture_molecular_weight(y);
  ideal_gas_density(pressure, temperature, mmw, &rho);
  const double cp = cp_mix(temperature, y);
  species_enthalpies(temperature, enthalpies);
  production_rates(temperature, rho, mmw, y, w);

  chem_rhs_isobaric(rho, cp, enthalpies, w, out_rhs);

  if (open) {
    double massRhs[nSpec];
    double inflowEnthalpies[nSpec];
    species_enthalpies(inflowTemperature, inflowEnthalpies);
    mass_rhs_isobaric(y, enthalpies, inflowEnthalpies, rho, cp, inflowY, tau, massRhs);
    griffon::blas::vector_plus_scaled_vector(nSpec, out_rhs, 1., massRhs);
  }

  switch (heatTransferOption) {
  case 0:  // adiabatic
    break;
  case 1:  // isothermal
    out_rhs[0] = 0.;
    break;
  case 2:  // diathermal
    heat_rhs_isobaric(temperature, rho, cp, fluidTemperature, surfTemperature, hConv, epsRad, surfaceAreaOverVolume, &heatTransferRate);
    out_rhs[0] += heatTransferRate;
    break;
  }
}

void CombustionKernels::reactor_jac_isobaric(const double *state, const double &pressure, const double &inflowTemperature,
    const double *inflowY, const double &tau, const double &fluidTemperature, const double &surfTemperature, const double &hConv,
    const double &epsRad, const double &surfaceAreaOverVolume, const int heatTransferOption, const bool open,
    const int rates_sensitivity_option, const int sensitivity_transform_option, double *out_rhs, double *out_jac) const {
  const int nSpec = mechanismData.phaseData.nSpecies;
  // out_jac is an n x n column-major array
  // row i col j is i + j * n

  double rho;
  double cp;
  double cpsensT;
  double cpi[nSpec];
  double cpisensT[nSpec];
  double enthalpies[nSpec];
  double w[nSpec];
  double wsens[(nSpec + 1) * (nSpec + 1)];

  double primJac[nSpec * (nSpec + 1)];
  double heatTransferRate;
  double heatTrasferRatePrimJac[nSpec + 1];

  const double temperature = state[0];
  double y[nSpec];
  griffon::blas::copy_vector(nSpec - 1, y, &state[1]);
  y[nSpec - 1] = 1. - std::accumulate(y, y + nSpec - 1, 0.);

  const double mmw = mixture_molecular_weight(y);
  ideal_gas_density(pressure, temperature, mmw, &rho);
  cp_mix_and_species(temperature, y, &cp, cpi);
  species_enthalpies(temperature, enthalpies);
  cp_sens_T(temperature, y, &cpsensT, cpisensT);

  switch (rates_sensitivity_option) {
  case 1:
    prod_rates_sens_no_tbaf(temperature, rho, mmw, y, w, wsens);
    break;
  case 0:
    prod_rates_sens_dense(temperature, rho, mmw, y, w, wsens);
    break;
  }

  chem_jacexactdense_isobaric(pressure, temperature, y, mmw, rho, cp, cpi, cpsensT, enthalpies, w, wsens, out_rhs, primJac);

  if (open) {
    double massRhs[nSpec];
    double massPrimJac[nSpec * (nSpec + 1)];
    double inflowEnthalpies[nSpec];
    species_enthalpies(inflowTemperature, inflowEnthalpies);
    mass_jacexactdense_isobaric(pressure, temperature, y, rho, cp, cpsensT, cpi, enthalpies, inflowEnthalpies, inflowTemperature, inflowY,
        tau, massRhs, massPrimJac);
    griffon::blas::vector_plus_scaled_vector(nSpec * (nSpec + 1), primJac, 1., massPrimJac);
    griffon::blas::vector_plus_scaled_vector(nSpec, out_rhs, 1., massRhs);
  }

  switch (heatTransferOption) {
  case 0:  // adiabatic
    break;
  case 1:  // isothermal
    for (std::size_t k = 0; k < nSpec + 1; ++k) {
      primJac[k * nSpec] = 0.;
    }
    out_rhs[0] = 0.;
    break;
  case 2:  // diathermal
    heat_jacexactdense_isobaric(temperature, rho, cp, cpsensT, cpi, fluidTemperature, surfTemperature, hConv, epsRad, surfaceAreaOverVolume,
        &heatTransferRate, heatTrasferRatePrimJac);
    for (std::size_t k = 0; k < nSpec + 1; ++k) {
      primJac[k * nSpec] += heatTrasferRatePrimJac[k];
    }
    out_rhs[0] += heatTransferRate;
    break;
  }

  switch (sensitivity_transform_option) {
  case 0:
    transform_isobaric_primitive_jacobian(rho, pressure, temperature, mmw, primJac, out_jac);
    break;
  }
}

void CombustionKernels::transform_isobaric_primitive_jacobian(const double &rho, const double &pressure, const double &temperature,
    const double &mmw, const double *primJac, double *out_jac) const {
  const int nSpec = mechanismData.phaseData.nSpecies;
  const auto &invMolecularWeights = mechanismData.phaseData.inverseMolecularWeights;

  for (std::size_t i = 0; i < nSpec * nSpec; ++i)
    out_jac[i] = 0.;

  griffon::blas::copy_vector(nSpec, out_jac, &primJac[nSpec]);
  griffon::blas::vector_plus_scaled_vector(nSpec, out_jac, -rho / temperature, primJac);

  const double negRhoMmw = -rho * mmw;
  for (std::size_t k = 0; k < nSpec - 1; ++k) {
    griffon::blas::copy_vector(nSpec, &out_jac[(1 + k) * nSpec], &primJac[(2 + k) * nSpec]);
    griffon::blas::vector_plus_scaled_vector(nSpec, &out_jac[(1 + k) * nSpec],
        negRhoMmw * (invMolecularWeights[k] - invMolecularWeights[nSpec - 1]), primJac);
  }
}
}

