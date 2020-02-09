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
#include "btddod_matrix_kernels.h"
#include <cmath>
#include <numeric>

namespace griffon {

void CombustionKernels::flamelet_stencils(const double *dz, const int &nzi, const double *dissipationRate, const double *invLewisNumbers,
    double *out_cmajor, double *out_csub, double *out_csup, double *out_mcoeff, double *out_ncoeff) const {
  const int nSpec = mechanismData.phaseData.nSpecies;
  for (int i = 0; i < nzi; ++i) {
    const double dzt = dz[i] + dz[i + 1];
    griffon::blas::copy_vector(nSpec, &out_cmajor[i * nSpec], invLewisNumbers);
    griffon::blas::copy_vector(nSpec, &out_csub[i * nSpec], invLewisNumbers);
    griffon::blas::copy_vector(nSpec, &out_csup[i * nSpec], invLewisNumbers);
    griffon::blas::scale_vector(nSpec, &out_cmajor[i * nSpec], -dissipationRate[1 + i] / (dz[i] * dz[i + 1]));
    griffon::blas::scale_vector(nSpec, &out_csub[i * nSpec], dissipationRate[1 + i] / (dzt * dz[i]));
    griffon::blas::scale_vector(nSpec, &out_csup[i * nSpec], dissipationRate[1 + i] / (dzt * dz[i + 1]));
    out_ncoeff[i] = 1 / (dz[i] + dz[i + 1]);
    out_mcoeff[i] = -out_ncoeff[i];
  }
}

void CombustionKernels::flamelet_jac_indices(const int &nzi, int *out_row_indices, int *out_col_indices) const {
  const int nSpec = mechanismData.phaseData.nSpecies;
  int idx = 0;
  // diagonal blocks
  for (int iz = 0; iz < nzi; ++iz) {
    const int blockBase = iz * nSpec;
    for (int iq = 0; iq < nSpec; ++iq) {
      for (int jq = 0; jq < nSpec; ++jq) {
        out_row_indices[idx] = blockBase + jq;
        out_col_indices[idx] = blockBase + iq;
        ++idx;
      }
    }
  }
  // subdiagonal
  for (int iz = 1; iz < nzi; ++iz) {
    const int blockBase = iz * nSpec;
    for (int iq = 0; iq < nSpec; ++iq) {
      out_row_indices[idx] = blockBase + iq;
      out_col_indices[idx] = blockBase + iq - nSpec;
      ++idx;
    }
  }
  // superdiagonal
  for (int iz = 0; iz < nzi - 1; ++iz) {
    const int blockBase = iz * nSpec;
    for (int iq = 0; iq < nSpec; ++iq) {
      out_row_indices[idx] = blockBase + iq;
      out_col_indices[idx] = blockBase + iq + nSpec;
      ++idx;
    }
  }
}

void CombustionKernels::flamelet_process_density(const double *state, const double pressure, const int nzi, double *out_density) const {
  const int nSpec = mechanismData.phaseData.nSpecies;
  for (int i = 0; i < nzi; ++i) {
    double y[nSpec];
    griffon::blas::copy_vector(nSpec - 1, y, &state[i * nSpec + 1]);
    y[nSpec - 1] = 1.;
    for (int j = 0; j < nSpec - 1; ++j) {
      y[nSpec - 1] -= y[j];
    }
    out_density[i] = ideal_gas_density(pressure, state[i * nSpec], y);
  }
}

void CombustionKernels::flamelet_process_enthalpy(const double *state, const int nzi, double *out_enthalpy) const {
  const int nSpec = mechanismData.phaseData.nSpecies;
  for (int i = 0; i < nzi; ++i) {
    double y[nSpec];
    griffon::blas::copy_vector(nSpec - 1, y, &state[i * nSpec + 1]);
    y[nSpec - 1] = 1.;
    for (int j = 0; j < nSpec - 1; ++j) {
      y[nSpec - 1] -= y[j];
    }
    out_enthalpy[i] = enthalpy_mix(state[i * nSpec], y);
  }
}

void CombustionKernels::flamelet_process_energy(const double *state, const int nzi, double *out_energy) const {
  const int nSpec = mechanismData.phaseData.nSpecies;
  for (int i = 0; i < nzi; ++i) {
    double y[nSpec];
    griffon::blas::copy_vector(nSpec - 1, y, &state[i * nSpec + 1]);
    y[nSpec - 1] = 1.;
    for (int j = 0; j < nSpec - 1; ++j) {
      y[nSpec - 1] -= y[j];
    }
    out_energy[i] = enthalpy_mix(state[i * nSpec], y);
  }
}

void CombustionKernels::flamelet_process_cp(const double *state, const int nzi, double *out_cp) const {
  const int nSpec = mechanismData.phaseData.nSpecies;
  for (int i = 0; i < nzi; ++i) {
    double y[nSpec];
    griffon::blas::copy_vector(nSpec - 1, y, &state[i * nSpec + 1]);
    y[nSpec - 1] = 1.;
    for (int j = 0; j < nSpec - 1; ++j) {
      y[nSpec - 1] -= y[j];
    }
    out_cp[i] = cp_mix(state[i * nSpec], y);
  }
}

void CombustionKernels::flamelet_process_cv(const double *state, const int nzi, double *out_cv) const {
  const int nSpec = mechanismData.phaseData.nSpecies;
  for (int i = 0; i < nzi; ++i) {
    double y[nSpec];
    griffon::blas::copy_vector(nSpec - 1, y, &state[i * nSpec + 1]);
    y[nSpec - 1] = 1.;
    for (int j = 0; j < nSpec - 1; ++j) {
      y[nSpec - 1] -= y[j];
    }
    out_cv[i] = cv_mix(state[i * nSpec], y);
  }
}

void CombustionKernels::flamelet_process_mole_fractions(const double *state, const int nzi, double *out_molefracs) const {
  const int nSpec = mechanismData.phaseData.nSpecies;
  for (int i = 0; i < nzi; ++i) {
    double y[nSpec];
    griffon::blas::copy_vector(nSpec - 1, y, &state[i * nSpec + 1]);
    y[nSpec - 1] = 1.;
    for (int j = 0; j < nSpec - 1; ++j) {
      y[nSpec - 1] -= y[j];
    }
    mole_fractions(y, &out_molefracs[i * nSpec]);
  }
}

void CombustionKernels::flamelet_process_isobaric_reactor_rhs(const double *state, const double pressure, const int nzi,
    double *out_rates) const {
  const int nSpec = mechanismData.phaseData.nSpecies;
  const double z = 0.;
  for (int i = 0; i < nzi; ++i) {
    reactor_rhs_isobaric(&state[i * nSpec], pressure, z, &z, z, z, z, z, z, z, 0, false, &out_rates[i * nSpec]);
  }
}

// ------------------------------------------------------------------------------------------------

void CombustionKernels::flamelet_rhs(const double *state, const double &pressure, const double *oxyState, const double *fuelState,
    const bool adiabatic, const double *T_convection, const double *h_convection, const double *T_radiation, const double *h_radiation,
    const int &nzi, const double *cmajor, const double *csub, const double *csup, const double *mcoeff, const double *ncoeff,
    const double *chi, const bool include_enthalpy_flux, const bool include_variable_cp, const bool use_scaled_heat_loss,
    double *out_rhs) const {
  const int nSpec = mechanismData.phaseData.nSpecies;

  double maxT = -1;
  double maxT4 = -1;
  if (use_scaled_heat_loss) {
    for (int i = 0; i < nzi; ++i) {
      maxT = std::max(maxT, state[i * nSpec]);
    }
    maxT4 = maxT * maxT * maxT * maxT;
  }

  double cpz_grid[nzi];

  if (include_variable_cp) {
    double cp_grid[nzi];
    for (int i = 0; i < nzi; ++i) {
      double y[nSpec];
      griffon::blas::copy_vector(nSpec - 1, y, &state[i * nSpec + 1]);
      y[nSpec - 1] = 1. - std::accumulate(y, y + nSpec - 1, 0.);
      cp_grid[i] = cp_mix(state[i * nSpec], y);
    }
    for (int i = 0; i < nzi; ++i) {
      if (i == 0) {
        double y[nSpec];
        const double* b_state = oxyState;
        griffon::blas::copy_vector(nSpec - 1, y, &b_state[1]);
        y[nSpec - 1] = 1. - std::accumulate(y, y + nSpec - 1, 0.);
        cpz_grid[i] = mcoeff[i] * cp_mix(b_state[0], y) + ncoeff[i] * cp_grid[1];
      } else if (i == nzi - 1) {
        double y[nSpec];
        const double* b_state = fuelState;
        griffon::blas::copy_vector(nSpec - 1, y, &b_state[1]);
        y[nSpec - 1] = 1. - std::accumulate(y, y + nSpec - 1, 0.);
        cpz_grid[i] = mcoeff[i] * cp_grid[nzi - 2] + ncoeff[i] * cp_mix(b_state[0], y);
      } else {
        const double* state_nm1 = &state[(i - 1) * nSpec];
        const double* state_np1 = &state[(i + 1) * nSpec];
        cpz_grid[i] = mcoeff[i] * cp_grid[i - 1] + ncoeff[i] * cp_grid[i + 1];
      }
    }
  }

  for (int i = 0; i < nzi; ++i) {
    double rho;
    double enthalpies[nSpec];
    double w[nSpec];
    double cp;
    double cpi[nSpec];

    const double T = state[i * nSpec];
    double y[nSpec];
    griffon::blas::copy_vector(nSpec - 1, y, &state[i * nSpec + 1]);
    y[nSpec - 1] = 1. - std::accumulate(y, y + nSpec - 1, 0.);

    const double mmw = mixture_molecular_weight(y);
    ideal_gas_density(pressure, T, mmw, &rho);
    if (include_enthalpy_flux) {
      cp_mix_and_species(T, y, &cp, cpi);
    } else {
      cp = cp_mix(T, y);
    }

    species_enthalpies(T, enthalpies);
    production_rates(T, rho, mmw, y, w);
    chem_rhs_isobaric(rho, cp, enthalpies, w, &out_rhs[i * nSpec]);

    if (!adiabatic) {
      const double hc = h_convection[i];
      const double hr = h_radiation[i];
      const double Tc = T_convection[i];
      const double Tr = T_radiation[i];
      if (use_scaled_heat_loss) {
        const double Tr4 = Tr * Tr * Tr * Tr;
        const double q = hc * (Tc - T) / (maxT - Tc) + hr * 5.67e-8 * (Tr4 - T * T * T * T) / (maxT4 - Tr4);
        out_rhs[i * nSpec] += q / (rho * cp);
      } else {
        const double q = hc * (Tc - T) + hr * 5.67e-8 * (Tr * Tr * Tr * Tr - T * T * T * T);
        out_rhs[i * nSpec] += q / (rho * cp);
      }
    }

    if (include_enthalpy_flux) {
      const double cpn = cpi[nSpec - 1];
      double dTdZ;
      double dYdZ_cpi = 0.;
      if (i == 0) {
        const double* state_nm1 = oxyState;
        const double* state_np1 = &state[nSpec];
        dTdZ = mcoeff[i] * state_nm1[0] + ncoeff[i] * state_np1[0];
        for (int j = 0; j < nSpec - 1; ++j) {
          dYdZ_cpi += (cpi[j] - cpn) * (mcoeff[i] * state_nm1[1 + j] + ncoeff[i] * state_np1[1 + j]);
        }
      } else if (i == nzi - 1) {
        const double* state_nm1 = &state[(nzi - 2) * nSpec];
        const double* state_np1 = fuelState;
        dTdZ = mcoeff[i] * state_nm1[0] + ncoeff[i] * state_np1[0];
        for (int j = 0; j < nSpec - 1; ++j) {
          dYdZ_cpi += (cpi[j] - cpn) * (mcoeff[i] * state_nm1[1 + j] + ncoeff[i] * state_np1[1 + j]);
        }
      } else {
        const double* state_nm1 = &state[(i - 1) * nSpec];
        const double* state_np1 = &state[(i + 1) * nSpec];
        dTdZ = mcoeff[i] * state_nm1[0] + ncoeff[i] * state_np1[0];
        for (int j = 0; j < nSpec - 1; ++j) {
          dYdZ_cpi += (cpi[j] - cpn) * (mcoeff[i] * state_nm1[1 + j] + ncoeff[i] * state_np1[1 + j]);
        }
      }
      out_rhs[i * nSpec] += 0.5 * chi[i] / cp * dTdZ * dYdZ_cpi;

      if (include_variable_cp) {
        out_rhs[i * nSpec] += 0.5 * chi[i] * cpz_grid[i] / cp * dTdZ;
      }
    }

    if (include_variable_cp and not include_enthalpy_flux) {
      double dTdZ;
      if (i == 0) {
        const double* state_nm1 = oxyState;
        const double* state_np1 = &state[nSpec];
        dTdZ = mcoeff[i] * state_nm1[0] + ncoeff[i] * state_np1[0];
      } else if (i == nzi - 1) {
        const double* state_nm1 = &state[(nzi - 2) * nSpec];
        const double* state_np1 = fuelState;
        dTdZ = mcoeff[i] * state_nm1[0] + ncoeff[i] * state_np1[0];
      } else {
        const double* state_nm1 = &state[(i - 1) * nSpec];
        const double* state_np1 = &state[(i + 1) * nSpec];
        dTdZ = mcoeff[i] * state_nm1[0] + ncoeff[i] * state_np1[0];
      }
      out_rhs[i * nSpec] += 0.5 * chi[i] * cpz_grid[i] / cp * dTdZ;
    }
  }

  const int endIdx = (nzi - 1) * nSpec;
  for (int i = nSpec; i < endIdx; ++i) {
    out_rhs[i] += cmajor[i] * state[i] + csub[i] * state[i - nSpec] + csup[i] * state[i + nSpec];
  }
  for (int j = 0; j < nSpec; ++j) {
    out_rhs[j] += cmajor[j] * state[j] + csub[j] * oxyState[j] + csup[j] * state[j + nSpec];
    out_rhs[endIdx + j] += cmajor[endIdx + j] * state[endIdx + j] + csub[endIdx + j] * state[endIdx + j - nSpec]
        + csup[endIdx + j] * fuelState[j];
  }
}

void CombustionKernels::flamelet_jacobian(const double *state, const double &pressure, const double *oxyState, const double *fuelState,
    const bool adiabatic, const double *T_convection, const double *h_convection, const double *T_radiation, const double *h_radiation,
    const int &nzi, const double *cmajor, const double *csub, const double *csup, const double *mcoeff, const double *ncoeff,
    const double *chi, const bool compute_eigenvalues, const double diffterm, const bool scale_and_offset, const double prefactor,
    const int &rates_sensitivity_option, const int &sensitivity_transform_option, const bool include_enthalpy_flux,
    const bool include_variable_cp, const bool use_scaled_heat_loss, double *out_expeig, double *out_jac) const {
  const int nSpec = mechanismData.phaseData.nSpecies;
  const int nelements = nSpec * (nzi * nSpec + 2 * (nzi - 1));
  const int blocksize = nSpec * nSpec;

  double rhsTemp[nSpec];
  double realParts[nSpec];
  double imagParts[nSpec];

  double cp[nzi];
  double cpsensT[nzi];

  double maxT = -1;
  double maxT4 = -1;
  if (use_scaled_heat_loss) {
    for (int i = 0; i < nzi; ++i) {
      maxT = std::max(maxT, state[i * nSpec]);
    }
    maxT4 = maxT * maxT * maxT * maxT;
  }

  int idx = 0;
  for (int iz = 0; iz < nzi; ++iz) {
    double rho;
    double cpi[nSpec];
    double cpisensT[nSpec];
    double enthalpies[nSpec];
    double w[nSpec];
    double wsens[(nSpec + 1) * (nSpec + 1)];

    double primJac[nSpec * (nSpec + 1)];

    const double T = state[iz * nSpec];
    double y[nSpec];
    griffon::blas::copy_vector(nSpec - 1, y, &state[iz * nSpec + 1]);
    y[nSpec - 1] = 1. - std::accumulate(y, y + nSpec - 1, 0.);

    const double mmw = mixture_molecular_weight(y);
    ideal_gas_density(pressure, T, mmw, &rho);
    cp_mix_and_species(T, y, &cp[iz], cpi);
    species_enthalpies(T, enthalpies);
    cp_sens_T(T, y, &cpsensT[iz], cpisensT);

    switch (rates_sensitivity_option) {
    case 1:
      prod_rates_sens_no_tbaf(T, rho, mmw, y, w, wsens);
      break;
    case 0:
      prod_rates_sens_exact(T, rho, mmw, y, w, wsens);
      break;
    case 2:
      prod_rates_sens_sparse(T, rho, mmw, y, w, wsens);
      break;
    }

    chem_jacexactdense_isobaric(pressure, T, y, mmw, rho, cp[iz], cpi, cpsensT[iz], enthalpies, w, wsens, rhsTemp, primJac);

    if (!adiabatic) {
      const double Tc = T_convection[iz];
      const double Tr = T_radiation[iz];
      const double hc = h_convection[iz];
      const double hr = h_radiation[iz];

      const double invRhoCp = 1. / (rho * cp[iz]);
      const double invCp = 1. / cp[iz];

      double q;
      if (use_scaled_heat_loss) {
        const double Tr4 = Tr * Tr * Tr * Tr;
        q = (hc * (Tc - T) / (maxT - Tc) + hr * 5.67e-8 * (Tr4 - T * T * T * T) / (maxT4 - Tr4)) * invRhoCp;
        primJac[nSpec] -= invCp * cpsensT[iz] * q + invRhoCp * (hc / (maxT - Tc) + 4. * hr / (maxT4 - Tr4) * 5.67e-8 * T * T * T);
      } else {
        q = (hc * (Tc - T) + hr * 5.67e-8 * (Tr * Tr * Tr * Tr - T * T * T * T)) * invRhoCp;
        primJac[nSpec] -= invCp * cpsensT[iz] * q + invRhoCp * (hc + 4. * hr * 5.67e-8 * T * T * T);
      }

      primJac[0] -= q / rho;

      const double cpn = cpi[nSpec - 1];
      for (std::size_t k = 0; k < nSpec + 1; ++k) {
        primJac[(2 + k) * nSpec] += invCp * q * (cpn - cpi[k]);
      }
    }

    switch (sensitivity_transform_option) {
    case 0:
      transform_isobaric_primitive_jacobian(rho, pressure, T, mmw, primJac, &out_jac[idx]);
      break;
    }

    if (compute_eigenvalues) {
      griffon::lapack::eigenvalues(nSpec, &out_jac[idx], realParts, imagParts);
      double exp_eig = 0.;
      for (int iq = 0; iq < nSpec; ++iq) {
        exp_eig = std::max(exp_eig, std::max(realParts[iq] - diffterm, 0.));
      }
      for (int iq = 0; iq < nSpec; ++iq) {
        out_expeig[iz * nSpec + iq] = exp_eig;
      }
    }

    for (int iq = 0; iq < nSpec; ++iq) {
      out_jac[idx + iq * (nSpec + 1)] += cmajor[iz * nSpec + iq];
    }
    idx += blocksize;
  }

  if (include_enthalpy_flux) { // note that this is very inexact. this should be improved but convergence seems fine
    for (int i = 1; i < nzi - 1; ++i) {
      const double dTdZ = mcoeff[i] * state[(i - 1) * nSpec] + ncoeff[i] * state[(i + 1) * nSpec];
      const double dcpdZ = mcoeff[i] * cp[i - 1] + ncoeff[i] * cp[i + 1];
      const double f1 = 0.5 * chi[i] / cp[i] * dTdZ * dcpdZ;
      out_jac[i * blocksize] -= f1 / cp[i] * cpsensT[i];
    }
    {
      double y[nSpec];
      griffon::blas::copy_vector(nSpec - 1, y, &oxyState[1]);
      y[nSpec - 1] = 1. - std::accumulate(y, y + nSpec - 1, 0.);
      const double cp_oxy = cp_mix(oxyState[0], y);

      const int i = 0;
      const double dTdZ = mcoeff[i] * oxyState[0] + ncoeff[i] * state[(i + 1) * nSpec];
      const double dcpdZ = mcoeff[i] * cp_oxy + ncoeff[i] * cp[i + 1];
      const double f1 = 0.5 * chi[i] / cp[i] * dTdZ * dcpdZ;
      out_jac[i * blocksize] -= f1 / cp[i] * cpsensT[i];
    }
    {
      double y[nSpec];
      griffon::blas::copy_vector(nSpec - 1, y, &fuelState[1]);
      y[nSpec - 1] = 1. - std::accumulate(y, y + nSpec - 1, 0.);
      const double cp_fuel = cp_mix(fuelState[0], y);

      const int i = nzi - 1;
      const double dTdZ = mcoeff[i] * state[(i - 1) * nSpec] + ncoeff[i] * fuelState[0];
      const double dcpdZ = mcoeff[i] * cp[i - 1] + ncoeff[i] * cp_fuel;
      const double f1 = 0.5 * chi[i] / cp[i] * dTdZ * dcpdZ;
      out_jac[i * blocksize] -= f1 / cp[i] * cpsensT[i];
    }
  }

  // we should consider adding the variable cp sensitivities here, at least something close

  const int off_diag_offset = (nzi - 1) * nSpec;
  for (int iz = 1; iz < nzi; ++iz) {
    for (int iq = 0; iq < nSpec; ++iq) {
      out_jac[idx] += csub[iz * nSpec + iq];
      out_jac[off_diag_offset + idx] += csup[(iz - 1) * nSpec + iq];
      ++idx;
    }
  }
  if (scale_and_offset) {
    griffon::blas::scale_vector(nelements, out_jac, prefactor);
    for (int iz = 0; iz < nzi; ++iz) {
      for (int iq = 0; iq < nSpec; ++iq) {
        out_jac[iz * blocksize + iq * (nSpec + 1)] -= 1.;
      }
    }
  }
}
}
