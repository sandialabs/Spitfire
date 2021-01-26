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

namespace griffon
{

void CombustionKernels::flamelet2d_rhs(const double *state, const double &pressure, const int &nx, const int &ny,
                                       const double *xcp, const double *xcl, const double *xcr, const double *ycp,
                                       const double *ycb, const double *yct, double *out_rhs) const
{
  const int nq = mechanismData.phaseData.nSpecies;
  const int nyq = ny * nq;
  for (int ix = 1; ix < nx - 1; ++ix)
  {
    const int i = ix * nyq;
    const int isx = (ix - 1) * nq;
    for (int iy = 1; iy < ny - 1; ++iy)
    {
      const int ij = i + iy * nq;

      double rho;
      double enthalpies[nq];
      double w[nq];

      const double T = state[ij];
      double y[nq];
      extract_y(&state[ij + 1], nq, y);

      const double mmw = mixture_molecular_weight(y);
      ideal_gas_density(pressure, T, mmw, &rho);
      const double cp = cp_mix(T, y);

      species_enthalpies(T, enthalpies);
      production_rates(T, rho, mmw, y, w);
      chem_rhs_isobaric(rho, cp, enthalpies, w, &out_rhs[ij]);

      const int ijm1 = ij - nq;
      const int ijp1 = ij + nq;
      const int im1j = ij - nyq;
      const int ip1j = ij + nyq;
      const int isy = (iy - 1) * nq;
      for (int iq = 0; iq < nq; ++iq)
      {
        out_rhs[ij + iq] += xcr[isx + iq] * state[ip1j + iq] + xcl[isx + iq] * state[im1j + iq] + xcp[isx + iq] * state[ij + iq] + yct[isy + iq] * state[ijp1 + iq] + ycb[isy + iq] * state[ijm1 + iq] + ycp[isy + iq] * state[ij + iq];
      }
    }
    const int ij_1 = ix * nyq;
    const int im1j_1 = ij_1 - nyq;
    const int ip1j_1 = ij_1 + nyq;
    const int ij_2 = ix * nyq + (ny - 1) * nq;
    const int im1j_2 = ij_2 - nyq;
    const int ip1j_2 = ij_2 + nyq;

    double rho_1, rho_2;
    double enthalpies_1[nq], enthalpies_2[nq];
    double w_1[nq], w_2[nq];
    double rhs_1[nq], rhs_2[nq];

    const double T_1 = state[ij_1];
    const double T_2 = state[ij_2];
    double y_1[nq];
    double y_2[nq];
    extract_y(&state[ij_1 + 1], nq, y_1);
    extract_y(&state[ij_2 + 1], nq, y_2);

    const double mmw_1 = mixture_molecular_weight(y_1);
    const double mmw_2 = mixture_molecular_weight(y_2);
    ideal_gas_density(pressure, T_1, mmw_1, &rho_1);
    ideal_gas_density(pressure, T_2, mmw_2, &rho_2);
    const double cp_1 = cp_mix(T_1, y_1);
    const double cp_2 = cp_mix(T_2, y_2);

    species_enthalpies(T_1, enthalpies_1);
    production_rates(T_1, rho_1, mmw_1, y_1, w_1);
    chem_rhs_isobaric(rho_1, cp_1, enthalpies_1, w_1, rhs_1);
    species_enthalpies(T_2, enthalpies_2);
    production_rates(T_2, rho_2, mmw_2, y_2, w_2);
    chem_rhs_isobaric(rho_2, cp_2, enthalpies_2, w_2, rhs_2);

    for (int iq = 0; iq < nq; ++iq)
    {
      out_rhs[ij_1 + iq] += xcr[isx + iq] * state[ip1j_1 + iq] + xcl[isx + iq] * state[im1j_1 + iq] + xcp[isx + iq] * state[ij_1 + iq] + rhs_1[iq];
      out_rhs[ij_2 + iq] += xcr[isx + iq] * state[ip1j_2 + iq] + xcl[isx + iq] * state[im1j_2 + iq] + xcp[isx + iq] * state[ij_2 + iq] + rhs_2[iq];
    }
  }

  for (int iy = 1; iy < ny - 1; ++iy)
  {
    const int ij_1 = iy * nq;
    const int ijm1_1 = ij_1 - nq;
    const int ijp1_1 = ij_1 + nq;
    const int ij_2 = (nx - 1) * nyq + iy * nq;
    const int ijm1_2 = ij_2 - nq;
    const int ijp1_2 = ij_2 + nq;
    const int isy = (iy - 1) * nq;

    double rho_1, rho_2;
    double enthalpies_1[nq], enthalpies_2[nq];
    double w_1[nq], w_2[nq];
    double rhs_1[nq], rhs_2[nq];

    const double T_1 = state[ij_1];
    const double T_2 = state[ij_2];
    double y_1[nq];
    double y_2[nq];
    extract_y(&state[ij_1 + 1], nq, y_1);
    extract_y(&state[ij_2 + 1], nq, y_2);

    const double mmw_1 = mixture_molecular_weight(y_1);
    const double mmw_2 = mixture_molecular_weight(y_2);
    ideal_gas_density(pressure, T_1, mmw_1, &rho_1);
    ideal_gas_density(pressure, T_2, mmw_2, &rho_2);
    const double cp_1 = cp_mix(T_1, y_1);
    const double cp_2 = cp_mix(T_2, y_2);

    species_enthalpies(T_1, enthalpies_1);
    production_rates(T_1, rho_1, mmw_1, y_1, w_1);
    chem_rhs_isobaric(rho_1, cp_1, enthalpies_1, w_1, rhs_1);
    species_enthalpies(T_2, enthalpies_2);
    production_rates(T_2, rho_2, mmw_2, y_2, w_2);
    chem_rhs_isobaric(rho_2, cp_2, enthalpies_2, w_2, rhs_2);

    for (int iq = 0; iq < nq; ++iq)
    {
      out_rhs[ij_1 + iq] += yct[isy + iq] * state[ijp1_1 + iq] + ycb[isy + iq] * state[ijm1_1 + iq] + ycp[isy + iq] * state[ij_1 + iq] + rhs_1[iq];
      out_rhs[ij_2 + iq] += yct[isy + iq] * state[ijp1_2 + iq] + ycb[isy + iq] * state[ijm1_2 + iq] + ycp[isy + iq] * state[ij_2 + iq] + rhs_2[iq];
    }
  }

  for (int iq = 0; iq < nq; ++iq)
  {
    out_rhs[iq] = 0.;
    out_rhs[(ny - 1) * nq + iq] = 0.;
    out_rhs[(nx - 1) * nyq + iq] = 0.;
    out_rhs[(nx - 1) * nyq + (ny - 1) * nq + iq] = 0.;
  }
}

void CombustionKernels::flamelet2d_factored_block_diag_jacobian(const double *state, const double &pressure, const int &nx,
                                                                const int &ny, const double *xcp, const double *ycp,
                                                                const double &prefactor, double *out_values,
                                                                double *out_factors, int *out_pivots) const
{
  const int nq = mechanismData.phaseData.nSpecies;
  const int nyq = ny * nq;
  const int nyqq = nyq * nq;
  const int nqq = nq * nq;
  for (int ix = 1; ix < nx - 1; ++ix)
  {
    const int i = ix * nyq;
    const int ijac = ix * nyqq;
    const int isx = (ix - 1) * nq;
    for (int iy = 1; iy < ny - 1; ++iy)
    {
      const int ij = i + iy * nq;
      const int ijjac = ijac + iy * nqq;
      const int isy = (iy - 1) * nq;

      double rho, cp, cpsensT;
      double cpi[nq], cpisensT[nq], rhsTemp[nq], enthalpies[nq], w[nq], wsens[(nq + 1) * (nq + 1)], primJac[nq * (nq + 1)];
      double jac[nqq];
      const double T = state[ij];
      double y[nq];
      extract_y(&state[ij + 1], nq, y);
      const double mmw = mixture_molecular_weight(y);
      ideal_gas_density(pressure, T, mmw, &rho);
      cp_mix_and_species(T, y, &cp, cpi);
      species_enthalpies(T, enthalpies);
      cp_sens_T(T, y, &cpsensT, cpisensT);
      prod_rates_sens_exact(T, rho, mmw, y, w, wsens);
      chem_jac_isobaric(pressure, T, y, mmw, rho, cp, cpi, cpsensT, enthalpies, w, wsens, rhsTemp, primJac);
      transform_isobaric_primitive_jacobian(rho, pressure, T, mmw, primJac, jac);
      for (int iq = 0; iq < nqq; ++iq)
      {
        jac[iq] *= prefactor;
      }
      for (int iq = 0; iq < nq; ++iq)
      {
        jac[iq * (nq + 1)] += prefactor * (xcp[isx + iq] + ycp[isy + iq]) - 1.;
      }
      for (int iq = 0; iq < nqq; ++iq)
      {
        out_values[ijjac + iq] = jac[iq];
      }
      griffon::lapack::lu_factorize_with_copy(nq, jac, &out_pivots[ij], &out_factors[ijjac]);
    }
    const int ij_1 = ix * nyq;
    const int ijac_1 = ix * nyqq;
    const int ijjac_1 = ijac_1;
    const int ij_2 = ix * nyq + (ny - 1) * nq;
    const int ijac_2 = ix * nyqq;
    const int ijjac_2 = ijac_2 + (ny - 1) * nqq;

    {
      double rho, cp, cpsensT;
      double cpi[nq], cpisensT[nq], rhsTemp[nq], enthalpies[nq], w[nq], wsens[(nq + 1) * (nq + 1)], primJac[nq * (nq + 1)];
      double jac[nqq];
      const double T = state[ij_1];
      double y[nq];
      extract_y(&state[ij_1 + 1], nq, y);
      const double mmw = mixture_molecular_weight(y);
      ideal_gas_density(pressure, T, mmw, &rho);
      cp_mix_and_species(T, y, &cp, cpi);
      species_enthalpies(T, enthalpies);
      cp_sens_T(T, y, &cpsensT, cpisensT);
      prod_rates_sens_exact(T, rho, mmw, y, w, wsens);
      chem_jac_isobaric(pressure, T, y, mmw, rho, cp, cpi, cpsensT, enthalpies, w, wsens, rhsTemp, primJac);
      transform_isobaric_primitive_jacobian(rho, pressure, T, mmw, primJac, jac);
      for (int iq = 0; iq < nqq; ++iq)
      {
        jac[iq] *= prefactor;
      }
      for (int iq = 0; iq < nq; ++iq)
      {
        jac[iq * (nq + 1)] += prefactor * xcp[isx + iq] - 1.;
      }
      for (int iq = 0; iq < nqq; ++iq)
      {
        out_values[ijjac_1 + iq] = jac[iq];
      }
      griffon::lapack::lu_factorize_with_copy(nq, jac, &out_pivots[ij_1], &out_factors[ijjac_1]);
    }
    {
      double rho, cp, cpsensT;
      double cpi[nq], cpisensT[nq], rhsTemp[nq], enthalpies[nq], w[nq], wsens[(nq + 1) * (nq + 1)], primJac[nq * (nq + 1)];
      double jac[nqq];
      const double T = state[ij_2];
      double y[nq];
      extract_y(&state[ij_2 + 1], nq, y);
      const double mmw = mixture_molecular_weight(y);
      ideal_gas_density(pressure, T, mmw, &rho);
      cp_mix_and_species(T, y, &cp, cpi);
      species_enthalpies(T, enthalpies);
      cp_sens_T(T, y, &cpsensT, cpisensT);
      prod_rates_sens_exact(T, rho, mmw, y, w, wsens);
      chem_jac_isobaric(pressure, T, y, mmw, rho, cp, cpi, cpsensT, enthalpies, w, wsens, rhsTemp, primJac);
      transform_isobaric_primitive_jacobian(rho, pressure, T, mmw, primJac, jac);
      for (int iq = 0; iq < nqq; ++iq)
      {
        jac[iq] *= prefactor;
      }
      for (int iq = 0; iq < nq; ++iq)
      {
        jac[iq * (nq + 1)] += prefactor * xcp[isx + iq] - 1.;
      }
      for (int iq = 0; iq < nqq; ++iq)
      {
        out_values[ijjac_2 + iq] = jac[iq];
      }
      griffon::lapack::lu_factorize_with_copy(nq, jac, &out_pivots[ij_2], &out_factors[ijjac_2]);
    }
  }

  for (int iy = 1; iy < ny - 1; ++iy)
  {
    const int isy = (iy - 1) * nq;
    const int ijac_1 = 0;
    const int ijjac_1 = ijac_1 + iy * nqq;
    const int ij_1 = iy * nq;
    const int ijac_2 = (nx - 1) * nyqq;
    const int ijjac_2 = ijac_2 + iy * nqq;
    const int ij_2 = (nx - 1) * nyq + iy * nq;

    {
      double rho, cp, cpsensT;
      double cpi[nq], cpisensT[nq], rhsTemp[nq], enthalpies[nq], w[nq], wsens[(nq + 1) * (nq + 1)], primJac[nq * (nq + 1)];
      double jac[nqq];
      const double T = state[ij_1];
      double y[nq];
      extract_y(&state[ij_1 + 1], nq, y);
      const double mmw = mixture_molecular_weight(y);
      ideal_gas_density(pressure, T, mmw, &rho);
      cp_mix_and_species(T, y, &cp, cpi);
      species_enthalpies(T, enthalpies);
      cp_sens_T(T, y, &cpsensT, cpisensT);
      prod_rates_sens_exact(T, rho, mmw, y, w, wsens);
      chem_jac_isobaric(pressure, T, y, mmw, rho, cp, cpi, cpsensT, enthalpies, w, wsens, rhsTemp, primJac);
      transform_isobaric_primitive_jacobian(rho, pressure, T, mmw, primJac, jac);
      for (int iq = 0; iq < nqq; ++iq)
      {
        jac[iq] *= prefactor;
      }
      for (int iq = 0; iq < nq; ++iq)
      {
        jac[iq * (nq + 1)] += prefactor * ycp[isy + iq] - 1.;
      }
      for (int iq = 0; iq < nqq; ++iq)
      {
        out_values[ijjac_1 + iq] = jac[iq];
      }
      griffon::lapack::lu_factorize_with_copy(nq, jac, &out_pivots[ij_1], &out_factors[ijjac_1]);
    }
    {
      double rho, cp, cpsensT;
      double cpi[nq], cpisensT[nq], rhsTemp[nq], enthalpies[nq], w[nq], wsens[(nq + 1) * (nq + 1)], primJac[nq * (nq + 1)];
      double jac[nqq];
      const double T = state[ij_2];
      double y[nq];
      extract_y(&state[ij_2 + 1], nq, y);
      const double mmw = mixture_molecular_weight(y);
      ideal_gas_density(pressure, T, mmw, &rho);
      cp_mix_and_species(T, y, &cp, cpi);
      species_enthalpies(T, enthalpies);
      cp_sens_T(T, y, &cpsensT, cpisensT);
      prod_rates_sens_exact(T, rho, mmw, y, w, wsens);
      chem_jac_isobaric(pressure, T, y, mmw, rho, cp, cpi, cpsensT, enthalpies, w, wsens, rhsTemp, primJac);
      transform_isobaric_primitive_jacobian(rho, pressure, T, mmw, primJac, jac);
      for (int iq = 0; iq < nqq; ++iq)
      {
        jac[iq] *= prefactor;
      }
      for (int iq = 0; iq < nq; ++iq)
      {
        jac[iq * (nq + 1)] += prefactor * ycp[isy + iq] - 1.;
      }
      for (int iq = 0; iq < nqq; ++iq)
      {
        out_values[ijjac_2 + iq] = jac[iq];
      }
      griffon::lapack::lu_factorize_with_copy(nq, jac, &out_pivots[ij_2], &out_factors[ijjac_2]);
    }
  }
  double jac[nqq];
  for (int iq = 0; iq < nq; ++iq)
  {
    jac[iq * (nq + 1)] = prefactor - 1.;
  }
  griffon::lapack::lu_factorize_with_copy(nq, jac, &out_pivots[0], &out_factors[0]);
  for (int iq = 0; iq < nqq; ++iq)
  {
    out_values[iq] = jac[iq];
    out_values[(ny - 1) * nqq + iq] = jac[iq];
    out_values[(nx - 1) * nyqq + iq] = jac[iq];
    out_values[(nx - 1) * nyqq + (ny - 1) * nqq + iq] = jac[iq];
  }
  for (int iq = 0; iq < nq; ++iq)
  {
    out_pivots[(ny - 1) * nq + iq] = out_pivots[iq];
    out_pivots[(nx - 1) * nyq + iq] = out_pivots[iq];
    out_pivots[(nx - 1) * nyq + (ny - 1) * nq + iq] = out_pivots[iq];
    for (int jq = 0; jq < nq; ++jq)
    {
      out_factors[(ny - 1) * nqq + iq * nq + jq] = out_factors[iq * nq + jq];
      out_factors[(nx - 1) * nyqq + iq * nq + jq] = out_factors[iq * nq + jq];
      out_factors[(nx - 1) * nyqq + (ny - 1) * nqq + iq * nq + jq] = out_factors[iq * nq + jq];
    }
  }
}

void CombustionKernels::flamelet2d_offdiag_matvec(const double *vec, const int &nx, const int &ny, const double *xcp,
                                                  const double *xcl, const double *xcr, const double *ycp,
                                                  const double *ycb, const double *yct, const double &prefactor,
                                                  double *out_vec) const
{
  const int nq = mechanismData.phaseData.nSpecies;
  const int nyq = ny * nq;
  for (int ix = 1; ix < nx - 1; ++ix)
  {
    const int i = ix * nyq;
    const int isx = (ix - 1) * nq;
    for (int iy = 1; iy < ny - 1; ++iy)
    {
      const int ij = i + iy * nq;
      const int ijm1 = ij - nq;
      const int ijp1 = ij + nq;
      const int im1j = ij - nyq;
      const int ip1j = ij + nyq;
      const int isy = (iy - 1) * nq;
      for (int iq = 0; iq < nq; ++iq)
      {
        out_vec[ij + iq] += prefactor * (xcr[isx + iq] * vec[ip1j + iq] + xcl[isx + iq] * vec[im1j + iq] + yct[isy + iq] * vec[ijp1 + iq] + ycb[isy + iq] * vec[ijm1 + iq]);
      }
    }
    const int ij_1 = ix * nyq;
    const int im1j_1 = ij_1 - nyq;
    const int ip1j_1 = ij_1 + nyq;
    const int ij_2 = ix * nyq + (ny - 1) * nq;
    const int im1j_2 = ij_2 - nyq;
    const int ip1j_2 = ij_2 + nyq;
    for (int iq = 0; iq < nq; ++iq)
    {
      out_vec[ij_1 + iq] += prefactor * (xcr[isx + iq] * vec[ip1j_1 + iq] + xcl[isx + iq] * vec[im1j_1 + iq]);
      out_vec[ij_2 + iq] += prefactor * (xcr[isx + iq] * vec[ip1j_2 + iq] + xcl[isx + iq] * vec[im1j_2 + iq]);
    }
  }

  for (int iy = 1; iy < ny - 1; ++iy)
  {
    const int ij_1 = iy * nq;
    const int ijm1_1 = ij_1 - nq;
    const int ijp1_1 = ij_1 + nq;
    const int ij_2 = (nx - 1) * nyq + iy * nq;
    const int ijm1_2 = ij_2 - nq;
    const int ijp1_2 = ij_2 + nq;
    const int isy = (iy - 1) * nq;
    for (int iq = 0; iq < nq; ++iq)
    {
      out_vec[ij_1 + iq] += prefactor * (yct[isy + iq] * vec[ijp1_1 + iq] + ycb[isy + iq] * vec[ijm1_1 + iq]);
      out_vec[ij_2 + iq] += prefactor * (yct[isy + iq] * vec[ijp1_2 + iq] + ycb[isy + iq] * vec[ijm1_2 + iq]);
    }
  }

  for (int iq = 0; iq < nq; ++iq)
  {
    out_vec[iq] = 0.;
    out_vec[(ny - 1) * nq + iq] = 0.;
    out_vec[(nx - 1) * nyq + iq] = 0.;
    out_vec[(nx - 1) * nyq + (ny - 1) * nq + iq] = 0.;
  }
}

void CombustionKernels::flamelet2d_matvec(const double *vec, const int &nx, const int &ny, const double *xcp,
                                          const double *xcl, const double *xcr, const double *ycp, const double *ycb,
                                          const double *yct, const double &prefactor, const double *block_diag_values,
                                          double *out_vec) const
{
  const int nq = mechanismData.phaseData.nSpecies;
  const int nqq = nq * nq;
  const int nyq = ny * nq;
  const int nyqq = nyq * nq;
  for (int ix = 1; ix < nx - 1; ++ix)
  {
    const int i = ix * nyq;
    const int ijac = ix * nyqq;
    const int isx = (ix - 1) * nq;
    for (int iy = 1; iy < ny - 1; ++iy)
    {
      const int ij = i + iy * nq;
      const int ijm1 = ij - nq;
      const int ijp1 = ij + nq;
      const int im1j = ij - nyq;
      const int ip1j = ij + nyq;
      const int isy = (iy - 1) * nq;
      const int ijjac = ijac + iy * nqq;

      griffon::blas::matrix_vector_multiply(nq, &out_vec[ij], 1., &block_diag_values[ijjac], &vec[ij], 1.);

      for (int iq = 0; iq < nq; ++iq)
      {
        out_vec[ij + iq] += prefactor * (xcr[isx + iq] * vec[ip1j + iq] + xcl[isx + iq] * vec[im1j + iq] + yct[isy + iq] * vec[ijp1 + iq] + ycb[isy + iq] * vec[ijm1 + iq]);
      }
    }
    const int ij_1 = ix * nyq;
    const int im1j_1 = ij_1 - nyq;
    const int ip1j_1 = ij_1 + nyq;
    const int ijac_1 = ix * nyqq;
    const int ijjac_1 = ijac_1;
    const int ij_2 = ix * nyq + (ny - 1) * nq;
    const int im1j_2 = ij_2 - nyq;
    const int ip1j_2 = ij_2 + nyq;
    const int ijac_2 = ix * nyqq;
    const int ijjac_2 = ijac_2 + (ny - 1) * nqq;

    griffon::blas::matrix_vector_multiply(nq, &out_vec[ij_1], 1., &block_diag_values[ijjac_1], &vec[ij_1], 1.);
    griffon::blas::matrix_vector_multiply(nq, &out_vec[ij_2], 1., &block_diag_values[ijjac_2], &vec[ij_2], 1.);
    for (int iq = 0; iq < nq; ++iq)
    {
      out_vec[ij_1 + iq] += prefactor * (xcr[isx + iq] * vec[ip1j_1 + iq] + xcl[isx + iq] * vec[im1j_1 + iq]);
      out_vec[ij_2 + iq] += prefactor * (xcr[isx + iq] * vec[ip1j_2 + iq] + xcl[isx + iq] * vec[im1j_2 + iq]);
    }
  }

  for (int iy = 1; iy < ny - 1; ++iy)
  {
    const int isy = (iy - 1) * nq;
    const int ijac_1 = 0;
    const int ijjac_1 = ijac_1 + iy * nqq;
    const int ij_1 = iy * nq;
    const int ijp1_1 = ij_1 + nq;
    const int ijac_2 = (nx - 1) * nyqq;
    const int ijjac_2 = ijac_2 + iy * nqq;
    const int ij_2 = (nx - 1) * nyq + iy * nq;
    const int ijp1_2 = ij_2 + nq;

    griffon::blas::matrix_vector_multiply(nq, &out_vec[ij_1], 1., &block_diag_values[ijjac_1], &vec[ij_1], 1.);
    griffon::blas::matrix_vector_multiply(nq, &out_vec[ij_2], 1., &block_diag_values[ijjac_2], &vec[ij_2], 1.);
    for (int iq = 0; iq < nq; ++iq)
    {
      out_vec[ij_1 + iq] += prefactor * (yct[isy + iq] * vec[ijp1_1 + iq] + ycp[isy + iq] * vec[ij_1 + iq]);
      out_vec[ij_2 + iq] += prefactor * (yct[isy + iq] * vec[ijp1_2 + iq] + ycp[isy + iq] * vec[ij_2 + iq]);
    }
  }

  for (int iq = 0; iq < nq; ++iq)
  {
    out_vec[iq] = 0.;
    out_vec[(ny - 1) * nq + iq] = 0.;
    out_vec[(nx - 1) * nyq + iq] = 0.;
    out_vec[(nx - 1) * nyq + (ny - 1) * nq + iq] = 0.;
  }
}

void CombustionKernels::flamelet2d_block_diag_solve(const int &nx, const int &ny, const double *factors, const int *pivots,
                                                    const double *b, double *out_x) const
{
  const int nq = mechanismData.phaseData.nSpecies;
  const int nxy = nx * ny;
  const int nqq = nq * nq;

  for (int id = 0; id < nxy; ++id)
  {
    griffon::lapack::lu_solve_with_copy(nq, &factors[id * nqq], &pivots[id * nq], &b[id * nq], &out_x[id * nq]);
  }
}
} // namespace griffon
