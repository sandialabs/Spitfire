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

#define GRIFFON_SUM2(prop) (prop(0) + prop(1))
#define GRIFFON_SUM3(prop) (GRIFFON_SUM2(prop) + prop(2))
#define GRIFFON_SUM4(prop) (GRIFFON_SUM3(prop) + prop(3))
#define GRIFFON_SUM5(prop) (GRIFFON_SUM4(prop) + prop(4))
#define GRIFFON_SUM6(prop) (GRIFFON_SUM5(prop) + prop(5))
#define GRIFFON_SUM7(prop) (GRIFFON_SUM6(prop) + prop(6))
#define GRIFFON_SUM8(prop) (GRIFFON_SUM7(prop) + prop(7))
#define THD_BDY(i) (tb_efficiencies[(0)] * y[tb_indices[(0) * nzi + i]])
#define GIBBS(i) (net_stoich[(i)] * gi[net_indices[(i)] * nzi + i])

namespace griffon
{

void CombustionKernels::flamelet_stencils(const double *dz, const int &nzi, const double *dissipationRate,
                                          const double *invLewisNumbers, double *out_cmajor, double *out_csub,
                                          double *out_csup, double *out_mcoeff, double *out_ncoeff) const
{
  const int nSpec = mechanismData.phaseData.nSpecies;
  for (int i = 0; i < nzi; ++i)
  {
    const double dzt = dz[i] + dz[i + 1];
    for (int l = 0; l < nSpec; ++l)
    {
      out_cmajor[i * nSpec + l] = -dissipationRate[1 + i] / (dz[i] * dz[i + 1]) * invLewisNumbers[l];
      out_csub[i * nSpec + l] = dissipationRate[1 + i] / (dzt * dz[i]) * invLewisNumbers[l];
      out_csup[i * nSpec + l] = dissipationRate[1 + i] / (dzt * dz[i + 1]) * invLewisNumbers[l];
    }
    out_ncoeff[i] = 1 / (dz[i] + dz[i + 1]);
    out_mcoeff[i] = -out_ncoeff[i];
  }
}

void CombustionKernels::flamelet_jac_indices(const int &nzi, int *out_row_indices, int *out_col_indices) const
{
  const int nSpec = mechanismData.phaseData.nSpecies;
  int idx = 0;
  // diagonal blocks
  for (int iz = 0; iz < nzi; ++iz)
  {
    const int blockBase = iz * nSpec;
    for (int iq = 0; iq < nSpec; ++iq)
    {
      for (int jq = 0; jq < nSpec; ++jq)
      {
        out_row_indices[idx] = blockBase + jq;
        out_col_indices[idx] = blockBase + iq;
        ++idx;
      }
    }
  }
  // subdiagonal
  for (int iz = 1; iz < nzi; ++iz)
  {
    const int blockBase = iz * nSpec;
    for (int iq = 0; iq < nSpec; ++iq)
    {
      out_row_indices[idx] = blockBase + iq;
      out_col_indices[idx] = blockBase + iq - nSpec;
      ++idx;
    }
  }
  // superdiagonal
  for (int iz = 0; iz < nzi - 1; ++iz)
  {
    const int blockBase = iz * nSpec;
    for (int iq = 0; iq < nSpec; ++iq)
    {
      out_row_indices[idx] = blockBase + iq;
      out_col_indices[idx] = blockBase + iq + nSpec;
      ++idx;
    }
  }
}

// ------------------------------------------------------------------------------------------------

void CombustionKernels::flamelet_rhs_test1(const double *state, const double &pressure, const double *oxyState, const double *fuelState,
                                           const bool adiabatic, const double *T_convection, const double *h_convection,
                                           const double *T_radiation, const double *h_radiation, const int &nzi, const double *cmajor,
                                           const double *csub, const double *csup, const double *mcoeff, const double *ncoeff,
                                           const double *chi, const bool include_enthalpy_flux, const bool include_variable_cp,
                                           const bool use_scaled_heat_loss, double *out_rhs) const
{
  const int ns = mechanismData.phaseData.nSpecies;
  const auto minv = mechanismData.phaseData.inverseMolecularWeights.data();
  const auto nr = mechanismData.reactionData.nReactions;
  const auto ps = mechanismData.phaseData.referencePressure;
  const auto invR = 1. / mechanismData.phaseData.Ru;
  const double poverRu = pressure * invR;
  const double psoverRu = ps * invR;

  const auto &cpTypes = mechanismData.heatCapacityData.types;
  const auto &cpCoeffs = mechanismData.heatCapacityData.coefficients;
  const auto &cpMaxTs = mechanismData.heatCapacityData.maxTemperatures;
  const auto &cpMinTs = mechanismData.heatCapacityData.minTemperatures;

  double T[nzi];
  double y[nzi * ns];
  double maxT = -1;
  for (int i = 0; i < nzi; ++i)
  {
    const int offsetTY = i * ns + 1;
    const int offsetY = i;
    T[i] = state[offsetTY];
    maxT = std::max(maxT, T[i]);
    y[offsetY + (ns - 1) * nzi] = 1.;
    for (int k = 0; k < ns - 1; ++k)
    {
      y[offsetY + k * nzi] = state[offsetTY + k];
      y[offsetY + (ns - 1) * nzi] -= y[offsetY + k * nzi];
    }
  }
  double maxT4 = maxT * maxT * maxT * maxT;

  double logT[nzi];
  double invT[nzi];
  double M[nzi];
  double rho[nzi];
  double conc[nzi];
  double cp[nzi];
  double cpi[nzi * ns];
  double hi[nzi * ns];
  double gi[nzi * ns];
  double wi[nzi * ns];

  double Tz[nzi];
  double yz[nzi * ns];
  double cpz[nzi];
  double yzcpi[nzi];

  double kf[nzi];
  double kr[nzi];
  double mtb[nzi];
  double pr[nzi];
  double logPrC[nzi];
  double logFCent[nzi];

  for (int k = 0; k < ns; ++k)
  {
    const double minvk = minv[k];
    const int offset = k * nzi;
    for (int i = 0; i < nzi; ++i)
    {
      M[i] += minvk * y[offset + i];
    }
  }
  for (int i = 0; i < nzi; ++i)
  {
    logT[i] = std::log(T[i]);
    invT[i] = 1. / T[i];
    M[i] = 1. / M[i];
    rho[i] = poverRu * M[i] * invT[i];
    conc[i] = poverRu * invT[i];
  }

  for (int k = 0; k < ns; ++k)
  {
    const auto polyType = cpTypes[k];
    const auto &c = cpCoeffs[k];
    const double maxT = cpMaxTs[k];
    const double minT = cpMinTs[k];

    const int offset = k * nzi;
    const double minvk = minv[k];

    switch (polyType)
    {
    case CpType::CONST:
      for (int i = 0; i < nzi; ++i)
      {
        const double t = T[i];
        const double logt = logT[i];
        cpi[offset + i] = minvk * c[3];
        cp[i] += y[offset + i] * cpi[offset + i];
        hi[offset + i] = minvk * (c[1] + c[3] * (t - c[0]));
        gi[offset + i] = c[1] + c[3] * (t - c[0]) - t * (c[2] + c[3] * (logt - log(c[0])));
      }
      break;
    case CpType::NASA7:
      for (int i = 0; i < nzi; ++i)
      {
        const double t = T[i];
        const double logt = logT[i];

        if (t <= c[0] && t >= minT)
        {
          cpi[offset + i] = minvk * (c[8] + t * (2. * c[9] + t * (6. * c[10] + t * (12. * c[11] + 20. * t * c[12]))));
          cp[i] += y[offset + i] * cpi[offset + i];
          hi[offset + i] = minvk * (c[13] + t * (c[8] + t * (c[9] + t * (2. * c[10] + t * (3. * c[11] + t * 4. * c[12])))));
          gi[offset + i] = c[13] + t * (c[8] - c[14] - c[8] * logt - t * (c[9] + t * (c[10] + t * (c[11] + t * c[12]))));
          break;
        }
        else if (t > c[0] && t <= maxT)
        {
          cpi[offset + i] = minvk * (c[1] + t * (2. * c[2] + t * (6. * c[3] + t * (12. * c[4] + 20. * t * c[5]))));
          cp[i] += y[offset + i] * cpi[offset + i];
          hi[offset + i] = minvk * (c[6] + t * (c[1] + t * (c[2] + t * (2. * c[3] + t * (3. * c[4] + t * 4. * c[5])))));
          gi[offset + i] = c[6] + t * (c[1] - c[7] - c[1] * logt - t * (c[2] + t * (c[3] + t * (c[4] + t * c[5]))));
          break;
        }
        else if (t < minT)
        {
          cpi[offset + i] = minvk * (c[8] + minT * (2. * c[9] + minT * (6. * c[10] + minT * (12. * c[11] + 20. * minT * c[12]))));
          cp[i] += y[offset + i] * cpi[offset + i];
          hi[offset + i] = minvk * (c[13] + c[8] * t + minT * (2. * c[9] * t + minT * (3. * 2. * c[10] * t - c[9] + minT * (4. * 3. * c[11] * t - 2. * 2. * c[10] + minT * (5. * 4. * c[12] * t - 3. * 3. * c[11] + minT * -4. * 4. * c[12])))));
          gi[offset + i] = c[13] + t * (c[8] - c[14] - c[8] * logt - t * (c[9] + t * (c[10] + t * (c[11] + t * c[12]))));
          break;
        }
        else
        {
          cpi[offset + i] = minvk * (c[1] + maxT * (2. * c[2] + maxT * (6. * c[3] + maxT * (12. * c[4] + 20. * maxT * c[5]))));
          cp[i] += y[offset + i] * cpi[offset + i];
          hi[offset + i] = minvk * (c[6] + c[1] * t + maxT * (2. * c[2] * t + maxT * (3. * 2. * c[3] * t - c[2] + maxT * (4. * 3. * c[4] * t - 2. * 2. * c[3] + maxT * (5. * 4. * c[5] * t - 3. * 3. * c[4] + maxT * -4. * 4. * c[5])))));
          gi[offset + i] = c[6] + t * (c[1] - c[7] - c[1] * logt - t * (c[2] + t * (c[3] + t * (c[4] + t * c[5]))));
          break;
        }
      }
    default:
      break;
    }
  }

  {
    for (int j = 0; j < ns * nzi; ++j)
    {
      wi[j] = 0.;
    }

    for (int r = 0; r < nr; ++r)
    {
      const auto &rxnData = mechanismData.reactionData.reactions[r];

      const auto &tb_indices = rxnData.tb_indices;
      const auto &tb_efficiencies = rxnData.tb_efficiencies;
      const auto n_tb = rxnData.n_tb;

      const auto &rc_indices = rxnData.reactant_indices;
      const auto &rc_stoich = rxnData.reactant_stoich;
      const auto &rc_invmw = rxnData.reactant_invmw;
      const auto n_rc = rxnData.n_reactants;

      const auto &pd_indices = rxnData.product_indices;
      const auto &pd_stoich = rxnData.product_stoich;
      const auto &pd_invmw = rxnData.product_invmw;
      const auto n_pd = rxnData.n_products;

      const auto &sp_indices = rxnData.special_indices;
      const auto &sp_order = rxnData.special_orders;
      const auto &sp_invmw = rxnData.special_invmw;
      const auto &sp_nonzero = rxnData.special_nonzero;
      const auto n_sp = rxnData.n_special;

      const auto &net_indices = rxnData.net_indices;
      const auto &net_stoich = rxnData.net_stoich;
      const auto &net_mw = rxnData.net_mw;
      const auto n_net = rxnData.n_net;

      const auto &kCoefs = rxnData.kFwdCoefs;
      const auto &kPCoefs = rxnData.kPressureCoefs;
      const auto baseEff = rxnData.thdBdyDefault;
      const auto &troe = rxnData.troeParams;

      switch (rxnData.kForm)
      {
      case RateConstantTForm::CONSTANT:
        for (int i = 0; i < nzi; ++i)
        {
          kf[i] = kCoefs[0];
        }
        break;
      case RateConstantTForm::LINEAR:
        for (int i = 0; i < nzi; ++i)
        {
          kf[i] = kCoefs[0] * T[i];
        }
        break;
      case RateConstantTForm::QUADRATIC:
        for (int i = 0; i < nzi; ++i)
        {
          kf[i] = kCoefs[0] * T[i] * T[i];
        }
        break;
      case RateConstantTForm::RECIPROCAL:
        for (int i = 0; i < nzi; ++i)
        {
          kf[i] = kCoefs[0] / T[i];
        }
        break;
      case RateConstantTForm::ARRHENIUS:
        for (int i = 0; i < nzi; ++i)
        {
          kf[i] = kCoefs[0] * std::exp(kCoefs[1] * logT[i] - kCoefs[2] * invT[i]);
        }
        break;
      }

      switch (rxnData.type)
      {
      case RateType::SIMPLE:
        break;
      case RateType::THIRD_BODY:
        switch (n_tb)
        {
        case 0:
          for (int i = 0; i < nzi; ++i)
          {
            kf[i] *= baseEff * conc[i];
          }
          break;
        case 1:
          for (int i = 0; i < nzi; ++i)
          {
            kf[i] *= baseEff * conc[i] + rho[i] * (THD_BDY(0));
          }
          break;
        case 2:
          for (int i = 0; i < nzi; ++i)
          {
            kf[i] *= baseEff * conc[i] + rho[i] * (GRIFFON_SUM2(THD_BDY));
          }
          break;
        case 3:
          for (int i = 0; i < nzi; ++i)
          {
            kf[i] *= baseEff * conc[i] + rho[i] * (GRIFFON_SUM3(THD_BDY));
          }
          break;
        case 4:
          for (int i = 0; i < nzi; ++i)
          {
            kf[i] *= baseEff * conc[i] + rho[i] * (GRIFFON_SUM4(THD_BDY));
          }
          break;
        case 5:
          for (int i = 0; i < nzi; ++i)
          {
            kf[i] *= baseEff * conc[i] + rho[i] * (GRIFFON_SUM5(THD_BDY));
          }
          break;
        case 6:
          for (int i = 0; i < nzi; ++i)
          {
            kf[i] *= baseEff * conc[i] + rho[i] * (GRIFFON_SUM6(THD_BDY));
          }
          break;
        case 7:
          for (int i = 0; i < nzi; ++i)
          {
            kf[i] *= baseEff * conc[i] + rho[i] * (GRIFFON_SUM7(THD_BDY));
          }
          break;
        case 8:
          for (int i = 0; i < nzi; ++i)
          {
            kf[i] *= baseEff * conc[i] + rho[i] * (GRIFFON_SUM8(THD_BDY));
          }
          break;
        default:
          for (int i = 0; i < nzi; ++i)
          {
            mtb[i] = baseEff * conc[i] + rho[i] * (GRIFFON_SUM8(THD_BDY));
          }
          for (int i = 8; i != n_tb; ++i)
          {
            for (int i = 0; i < nzi; ++i)
            {
              mtb[i] = rho[i] * THD_BDY(i);
            }
          }
          for (int i = 0; i < nzi; ++i)
          {
            kf[i] *= mtb[i];
          }
          break;
        }
        break;
      case RateType::LINDEMANN:
        switch (n_tb)
        {
        case 0:
          for (int i = 0; i < nzi; ++i)
          {
            kf[i] /= (1. + kf[i] / ((kPCoefs[0] * std::exp(kPCoefs[1] * logT[i] - kPCoefs[2] * invT[i])) * (baseEff * conc[i])));
          }
          break;
        case 1:
          for (int i = 0; i < nzi; ++i)
          {
            kf[i] /= (1. + kf[i] / ((kPCoefs[0] * std::exp(kPCoefs[1] * logT[i] - kPCoefs[2] * invT[i])) * (baseEff * conc[i] + rho[i] * (THD_BDY(0)))));
          }
          break;
        case 2:
          for (int i = 0; i < nzi; ++i)
          {
            kf[i] /= (1. + kf[i] / ((kPCoefs[0] * std::exp(kPCoefs[1] * logT[i] - kPCoefs[2] * invT[i])) * (baseEff * conc[i] + rho[i] * (GRIFFON_SUM2(THD_BDY)))));
          }
          break;
        case 3:
          for (int i = 0; i < nzi; ++i)
          {
            kf[i] /= (1. + kf[i] / ((kPCoefs[0] * std::exp(kPCoefs[1] * logT[i] - kPCoefs[2] * invT[i])) * (baseEff * conc[i] + rho[i] * (GRIFFON_SUM3(THD_BDY)))));
          }
          break;
        case 4:
          for (int i = 0; i < nzi; ++i)
          {
            kf[i] /= (1. + kf[i] / ((kPCoefs[0] * std::exp(kPCoefs[1] * logT[i] - kPCoefs[2] * invT[i])) * (baseEff * conc[i] + rho[i] * (GRIFFON_SUM4(THD_BDY)))));
          }
          break;
        case 5:
          for (int i = 0; i < nzi; ++i)
          {
            kf[i] /= (1. + kf[i] / ((kPCoefs[0] * std::exp(kPCoefs[1] * logT[i] - kPCoefs[2] * invT[i])) * (baseEff * conc[i] + rho[i] * (GRIFFON_SUM5(THD_BDY)))));
          }
          break;
        case 6:
          for (int i = 0; i < nzi; ++i)
          {
            kf[i] /= (1. + kf[i] / ((kPCoefs[0] * std::exp(kPCoefs[1] * logT[i] - kPCoefs[2] * invT[i])) * (baseEff * conc[i] + rho[i] * (GRIFFON_SUM6(THD_BDY)))));
          }
          break;
        case 7:
          for (int i = 0; i < nzi; ++i)
          {
            kf[i] /= (1. + kf[i] / ((kPCoefs[0] * std::exp(kPCoefs[1] * logT[i] - kPCoefs[2] * invT[i])) * (baseEff * conc[i] + rho[i] * (GRIFFON_SUM7(THD_BDY)))));
          }
          break;
        case 8:
          for (int i = 0; i < nzi; ++i)
          {
            kf[i] /= (1. + kf[i] / ((kPCoefs[0] * std::exp(kPCoefs[1] * logT[i] - kPCoefs[2] * invT[i])) * (baseEff * conc[i] + rho[i] * (GRIFFON_SUM8(THD_BDY)))));
          }
          break;
        default:
          for (int i = 0; i < nzi; ++i)
          {
            mtb[i] = baseEff * conc[i] + rho[i] * (GRIFFON_SUM8(THD_BDY));
          }
          for (int i = 8; i != n_tb; ++i)
          {
            for (int i = 0; i < nzi; ++i)
            {
              mtb[i] = rho[i] * THD_BDY(i);
            }
          }
          for (int i = 0; i < nzi; ++i)
          {
            kf[i] /= (1 + kf[i] / ((kPCoefs[0] * std::exp(kPCoefs[1] * logT[i] - kPCoefs[2] * invT[i])) * mtb[i]));
          }
          break;
        }
        break;
      case RateType::TROE:
        switch (n_tb)
        {
        case 0:
          for (int i = 0; i < nzi; ++i)
          {
            pr[i] = (kPCoefs[0] * std::exp(kPCoefs[1] * logT[i] - kPCoefs[2] * invT[i])) / kf[i] * (baseEff * conc[i]);
          }
          break;
        case 1:
          for (int i = 0; i < nzi; ++i)
          {
            pr[i] = (kPCoefs[0] * std::exp(kPCoefs[1] * logT[i] - kPCoefs[2] * invT[i])) / kf[i] * (baseEff * conc[i] + rho[i] * (THD_BDY(0)));
          }
          break;
        case 2:
          for (int i = 0; i < nzi; ++i)
          {
            pr[i] = (kPCoefs[0] * std::exp(kPCoefs[1] * logT[i] - kPCoefs[2] * invT[i])) / kf[i] * (baseEff * conc[i] + rho[i] * (GRIFFON_SUM2(THD_BDY)));
          }
          break;
        case 3:
          for (int i = 0; i < nzi; ++i)
          {
            pr[i] = (kPCoefs[0] * std::exp(kPCoefs[1] * logT[i] - kPCoefs[2] * invT[i])) / kf[i] * (baseEff * conc[i] + rho[i] * (GRIFFON_SUM3(THD_BDY)));
          }
          break;
        case 4:
          for (int i = 0; i < nzi; ++i)
          {
            pr[i] = (kPCoefs[0] * std::exp(kPCoefs[1] * logT[i] - kPCoefs[2] * invT[i])) / kf[i] * (baseEff * conc[i] + rho[i] * (GRIFFON_SUM4(THD_BDY)));
          }
          break;
        case 5:
          for (int i = 0; i < nzi; ++i)
          {
            pr[i] = (kPCoefs[0] * std::exp(kPCoefs[1] * logT[i] - kPCoefs[2] * invT[i])) / kf[i] * (baseEff * conc[i] + rho[i] * (GRIFFON_SUM5(THD_BDY)));
          }
          break;
        case 6:
          for (int i = 0; i < nzi; ++i)
          {
            pr[i] = (kPCoefs[0] * std::exp(kPCoefs[1] * logT[i] - kPCoefs[2] * invT[i])) / kf[i] * (baseEff * conc[i] + rho[i] * (GRIFFON_SUM6(THD_BDY)));
          }
          break;
        case 7:
          for (int i = 0; i < nzi; ++i)
          {
            pr[i] = (kPCoefs[0] * std::exp(kPCoefs[1] * logT[i] - kPCoefs[2] * invT[i])) / kf[i] * (baseEff * conc[i] + rho[i] * (GRIFFON_SUM7(THD_BDY)));
          }
          break;
        case 8:
          for (int i = 0; i < nzi; ++i)
          {
            pr[i] = (kPCoefs[0] * std::exp(kPCoefs[1] * logT[i] - kPCoefs[2] * invT[i])) / kf[i] * (baseEff * conc[i] + rho[i] * (GRIFFON_SUM8(THD_BDY)));
          }
          break;
        default:
          for (int i = 0; i < nzi; ++i)
          {
            mtb[i] = baseEff * conc[i] + rho[i] * (GRIFFON_SUM8(THD_BDY));
          }
          for (int i = 8; i != n_tb; ++i)
          {
            for (int i = 0; i < nzi; ++i)
            {
              mtb[i] = rho[i] * THD_BDY(i);
            }
          }
          for (int i = 0; i < nzi; ++i)
          {
            pr[i] = (kPCoefs[0] * std::exp(kPCoefs[1] * logT[i] - kPCoefs[2] * invT[i])) / kf[i] * mtb[i];
          }
          break;
        }

        switch (rxnData.troeForm)
        {
        case TroeTermsPresent::T123:
          for (int i = 0; i < nzi; ++i)
          {
            logFCent[i] = std::log10(
                (1 - troe[0]) * std::exp(-T[i] / troe[1]) + troe[0] * std::exp(-T[i] / troe[2]) + std::exp(-invT[i] * troe[3]));
          }
          break;
        case TroeTermsPresent::T12:
          for (int i = 0; i < nzi; ++i)
          {
            logFCent[i] = std::log10((1 - troe[0]) * std::exp(-T[i] / troe[1]) + troe[0] * std::exp(-T[i] / troe[2]) + 0.0);
          }
          break;
        case TroeTermsPresent::T1:
          for (int i = 0; i < nzi; ++i)
          {
            logFCent[i] = std::log10((1 - troe[0]) * std::exp(-T[i] / troe[1]) + 0.0 + 0.0);
          }
          break;
        case TroeTermsPresent::T23:
          for (int i = 0; i < nzi; ++i)
          {
            logFCent[i] = std::log10(0.0 + troe[0] * std::exp(-T[i] / troe[2]) + std::exp(-invT[i] * troe[3]));
          }
          break;
        case TroeTermsPresent::T2:
          for (int i = 0; i < nzi; ++i)
          {
            logFCent[i] = std::log10(0.0 + troe[0] * std::exp(-T[i] / troe[2]) + 0.0);
          }
          break;
        case TroeTermsPresent::T13:
          for (int i = 0; i < nzi; ++i)
          {
            logFCent[i] = std::log10((1 - troe[0]) * std::exp(-T[i] / troe[1]) + 0.0 + std::exp(-invT[i] * troe[3]));
          }
          break;
        case TroeTermsPresent::T3:
          for (int i = 0; i < nzi; ++i)
          {
            logFCent[i] = std::log10(0.0 + 0.0 + std::exp(-invT[i] * troe[3]));
          }
          break;
        case TroeTermsPresent::NO_TROE_TERMS:
        default:
        {
          throw std::runtime_error("no troe Terms flagged for evaluation");
        }
        }

#define CTROE (-0.4 - 0.67 * logFCent[i])
#define NTROE (0.75 - 1.27 * logFCent[i])
#define F1 (logPrC[i] / (NTROE - 0.14 * logPrC[i]))

        for (int i = 0; i < nzi; ++i)
        {
          logPrC[i] = std::log10(std::max(pr[i], 1.e-300)) + CTROE;
          kf[i] *= std::pow(10, logFCent[i] / (1. + F1 * F1)) * pr[i] / (1 + pr[i]);
        }
        break;
      default:
      {
        throw std::runtime_error("unidentified reaction");
      }
      }

#undef CTROE
#undef NTROE
#undef F1

      if (rxnData.hasOrders)
      {
        for (int k = 0; k < n_sp; ++k)
        {
          if (sp_nonzero[k])
          {
            const double sporderk = sp_order[k];
            const double spminvk = sp_invmw[k];
            const int offset = sp_indices[k] * nzi;
            for (int i = 0; i < nzi; ++i)
            {
              kf[i] *= std::pow(std::max(y[offset + i] * rho[i] * spminvk, 0.), sporderk);
              kr[i] = 0.;
            }
          }
        }
      }
      else
      {
        if (rxnData.reversible)
        {
          switch (n_net)
          {
          case 3:
            for (int i = 0; i < nzi; ++i)
            {
              kr[i] = kf[i] * std::exp(rxnData.sumStoich * std::log(psoverRu * invT[i]) - invT[i] * invR * (GRIFFON_SUM3(GIBBS)));
            }
            break;
          case 2:
            for (int i = 0; i < nzi; ++i)
            {
              kr[i] = kf[i] * std::exp(rxnData.sumStoich * std::log(psoverRu * invT[i]) - invT[i] * invR * (GRIFFON_SUM2(GIBBS)));
            }
            break;
          case 4:
            for (int i = 0; i < nzi; ++i)
            {
              kr[i] = kf[i] * std::exp(rxnData.sumStoich * std::log(psoverRu * invT[i]) - invT[i] * invR * (GRIFFON_SUM4(GIBBS)));
            }
            break;
          case 5:
            for (int i = 0; i < nzi; ++i)
            {
              kr[i] = kf[i] * std::exp(rxnData.sumStoich * std::log(psoverRu * invT[i]) - invT[i] * invR * (GRIFFON_SUM5(GIBBS)));
            }
            break;
          case 6:
            for (int i = 0; i < nzi; ++i)
            {
              kr[i] = kf[i] * std::exp(rxnData.sumStoich * std::log(psoverRu * invT[i]) - invT[i] * invR * (GRIFFON_SUM6(GIBBS)));
            }
            break;
          case 7:
            for (int i = 0; i < nzi; ++i)
            {
              kr[i] = kf[i] * std::exp(rxnData.sumStoich * std::log(psoverRu * invT[i]) - invT[i] * invR * (GRIFFON_SUM7(GIBBS)));
            }
            break;
          case 8:
            for (int i = 0; i < nzi; ++i)
            {
              kr[i] = kf[i] * std::exp(rxnData.sumStoich * std::log(psoverRu * invT[i]) - invT[i] * invR * (GRIFFON_SUM8(GIBBS)));
            }
            break;
          }
        }

#define C_R(j) (y[rc_indices[j] * nzi + i] * rho[i] * rc_invmw[j])
#define C_P(j) (y[pd_indices[j] * nzi + i] * rho[i] * pd_invmw[j])

        switch (rxnData.forwardOrder)
        {
        case ReactionOrder::ONE:
          for (int i = 0; i < nzi; ++i)
          {
            kf[i] *= C_R(0);
          }
          break;
        case ReactionOrder::TWO:
          for (int i = 0; i < nzi; ++i)
          {
            kf[i] *= C_R(0) * C_R(0);
          }
          break;
        case ReactionOrder::ONE_ONE:
          for (int i = 0; i < nzi; ++i)
          {
            kf[i] *= C_R(0) * C_R(1);
          }
          break;
        case ReactionOrder::ONE_ONE_ONE:
          for (int i = 0; i < nzi; ++i)
          {
            kf[i] *= C_R(0) * C_R(1) * C_R(2);
          }
          break;
        case ReactionOrder::TWO_ONE:
          for (int i = 0; i < nzi; ++i)
          {
            kf[i] *= C_R(0) * C_R(0) * C_R(1);
          }
          break;
        case ReactionOrder::ONE_TWO:
          for (int i = 0; i < nzi; ++i)
          {
            kf[i] *= C_R(0) * C_R(1) * C_R(1);
          }
          break;
        default:
          for (int j = 0; j < n_rc; ++j)
          {
            switch (rc_stoich[j])
            {
            case 1:
              for (int i = 0; i < nzi; ++i)
              {
                kf[i] *= C_R(j);
              }
              break;
            case 2:
              for (int i = 0; i < nzi; ++i)
              {
                kf[i] *= C_R(j) * C_R(j);
              }
              break;
            case 3:
              for (int i = 0; i < nzi; ++i)
              {
                kf[i] *= C_R(j) * C_R(j) * C_R(j);
              }
              break;
            }
          }
          break;
        }

        if (rxnData.reversible)
        {
          switch (rxnData.reverseOrder)
          {
          case ReactionOrder::ONE:
            for (int i = 0; i < nzi; ++i)
            {
              kr[i] *= C_P(0);
            }
            break;
          case ReactionOrder::TWO:
            for (int i = 0; i < nzi; ++i)
            {
              kr[i] *= C_P(0) * C_P(0);
            }
            break;
          case ReactionOrder::ONE_ONE:
            for (int i = 0; i < nzi; ++i)
            {
              kr[i] *= C_P(0) * C_P(1);
            }
            break;
          case ReactionOrder::ONE_ONE_ONE:
            for (int i = 0; i < nzi; ++i)
            {
              kr[i] *= C_P(0) * C_P(1) * C_P(2);
            }
            break;
          case ReactionOrder::TWO_ONE:
            for (int i = 0; i < nzi; ++i)
            {
              kr[i] *= C_P(0) * C_P(0) * C_P(1);
            }
            break;
          case ReactionOrder::ONE_TWO:
            for (int i = 0; i < nzi; ++i)
            {
              kr[i] *= C_P(0) * C_P(1) * C_P(1);
            }
            break;
          default:
            for (int j = 0; j < n_pd; ++j)
            {
              switch (pd_stoich[j])
              {
              case -1:
                for (int i = 0; i < nzi; ++i)
                {
                  kr[i] *= C_P(j);
                }
                break;
              case -2:
                for (int i = 0; i < nzi; ++i)
                {
                  kr[i] *= C_P(j) * C_P(j);
                }
                break;
              case -3:
                for (int i = 0; i < nzi; ++i)
                {
                  kr[i] *= C_P(j) * C_P(j) * C_P(j);
                }
                break;
              }
            }
            break;
          }
        }
#undef C_R
#undef C_P
      }

      for (int k = 0; k < n_net; ++k)
      {
        const auto offset = net_indices[k] * nzi;
        const auto numw = net_stoich[k] * net_mw[k];
        for (int i = 0; i < nzi; ++i)
        {
          wi[offset + i] -= numw * (kf[i] - kr[i]);
        }
      }
    }
  }

  for (int k = 0; k < ns - 1; ++k)
  {
    const int offset = k * nzi;
    for (int i = 0; i < nzi; ++i)
    {
      out_rhs[i * ns] -= wi[offset + i] * hi[offset + i] / (rho[i] * cp[i]);
      out_rhs[i * ns + k] = wi[offset + i] / rho[i];
    }
  }
  {
    const int offset = (ns - 1) * nzi;
    for (int i = 0; i < nzi; ++i)
    {
      out_rhs[i * ns] = (out_rhs[i * ns] - wi[offset + i] * hi[offset + i] / (rho[i] * cp[i])) / (rho[i] * cp[i]);
    }
  }

  if (!adiabatic)
  {
    if (use_scaled_heat_loss)
    {
      for (int i = 0; i < nzi; ++i)
      {
        const double &hc = h_convection[i];
        const double &hr = h_radiation[i];
        const double &Tc = T_convection[i];
        const double &Tr = T_radiation[i];
        const double Tr4 = Tr * Tr * Tr * Tr;
        out_rhs[i * ns] += (hc * (Tc - T[i]) / (maxT - Tc) + hr * 5.67e-8 * (Tr4 - T[i] * T[i] * T[i] * T[i]) / (maxT4 - Tr4)) / (rho[i] * cp[i]);
      }
    }
    else
    {
      for (int i = 0; i < nzi; ++i)
      {
        const double &hc = h_convection[i];
        const double &hr = h_radiation[i];
        const double &Tc = T_convection[i];
        const double &Tr = T_radiation[i];
        out_rhs[i * ns] += (hc * (Tc - T[i]) + hr * 5.67e-8 * (Tr * Tr * Tr * Tr - T[i] * T[i] * T[i] * T[i])) / (rho[i] * cp[i]);
      }
    }
  }

  if (include_enthalpy_flux or include_variable_cp)
  {
    for (int i = 1; i < nzi - 1; ++i)
    {
      Tz[i] = mcoeff[i] * T[i - 1] + ncoeff[i] * T[i + 1];
      cpz[i] = mcoeff[i] * cp[i - 1] + ncoeff[i] * cp[i + 1];
    }
    for (int k = 0; k < ns; ++k)
    {
      const int offset = k * nzi;
      for (int i = 1; i < nzi - 1; ++i)
      {
        yz[offset + i] = mcoeff[i] * y[offset + i - 1] + ncoeff[i] * y[offset + i + 1];
      }
    }

    double yoxy[ns];
    double yfuel[ns];
    double Moxy = 0.;
    double Mfuel = 0.;
    double cpoxy = 0.;
    double cpfuel = 0.;
    yoxy[ns - 1] = 1.;
    yfuel[ns - 1] = 1.;
    for (int k = 0; k < ns; ++k)
    {
      yoxy[k] = oxyState[1 + k];
      yfuel[k] = fuelState[1 + k];
      yoxy[ns - 1] -= yoxy[k];
      yfuel[ns - 1] -= yfuel[k];

      Moxy += yoxy[k] * minv[k];
      Mfuel += yfuel[k] * minv[k];
    }

    for (int k = 0; k < ns; ++k)
    {
      const auto polyType = cpTypes[k];
      const auto &c = cpCoeffs[k];
      const double maxT = cpMaxTs[k];
      const double minT = cpMinTs[k];

      const double minvk = minv[k];
      double t;

      switch (polyType)
      {
      case CpType::CONST:
        cpoxy += yoxy[k] * minvk * c[3];
        cpfuel += yfuel[k] * minvk * c[3];
        break;
      case CpType::NASA7:
        t = oxyState[0];
        if (t <= c[0] && t >= minT)
        {
          cpoxy += yoxy[k] * minvk * (c[8] + t * (2. * c[9] + t * (6. * c[10] + t * (12. * c[11] + 20. * t * c[12]))));
          break;
        }
        else if (t > c[0] && t <= maxT)
        {
          cpoxy += yoxy[k] * minvk * (c[1] + t * (2. * c[2] + t * (6. * c[3] + t * (12. * c[4] + 20. * t * c[5]))));
          break;
        }
        else if (t < minT)
        {
          cpoxy += yoxy[k] * minvk * (c[8] + minT * (2. * c[9] + minT * (6. * c[10] + minT * (12. * c[11] + 20. * minT * c[12]))));
          break;
        }
        else
        {
          cpoxy += yoxy[k] * minvk * (c[1] + maxT * (2. * c[2] + maxT * (6. * c[3] + maxT * (12. * c[4] + 20. * maxT * c[5]))));
          break;
        }

        t = fuelState[0];
        if (t <= c[0] && t >= minT)
        {
          cpfuel += yfuel[k] * minvk * (c[8] + t * (2. * c[9] + t * (6. * c[10] + t * (12. * c[11] + 20. * t * c[12]))));
          break;
        }
        else if (t > c[0] && t <= maxT)
        {
          cpfuel += yfuel[k] * minvk * (c[1] + t * (2. * c[2] + t * (6. * c[3] + t * (12. * c[4] + 20. * t * c[5]))));
          break;
        }
        else if (t < minT)
        {
          cpfuel += yfuel[k] * minvk * (c[8] + minT * (2. * c[9] + minT * (6. * c[10] + minT * (12. * c[11] + 20. * minT * c[12]))));
          break;
        }
        else
        {
          cpfuel += yfuel[k] * minvk * (c[1] + maxT * (2. * c[2] + maxT * (6. * c[3] + maxT * (12. * c[4] + 20. * maxT * c[5]))));
          break;
        }
      default:
        break;
      }
    }
    {
      const int i = 0;
      Tz[i] = mcoeff[0] * oxyState[0] + ncoeff[0] * T[1];
      cpz[i] = mcoeff[0] * cpoxy + ncoeff[0] * cp[1];
      for (int k = 0; k < ns; ++k)
      {
        yz[k * ns] = mcoeff[0] * yoxy[k] + ncoeff[0] * y[k * ns + 1];
      }
    }
    {
      const int i = nzi - 1;
      Tz[i] = mcoeff[nzi - 1] * T[nzi - 2] + ncoeff[nzi - 1] * fuelState[0];
      cpz[i] = mcoeff[nzi - 1] * cp[nzi - 2] + ncoeff[nzi - 1] * cpfuel;
      for (int k = 0; k < ns; ++k)
      {
        yz[k * ns + nzi - 1] = mcoeff[nzi - 1] * y[k * ns + nzi - 2] + ncoeff[0] * yfuel[k];
      }
    }

    if (include_variable_cp)
    {
      for (int i = 0; i < nzi; ++i)
      {
        out_rhs[i * ns] += 0.5 * chi[i] * cpz[i] / cp[i] * Tz[i];
      }
    }
    if (include_enthalpy_flux)
    {
      for (int k = 0; k < ns; ++k)
      {
        const int offset = k * nzi;
        for (int i = 0; i < nzi; ++i)
        {
          yzcpi[i] += yz[offset + i] * cpi[offset + i];
        }
      }
      for (int i = 0; i < nzi; ++i)
      {
        out_rhs[i * ns] += 0.5 * chi[i] / cp[i] * Tz[i] * yzcpi[i];
      }
    }
  }

  const int endIdx = (nzi - 1) * ns;
  for (int i = ns; i < endIdx; ++i)
  {
    out_rhs[i] += cmajor[i] * state[i] + csub[i] * state[i - ns] + csup[i] * state[i + ns];
  }
  for (int j = 0; j < ns; ++j)
  {
    out_rhs[j] += cmajor[j] * state[j] + csub[j] * oxyState[j] + csup[j] * state[j + ns];
    out_rhs[endIdx + j] += cmajor[endIdx + j] * state[endIdx + j] + csub[endIdx + j] * state[endIdx + j - ns] + csup[endIdx + j] * fuelState[j];
  }
}

// ------------------------------------------------------------------------------------------------

void CombustionKernels::flamelet_rhs(const double *state, const double &pressure, const double *oxyState,
                                     const double *fuelState, const bool adiabatic, const double *T_convection,
                                     const double *h_convection, const double *T_radiation, const double *h_radiation,
                                     const int &nzi, const double *cmajor, const double *csub, const double *csup,
                                     const double *mcoeff, const double *ncoeff, const double *chi,
                                     const bool include_enthalpy_flux, const bool include_variable_cp,
                                     const bool use_scaled_heat_loss, double *out_rhs) const
{
  const int nSpec = mechanismData.phaseData.nSpecies;

  double maxT = -1;
  double maxT4 = -1;
  if (use_scaled_heat_loss)
  {
    for (int i = 0; i < nzi; ++i)
    {
      maxT = std::max(maxT, state[i * nSpec]);
    }
    maxT4 = maxT * maxT * maxT * maxT;
  }

  double cpz_grid[nzi];

  if (include_variable_cp)
  {
    double cp_grid[nzi];
    for (int i = 0; i < nzi; ++i)
    {
      double y[nSpec];
      extract_y(&state[i * nSpec + 1], nSpec, y);
      cp_grid[i] = cp_mix(state[i * nSpec], y);
    }
    for (int i = 0; i < nzi; ++i)
    {
      if (i == 0)
      {
        double y[nSpec];
        const double *b_state = oxyState;
        extract_y(&b_state[1], nSpec, y);
        cpz_grid[i] = mcoeff[i] * cp_mix(b_state[0], y) + ncoeff[i] * cp_grid[1];
      }
      else if (i == nzi - 1)
      {
        double y[nSpec];
        const double *b_state = fuelState;
        extract_y(&b_state[1], nSpec, y);
        cpz_grid[i] = mcoeff[i] * cp_grid[nzi - 2] + ncoeff[i] * cp_mix(b_state[0], y);
      }
      else
      {
        cpz_grid[i] = mcoeff[i] * cp_grid[i - 1] + ncoeff[i] * cp_grid[i + 1];
      }
    }
  }

  for (int i = 0; i < nzi; ++i)
  {
    double rho;
    double enthalpies[nSpec];
    double w[nSpec];
    double cp;
    double cpi[nSpec];

    const double T = state[i * nSpec];
    double y[nSpec];
    extract_y(&state[i * nSpec + 1], nSpec, y);

    const double mmw = mixture_molecular_weight(y);
    ideal_gas_density(pressure, T, mmw, &rho);
    if (include_enthalpy_flux)
    {
      cp_mix_and_species(T, y, &cp, cpi);
    }
    else
    {
      cp = cp_mix(T, y);
    }

    species_enthalpies(T, enthalpies);
    production_rates(T, rho, mmw, y, w);
    chem_rhs_isobaric(rho, cp, enthalpies, w, &out_rhs[i * nSpec]);

    if (!adiabatic)
    {
      const double hc = h_convection[i];
      const double hr = h_radiation[i];
      const double Tc = T_convection[i];
      const double Tr = T_radiation[i];
      if (use_scaled_heat_loss)
      {
        const double Tr4 = Tr * Tr * Tr * Tr;
        const double q = hc * (Tc - T) / (maxT - Tc) + hr * 5.67e-8 * (Tr4 - T * T * T * T) / (maxT4 - Tr4);
        out_rhs[i * nSpec] += q / (rho * cp);
      }
      else
      {
        const double q = hc * (Tc - T) + hr * 5.67e-8 * (Tr * Tr * Tr * Tr - T * T * T * T);
        out_rhs[i * nSpec] += q / (rho * cp);
      }
    }

    if (include_enthalpy_flux)
    {
      const double cpn = cpi[nSpec - 1];
      double dTdZ;
      double dYdZ_cpi = 0.;
      if (i == 0)
      {
        const double *state_nm1 = oxyState;
        const double *state_np1 = &state[nSpec];
        dTdZ = mcoeff[i] * state_nm1[0] + ncoeff[i] * state_np1[0];
        for (int j = 0; j < nSpec - 1; ++j)
        {
          dYdZ_cpi += (cpi[j] - cpn) * (mcoeff[i] * state_nm1[1 + j] + ncoeff[i] * state_np1[1 + j]);
        }
      }
      else if (i == nzi - 1)
      {
        const double *state_nm1 = &state[(nzi - 2) * nSpec];
        const double *state_np1 = fuelState;
        dTdZ = mcoeff[i] * state_nm1[0] + ncoeff[i] * state_np1[0];
        for (int j = 0; j < nSpec - 1; ++j)
        {
          dYdZ_cpi += (cpi[j] - cpn) * (mcoeff[i] * state_nm1[1 + j] + ncoeff[i] * state_np1[1 + j]);
        }
      }
      else
      {
        const double *state_nm1 = &state[(i - 1) * nSpec];
        const double *state_np1 = &state[(i + 1) * nSpec];
        dTdZ = mcoeff[i] * state_nm1[0] + ncoeff[i] * state_np1[0];
        for (int j = 0; j < nSpec - 1; ++j)
        {
          dYdZ_cpi += (cpi[j] - cpn) * (mcoeff[i] * state_nm1[1 + j] + ncoeff[i] * state_np1[1 + j]);
        }
      }
      out_rhs[i * nSpec] += 0.5 * chi[i] / cp * dTdZ * dYdZ_cpi;

      if (include_variable_cp)
      {
        out_rhs[i * nSpec] += 0.5 * chi[i] * cpz_grid[i] / cp * dTdZ;
      }
    }

    if (include_variable_cp and not include_enthalpy_flux)
    {
      double dTdZ;
      if (i == 0)
      {
        const double *state_nm1 = oxyState;
        const double *state_np1 = &state[nSpec];
        dTdZ = mcoeff[i] * state_nm1[0] + ncoeff[i] * state_np1[0];
      }
      else if (i == nzi - 1)
      {
        const double *state_nm1 = &state[(nzi - 2) * nSpec];
        const double *state_np1 = fuelState;
        dTdZ = mcoeff[i] * state_nm1[0] + ncoeff[i] * state_np1[0];
      }
      else
      {
        const double *state_nm1 = &state[(i - 1) * nSpec];
        const double *state_np1 = &state[(i + 1) * nSpec];
        dTdZ = mcoeff[i] * state_nm1[0] + ncoeff[i] * state_np1[0];
      }
      out_rhs[i * nSpec] += 0.5 * chi[i] * cpz_grid[i] / cp * dTdZ;
    }
  }

  const int endIdx = (nzi - 1) * nSpec;
  for (int i = nSpec; i < endIdx; ++i)
  {
    out_rhs[i] += cmajor[i] * state[i] + csub[i] * state[i - nSpec] + csup[i] * state[i + nSpec];
  }
  for (int j = 0; j < nSpec; ++j)
  {
    out_rhs[j] += cmajor[j] * state[j] + csub[j] * oxyState[j] + csup[j] * state[j + nSpec];
    out_rhs[endIdx + j] += cmajor[endIdx + j] * state[endIdx + j] + csub[endIdx + j] * state[endIdx + j - nSpec] + csup[endIdx + j] * fuelState[j];
  }
}

void CombustionKernels::flamelet_jacobian(const double *state, const double &pressure, const double *oxyState,
                                          const double *fuelState, const bool adiabatic, const double *T_convection,
                                          const double *h_convection, const double *T_radiation, const double *h_radiation,
                                          const int &nzi, const double *cmajor, const double *csub, const double *csup,
                                          const double *mcoeff, const double *ncoeff, const double *chi,
                                          const bool compute_eigenvalues, const double diffterm,
                                          const bool scale_and_offset, const double prefactor,
                                          const int &rates_sensitivity_option, const int &sensitivity_transform_option,
                                          const bool include_enthalpy_flux, const bool include_variable_cp,
                                          const bool use_scaled_heat_loss, double *out_expeig, double *out_jac) const
{
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
  if (use_scaled_heat_loss)
  {
    for (int i = 0; i < nzi; ++i)
    {
      maxT = std::max(maxT, state[i * nSpec]);
    }
    maxT4 = maxT * maxT * maxT * maxT;
  }

  int idx = 0;
  for (int iz = 0; iz < nzi; ++iz)
  {
    double rho;
    double cpi[nSpec];
    double cpisensT[nSpec];
    double enthalpies[nSpec];
    double w[nSpec];
    double wsens[(nSpec + 1) * (nSpec + 1)];

    double primJac[nSpec * (nSpec + 1)];

    const double T = state[iz * nSpec];
    double y[nSpec];
    extract_y(&state[iz * nSpec + 1], nSpec, y);

    const double mmw = mixture_molecular_weight(y);
    ideal_gas_density(pressure, T, mmw, &rho);
    cp_mix_and_species(T, y, &cp[iz], cpi);
    species_enthalpies(T, enthalpies);
    cp_sens_T(T, y, &cpsensT[iz], cpisensT);

    switch (rates_sensitivity_option)
    {
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

    chem_jac_isobaric(pressure, T, y, mmw, rho, cp[iz], cpi, cpsensT[iz], enthalpies, w, wsens, rhsTemp, primJac);

    if (!adiabatic)
    {
      const double Tc = T_convection[iz];
      const double Tr = T_radiation[iz];
      const double hc = h_convection[iz];
      const double hr = h_radiation[iz];

      const double invRhoCp = 1. / (rho * cp[iz]);
      const double invCp = 1. / cp[iz];

      double q;
      if (use_scaled_heat_loss)
      {
        const double Tr4 = Tr * Tr * Tr * Tr;
        q = (hc * (Tc - T) / (maxT - Tc) + hr * 5.67e-8 * (Tr4 - T * T * T * T) / (maxT4 - Tr4)) * invRhoCp;
        primJac[nSpec] -= invCp * cpsensT[iz] * q + invRhoCp * (hc / (maxT - Tc) + 4. * hr / (maxT4 - Tr4) * 5.67e-8 * T * T * T);
      }
      else
      {
        q = (hc * (Tc - T) + hr * 5.67e-8 * (Tr * Tr * Tr * Tr - T * T * T * T)) * invRhoCp;
        primJac[nSpec] -= invCp * cpsensT[iz] * q + invRhoCp * (hc + 4. * hr * 5.67e-8 * T * T * T);
      }

      primJac[0] -= q / rho;

      const double cpn = cpi[nSpec - 1];
      for (int k = 0; k < nSpec + 1; ++k)
      {
        primJac[(2 + k) * nSpec] += invCp * q * (cpn - cpi[k]);
      }
    }

    switch (sensitivity_transform_option)
    {
    case 0:
      transform_isobaric_primitive_jacobian(rho, pressure, T, mmw, primJac, &out_jac[idx]);
      break;
    }

    if (compute_eigenvalues)
    {
      griffon::lapack::eigenvalues(nSpec, &out_jac[idx], realParts, imagParts);
      double exp_eig = 0.;
      for (int iq = 0; iq < nSpec; ++iq)
      {
        exp_eig = std::max(exp_eig, std::max(realParts[iq] - diffterm, 0.));
      }
      for (int iq = 0; iq < nSpec; ++iq)
      {
        out_expeig[iz * nSpec + iq] = exp_eig;
      }
    }

    for (int iq = 0; iq < nSpec; ++iq)
    {
      out_jac[idx + iq * (nSpec + 1)] += cmajor[iz * nSpec + iq];
    }
    idx += blocksize;
  }

  if (include_enthalpy_flux)
  { // note that this is very inexact. this should be improved but convergence seems fine
    for (int i = 1; i < nzi - 1; ++i)
    {
      const double dTdZ = mcoeff[i] * state[(i - 1) * nSpec] + ncoeff[i] * state[(i + 1) * nSpec];
      const double dcpdZ = mcoeff[i] * cp[i - 1] + ncoeff[i] * cp[i + 1];
      const double f1 = 0.5 * chi[i] / cp[i] * dTdZ * dcpdZ;
      out_jac[i * blocksize] -= f1 / cp[i] * cpsensT[i];
    }
    {
      double y[nSpec];
      extract_y(&oxyState[1], nSpec, y);
      const double cp_oxy = cp_mix(oxyState[0], y);

      const int i = 0;
      const double dTdZ = mcoeff[i] * oxyState[0] + ncoeff[i] * state[(i + 1) * nSpec];
      const double dcpdZ = mcoeff[i] * cp_oxy + ncoeff[i] * cp[i + 1];
      const double f1 = 0.5 * chi[i] / cp[i] * dTdZ * dcpdZ;
      out_jac[i * blocksize] -= f1 / cp[i] * cpsensT[i];
    }
    {
      double y[nSpec];
      extract_y(&fuelState[1], nSpec, y);
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
  for (int iz = 1; iz < nzi; ++iz)
  {
    for (int iq = 0; iq < nSpec; ++iq)
    {
      out_jac[idx] += csub[iz * nSpec + iq];
      out_jac[off_diag_offset + idx] += csup[(iz - 1) * nSpec + iq];
      ++idx;
    }
  }
  if (scale_and_offset)
  {
    for (int i = 0; i < nelements; ++i)
    {
      out_jac[i] *= prefactor;
    }
    for (int iz = 0; iz < nzi; ++iz)
    {
      for (int iq = 0; iq < nSpec; ++iq)
      {
        out_jac[iz * blocksize + iq * (nSpec + 1)] -= 1.;
      }
    }
  }
}
} // namespace griffon
