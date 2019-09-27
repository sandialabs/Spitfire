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
#include <cmath>

#define GRIFFON_SUM2( prop ) (           prop(0) + prop(1))
#define GRIFFON_SUM3( prop ) (GRIFFON_SUM2(prop) + prop(2))
#define GRIFFON_SUM4( prop ) (GRIFFON_SUM3(prop) + prop(3))
#define GRIFFON_SUM5( prop ) (GRIFFON_SUM4(prop) + prop(4))
#define GRIFFON_SUM6( prop ) (GRIFFON_SUM5(prop) + prop(5))
#define GRIFFON_SUM7( prop ) (GRIFFON_SUM6(prop) + prop(6))
#define GRIFFON_SUM8( prop ) (GRIFFON_SUM7(prop) + prop(7))
#define ARRHENIUS( coef ) ((coef)[0] * std::exp( (coef)[1] * logT - (coef)[2] * invT ))
#define THD_BDY( i ) (tb_efficiencies[(i)] * y[tb_indices[(i)]])
#define GIBBS( i ) (net_stoich[(i)] * specG[net_indices[(i)]])

namespace griffon {

void CombustionKernels::production_rates(const double &temperature, const double &density, const double *y, double *out_prodrates) const {
  production_rates(temperature, density, mixture_molecular_weight(y), y, out_prodrates);
}

void CombustionKernels::production_rates(const double &temperature, const double &density, const double &mmw, const double *y,
    double *out_prodrates) const {

  const auto T = temperature;
  const auto invT = 1 / T;
  const auto logT = std::log(T);
  const auto rho = density;
  const auto conc = rho / mmw;

  const auto nSpec = mechanismData.phaseData.nSpecies;
  const auto nRxns = mechanismData.reactionData.nReactions;
  const auto standardStatePressure = mechanismData.phaseData.referencePressure;
  const auto invGasConstant = 1. / mechanismData.phaseData.Ru;
  const auto port = standardStatePressure * invT * invGasConstant;

  for (int i = 0; i < nSpec; ++i) {
    out_prodrates[i] = 0.;
  }

  double specG[nSpec];
  for (int n = 0; n < nSpec; ++n) {
    const auto polyType = mechanismData.heatCapacityData.types[n];
    const auto& c = mechanismData.heatCapacityData.coefficients[n];
    switch (polyType) {
    case griffon::HeatCapacityType::NASA7:
      if (T <= c[0]) {
        specG[n] = c[13] + T * (c[8] - c[14] - c[8] * logT - T * (c[9] + T * (c[10] + T * (c[11] + T * c[12]))));
      } else {
        specG[n] = c[6] + T * (c[1] - c[7] - c[1] * logT - T * (c[2] + T * (c[3] + T * (c[4] + T * c[5]))));
      }
      break;
    case griffon::HeatCapacityType::CONST:
      specG[n] = c[1] + c[3] * (T - c[0]) - T * (c[2] + c[3] * (logT - log(c[0])));
      break;
    default: {
      throw std::runtime_error("unsupported thermo");
    }
    }
  }

  double k;
  double kr;
  double m;
  double pr;
  double logPrC;
  double logFCent;

  for (int r = 0; r < nRxns; ++r) {
    const auto& rxnData = mechanismData.reactionData.reactions[r];

    const auto& tb_indices = rxnData.tb_indices;
    const auto& tb_efficiencies = rxnData.tb_efficiencies;
    const auto n_tb = rxnData.n_tb;

    const auto& rc_indices = rxnData.reactant_indices;
    const auto& rc_stoich = rxnData.reactant_stoich;
    const auto& rc_invmw = rxnData.reactant_invmw;
    const auto n_rc = rxnData.n_reactants;

    const auto& pd_indices = rxnData.product_indices;
    const auto& pd_stoich = rxnData.product_stoich;
    const auto& pd_invmw = rxnData.product_invmw;
    const auto n_pd = rxnData.n_products;

    const auto& sp_indices = rxnData.special_indices;
    const auto& sp_order = rxnData.special_orders;
    const auto& sp_invmw = rxnData.special_invmw;
    const auto& sp_nonzero = rxnData.special_nonzero;
    const auto n_sp = rxnData.n_special;

    const auto& net_indices = rxnData.net_indices;
    const auto& net_stoich = rxnData.net_stoich;
    const auto& net_mw = rxnData.net_mw;
    const auto n_net = rxnData.n_net;

    const auto& kCoefs = rxnData.kFwdCoefs;
    const auto& kPCoefs = rxnData.kPressureCoefs;
    const auto baseEff = rxnData.thdBdyDefault;
    const auto& troe = rxnData.troeParams;

    switch (rxnData.kForm) {
    case griffon::RateConstantTemperatureForm::CONSTANT:
      k = kCoefs[0];
      break;
    case griffon::RateConstantTemperatureForm::LINEAR:
      k = kCoefs[0] * T;
      break;
    case griffon::RateConstantTemperatureForm::QUADRATIC:
      k = kCoefs[0] * T * T;
      break;
    case griffon::RateConstantTemperatureForm::RECIPROCAL:
      k = kCoefs[0] * invT;
      break;
    case griffon::RateConstantTemperatureForm::ARRHENIUS:
      k = ARRHENIUS(kCoefs);
      break;
    }

    switch (rxnData.type) {
    case griffon::ReactionRateType::SIMPLE:
      break;
    case griffon::ReactionRateType::THIRD_BODY:
      switch (n_tb) {
      case 0:
        k *= baseEff * conc;
        break;
      case 1:
        k *= (baseEff * conc + rho * (THD_BDY(0)));
        break;
      case 2:
        k *= (baseEff * conc + rho * (GRIFFON_SUM2(THD_BDY)));
        break;
      case 3:
        k *= (baseEff * conc + rho * (GRIFFON_SUM3(THD_BDY)));
        break;
      case 4:
        k *= (baseEff * conc + rho * (GRIFFON_SUM4(THD_BDY)));
        break;
      case 5:
        k *= (baseEff * conc + rho * (GRIFFON_SUM5(THD_BDY)));
        break;
      case 6:
        k *= (baseEff * conc + rho * (GRIFFON_SUM6(THD_BDY)));
        break;
      case 7:
        k *= (baseEff * conc + rho * (GRIFFON_SUM7(THD_BDY)));
        break;
      case 8:
        k *= (baseEff * conc + rho * (GRIFFON_SUM8(THD_BDY)));
        break;
      default:
        m = baseEff * conc + rho * (GRIFFON_SUM8(THD_BDY));
        for (int i = 8; i != n_tb; ++i)
          m += rho * THD_BDY(i);
        k *= m;
        break;
      }
      break;
    case griffon::ReactionRateType::LINDEMANN:
      switch (n_tb) {
      case 0:
        k /= (1 + k / (ARRHENIUS( kPCoefs ) * (baseEff * conc)));
        break;
      case 1:
        k /= (1 + k / (ARRHENIUS( kPCoefs ) * (baseEff * conc + rho * (THD_BDY(0)))));
        break;
      case 2:
        k /= (1 + k / (ARRHENIUS( kPCoefs ) * (baseEff * conc + rho * (GRIFFON_SUM2(THD_BDY)))));
        break;
      case 3:
        k /= (1 + k / (ARRHENIUS( kPCoefs ) * (baseEff * conc + rho * (GRIFFON_SUM3(THD_BDY)))));
        break;
      case 4:
        k /= (1 + k / (ARRHENIUS( kPCoefs ) * (baseEff * conc + rho * (GRIFFON_SUM4(THD_BDY)))));
        break;
      case 5:
        k /= (1 + k / (ARRHENIUS( kPCoefs ) * (baseEff * conc + rho * (GRIFFON_SUM5(THD_BDY)))));
        break;
      case 6:
        k /= (1 + k / (ARRHENIUS( kPCoefs ) * (baseEff * conc + rho * (GRIFFON_SUM6(THD_BDY)))));
        break;
      case 7:
        k /= (1 + k / (ARRHENIUS( kPCoefs ) * (baseEff * conc + rho * (GRIFFON_SUM7(THD_BDY)))));
        break;
      case 8:
        k /= (1 + k / (ARRHENIUS( kPCoefs ) * (baseEff * conc + rho * (GRIFFON_SUM8(THD_BDY)))));
        break;
      default:
        m = baseEff * conc + rho * (GRIFFON_SUM8(THD_BDY));
        for (int i = 8; i != n_tb; ++i)
          m += rho * THD_BDY(i);
        k /= (1 + k / (ARRHENIUS( kPCoefs ) * m));
        break;
      }
      break;
    case griffon::ReactionRateType::TROE:
      switch (n_tb) {
      case 0:
        pr = ARRHENIUS( kPCoefs ) / k * (baseEff * conc);
        break;
      case 1:
        pr = ARRHENIUS( kPCoefs ) / k * (baseEff * conc + rho * (THD_BDY(0)));
        break;
      case 2:
        pr = ARRHENIUS( kPCoefs ) / k * (baseEff * conc + rho * (GRIFFON_SUM2(THD_BDY)));
        break;
      case 3:
        pr = ARRHENIUS( kPCoefs ) / k * (baseEff * conc + rho * (GRIFFON_SUM3(THD_BDY)));
        break;
      case 4:
        pr = ARRHENIUS( kPCoefs ) / k * (baseEff * conc + rho * (GRIFFON_SUM4(THD_BDY)));
        break;
      case 5:
        pr = ARRHENIUS( kPCoefs ) / k * (baseEff * conc + rho * (GRIFFON_SUM5(THD_BDY)));
        break;
      case 6:
        pr = ARRHENIUS( kPCoefs ) / k * (baseEff * conc + rho * (GRIFFON_SUM6(THD_BDY)));
        break;
      case 7:
        pr = ARRHENIUS( kPCoefs ) / k * (baseEff * conc + rho * (GRIFFON_SUM7(THD_BDY)));
        break;
      case 8:
        pr = ARRHENIUS( kPCoefs ) / k * (baseEff * conc + rho * (GRIFFON_SUM8(THD_BDY)));
        break;
      default:
        m = baseEff * conc + rho * (GRIFFON_SUM8(THD_BDY));
        for (int i = 8; i != n_tb; ++i)
          m += rho * THD_BDY(i);
        pr = ARRHENIUS( kPCoefs ) / k * m;
        break;
      }

      switch (rxnData.troeForm) {
      case griffon::TroeTermsPresent::T123:
        logFCent = std::log10((1 - troe[0]) * std::exp(-T / troe[1]) + troe[0] * std::exp(-T / troe[2]) + std::exp(-invT * troe[3]));
        break;
      case griffon::TroeTermsPresent::T12:
        logFCent = std::log10((1 - troe[0]) * std::exp(-T / troe[1]) + troe[0] * std::exp(-T / troe[2]) + 0.0);
        break;
      case griffon::TroeTermsPresent::T1:
        logFCent = std::log10((1 - troe[0]) * std::exp(-T / troe[1]) + 0.0 + 0.0);
        break;
      case griffon::TroeTermsPresent::T23:
        logFCent = std::log10(0.0 + troe[0] * std::exp(-T / troe[2]) + std::exp(-invT * troe[3]));
        break;
      case griffon::TroeTermsPresent::T2:
        logFCent = std::log10(0.0 + troe[0] * std::exp(-T / troe[2]) + 0.0);
        break;
      case griffon::TroeTermsPresent::T13:
        logFCent = std::log10((1 - troe[0]) * std::exp(-T / troe[1]) + 0.0 + std::exp(-invT * troe[3]));
        break;
      case griffon::TroeTermsPresent::T3:
        logFCent = std::log10(0.0 + 0.0 + std::exp(-invT * troe[3]));
        break;
      case griffon::TroeTermsPresent::NO_TROE_TERMS:
      default: {
        throw std::runtime_error("no troe Terms flagged for evaluation");
      }
      }

#define CTROE ( -0.4 - 0.67 * logFCent )
#define NTROE ( 0.75 - 1.27 * logFCent )
#define F1 ( logPrC / ( NTROE - 0.14 * logPrC ) )

      logPrC = std::log10(std::max(pr, 1.e-300)) + CTROE;
      k = k * std::pow(10, logFCent / (1 + F1 * F1)) * pr / (1 + pr);
      break;
    default: {
      throw std::runtime_error("unidentified reaction");
    }
    }

#undef CTROE
#undef NTROE
#undef F1

    if (rxnData.hasOrders) {
#define C_S( i ) ( y[sp_indices[i]] * rho * sp_invmw[i] )
      for (int i = 0; i < n_sp; ++i) {
        if (sp_nonzero[i]) {
          k *= std::pow(std::max(C_S(i), 0.), sp_order[i]);
        }
      }
      kr = 0.;
#undef C_S
    } else {

      kr = 0.;
      if (rxnData.reversible) {
        switch (n_net) {
        case 3:
          kr = k * std::exp(rxnData.sumStoich * std::log(port) - invT * invGasConstant * (GRIFFON_SUM3(GIBBS)));
          break;
        case 2:
          kr = k * std::exp(rxnData.sumStoich * std::log(port) - invT * invGasConstant * (GRIFFON_SUM2(GIBBS)));
          break;
        case 4:
          kr = k * std::exp(rxnData.sumStoich * std::log(port) - invT * invGasConstant * (GRIFFON_SUM4(GIBBS)));
          break;
        case 5:
          kr = k * std::exp(rxnData.sumStoich * std::log(port) - invT * invGasConstant * (GRIFFON_SUM5(GIBBS)));
          break;
        }
      }

#   define C_R( i ) ( y[rc_indices[i]] * rho * rc_invmw[i] )
#   define C_P( i ) ( y[pd_indices[i]] * rho * pd_invmw[i] )

      switch (rxnData.forwardOrder) {
      case griffon::ReactionOrder::ONE:
        k *= C_R(0);
        break;
      case griffon::ReactionOrder::TWO:
        k *= C_R(0) * C_R(0);
        break;
      case griffon::ReactionOrder::ONE_ONE:
        k *= C_R(0) * C_R(1);
        break;
      case griffon::ReactionOrder::ONE_ONE_ONE:
        k *= C_R(0) * C_R(1) * C_R(2);
        break;
      case griffon::ReactionOrder::TWO_ONE:
        k *= C_R(0) * C_R(0) * C_R(1);
        break;
      case griffon::ReactionOrder::ONE_TWO:
        k *= C_R(0) * C_R(1) * C_R(1);
        break;
      default:
        for (int i = 0; i < n_rc; ++i) {
          switch (rc_stoich[i]) {
          case 1:
            k *= C_R(i);
            break;
          case 2:
            k *= C_R(i) * C_R(i);
            break;
          case 3:
            k *= C_R(i) * C_R(i) * C_R(i);
            break;
          }
        }
        break;
      }

      if (rxnData.reversible) {
        switch (rxnData.reverseOrder) {
        case griffon::ReactionOrder::ONE:
          kr *= C_P(0);
          break;
        case griffon::ReactionOrder::TWO:
          kr *= C_P(0) * C_P(0);
          break;
        case griffon::ReactionOrder::ONE_ONE:
          kr *= C_P(0) * C_P(1);
          break;
        case griffon::ReactionOrder::ONE_ONE_ONE:
          kr *= C_P(0) * C_P(1) * C_P(2);
          break;
        case griffon::ReactionOrder::TWO_ONE:
          kr *= C_P(0) * C_P(0) * C_P(1);
          break;
        case griffon::ReactionOrder::ONE_TWO:
          kr *= C_P(0) * C_P(1) * C_P(1);
          break;
        default:
          for (int i = 0; i < n_pd; ++i) {
            switch (pd_stoich[i]) {
            case -1:
              kr *= C_P(i);
              break;
            case -2:
              kr *= C_P(i) * C_P(i);
              break;
            case -3:
              kr *= C_P(i) * C_P(i) * C_P(i);
              break;
            }
          }
          break;
        }
      }
#undef C_R
#undef C_P
    }

    for (int i = 0; i < n_net; ++i) {
      const auto idx = net_indices[i];
      out_prodrates[idx] -= net_stoich[i] * net_mw[i] * (k - kr);
    }
  }
}
void CombustionKernels::prod_rates_primitive_sensitivities(const double &density, const double &temperature, const double *y,
    double *out_prodratessens) const {
  const int nSpec = mechanismData.phaseData.nSpecies;
  double prod_rates[nSpec];
  prod_rates_sens_dense(temperature, density, mixture_molecular_weight(y), y, prod_rates, out_prodratessens);

}

}
