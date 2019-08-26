/*
 *  Copyright (c) 2018-2019 Michael Alan Hansen - All Rights Reserved
 *  You may use, distribute and modify this code under the terms of the MIT license.
 *
 *  You should have received a copy of the MIT license with this file.
 *  If not, please write to mahanse@sandia.gov or mike.hansen@chemeng.utah.edu
 */

// an inexact Jacobian option, magic number = 1
// this one does not consider sensitivities of the three-body enhancement term or falloff terms
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
#define ARRHENIUS_SENS_OVER_K( coef ) ( invT * ( (coef)[1] + (coef)[2] * invT ) ) // \frac{1}{k}\pder{k}{T}
#define THD_BDY( i ) (tb_efficiencies[(i)] * y[tb_indices[(i)]])
#define THD_BDY_NOY( i ) (tb_efficiencies[(i)])
#define GIBBS( i ) (net_stoich[(i)] * specG[net_indices[(i)]])
#define DBDT( i ) (net_stoich[(i)] * dBdTSpec[net_indices[(i)]])

namespace griffon {
void CombustionKernels::prod_rates_sens_no_tbaf(const double &temperature, const double &density, const double &mmw, const double *y,
    double *out_prodrates, double *out_prodratessens) const {
  const double T = temperature;
  const double rho = density;
  const double Mmix = mmw;

  const int ns = mechanismData.phaseData.nSpecies;
  const int nr = mechanismData.reactionData.nReactions;
  const double standardStatePressure = mechanismData.phaseData.referencePressure;
  const double invGasConstant = 1. / mechanismData.phaseData.Ru;
  const auto Msp = mechanismData.phaseData.molecularWeights.data();
  const auto invMsp = mechanismData.phaseData.inverseMolecularWeights.data();

  double kf, Kc, kr, Rr, Rnet, q, Ctbaf, pr, fCent, flfConc, fTroe, gTroe;
  double dRnetdrho, dRnetdT, dKcdToverKc;
  double dqdrho, dqdT, aTroe, bTroe;
  double nsTmp;

  double specG[ns];
  double dBdTSpec[ns];
  double dRnetdY[ns - 1];
  double dqdY[ns - 1];

  for (int i = 0; i < ns; ++i) {
    out_prodrates[i] = 0.;
    specG[i] = 0.;
  }
  for (int i = 0; i < (ns + 1) * (ns + 1); ++i) {
    out_prodratessens[i] = 0.;
  }

  const double invT = 1. / T;
  const double logT = std::log(T);

  const double invM = 1. / Mmix;
  const double ct = rho * invM;

  const double invRu = invGasConstant;
  const double Ru = 1. / invRu;

  for (int n = 0; n < ns; ++n) {
    const auto polyType = mechanismData.heatCapacityData.types[n];
    const auto& c = mechanismData.heatCapacityData.coefficients[n];
    switch (polyType) {
    case griffon::HeatCapacityType::NASA7:
      if (T <= c[0]) {
        specG[n] = c[13] + T * (c[8] - c[14] - c[8] * logT - T * (c[9] + T * (c[10] + T * (c[11] + T * c[12]))));
        dBdTSpec[n] = invRu * ((c[8] - Ru) * invT + c[9] + T * (2 * c[10] + T * (3 * c[11] + T * 4 * c[12])) + c[13] * invT * invT);
      } else {
        specG[n] = c[6] + T * (c[1] - c[7] - c[1] * logT - T * (c[2] + T * (c[3] + T * (c[4] + T * c[5]))));
        dBdTSpec[n] = invRu * ((c[1] - Ru) * invT + c[2] + T * (2 * c[3] + T * (3 * c[4] + T * 4 * c[5])) + c[6] * invT * invT);
      }
      break;
    case griffon::HeatCapacityType::CONST:
      specG[n] = c[1] + c[3] * (T - c[0]) - T * (c[2] + c[3] * (logT - std::log(c[0])));
      dBdTSpec[n] = invT * (Msp[n] * invRu * (c[3] - invT * (c[3] * c[0] - c[1])) - 1);
      break;
    default: {
      throw std::runtime_error("unsupported thermo model");
    }
    }
  }

  for (int r = 0; r < nr; ++r) {
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

    const auto& net_indices = rxnData.net_indices;
    const auto& net_stoich = rxnData.net_stoich;
    const auto& net_mw = rxnData.net_mw;
    const auto n_net = rxnData.n_net;

    const auto& kCoefs = rxnData.kFwdCoefs;
    const auto& kPCoefs = rxnData.kPressureCoefs;
    const auto baseEff = rxnData.thdBdyDefault;
    const auto& troe = rxnData.troeParams;

    Rnet = 0.;
    dRnetdrho = 0.;
    dRnetdT = 0.;

    dqdrho = 0.;
    dqdT = 0.;

    for (int i = 0; i < ns - 1; ++i) {
      dqdY[i] = 0.0;
      dRnetdY[i] = 0.0;
    }

    switch (rxnData.kForm) {
    case griffon::RateConstantTemperatureForm::CONSTANT:
      kf = kCoefs[0];
      break;
    case griffon::RateConstantTemperatureForm::LINEAR:
      kf = kCoefs[0] * T;
      break;
    case griffon::RateConstantTemperatureForm::QUADRATIC:
      kf = kCoefs[0] * T * T;
      break;
    case griffon::RateConstantTemperatureForm::RECIPROCAL:
      kf = kCoefs[0] * invT;
      break;
    case griffon::RateConstantTemperatureForm::ARRHENIUS:
      kf = ARRHENIUS(kCoefs);
      break;
    }

#   define C_R( i ) ( y[rc_indices[i]] * rho * rc_invmw[i] )
#   define C_P( i ) ( y[pd_indices[i]] * rho * pd_invmw[i] )
    switch (rxnData.forwardOrder) {
    case griffon::ReactionOrder::ONE:
      Rnet = kf * C_R(0);
      break;
    case griffon::ReactionOrder::TWO:
      Rnet = kf * C_R(0) * C_R(0);
      break;
    case griffon::ReactionOrder::ONE_ONE:
      Rnet = kf * C_R(0) * C_R(1);
      break;
    case griffon::ReactionOrder::ONE_ONE_ONE:
      Rnet = kf * C_R(0) * C_R(1) * C_R(2);
      break;
    case griffon::ReactionOrder::TWO_ONE:
      Rnet = kf * C_R(0) * C_R(0) * C_R(1);
      break;
    case griffon::ReactionOrder::ONE_TWO:
      Rnet = kf * C_R(0) * C_R(1) * C_R(1);
      break;
    default:
      Rnet = kf;
      for (int i = 0; i != n_rc; ++i) {
        switch (rc_stoich[i]) {
        case 1:
          Rnet *= C_R(i);
          break;
        case 2:
          Rnet *= C_R( i ) * C_R(i);
          break;
        case 3:
          Rnet *= C_R( i ) * C_R(i) * C_R(i);
          break;
        }
      }
      break;
    }

    dRnetdrho = Rnet / ct * invM * rxnData.sumReactantStoich;
    dRnetdT = Rnet * ARRHENIUS_SENS_OVER_K(kCoefs);

    bool nsIsReactant = false;
    int nsReactantIdx = -1;
    for (int sridx = 0; sridx < n_rc; ++sridx) {
      int s = rc_indices[sridx];

      if (s != ns - 1) {
        switch (rxnData.forwardOrder) {
        case griffon::ReactionOrder::ONE:
          dRnetdY[s] = kf * rho * invMsp[s];
          break;
        case griffon::ReactionOrder::TWO:
          dRnetdY[s] = kf * rho * invMsp[s] * 2. * C_R(sridx);
          break;
        case griffon::ReactionOrder::ONE_ONE:
          switch (sridx) {
          case 0:
            dRnetdY[s] = kf * rho * invMsp[s] * C_R(1);
            break;
          case 1:
            dRnetdY[s] = kf * rho * invMsp[s] * C_R(0);
            break;
          }
          break;
        case griffon::ReactionOrder::ONE_ONE_ONE:
          switch (sridx) {
          case 0:
            dRnetdY[s] = kf * rho * invMsp[s] * C_R(1) * C_R(2);
            break;
          case 1:
            dRnetdY[s] = kf * rho * invMsp[s] * C_R(0) * C_R(2);
            break;
          case 2:
            dRnetdY[s] = kf * rho * invMsp[s] * C_R(0) * C_R(1);
            break;
          }
          break;
        case griffon::ReactionOrder::TWO_ONE:
          switch (sridx) {
          case 0:
            dRnetdY[s] = kf * rho * invMsp[s] * 2. * C_R(0) * C_R(1);
            break;
          case 1:
            dRnetdY[s] = kf * rho * invMsp[s] * C_R(0) * C_R(0);
            break;
          }
          break;
        case griffon::ReactionOrder::ONE_TWO:
          switch (sridx) {
          case 0:
            dRnetdY[s] = kf * rho * invMsp[s] * C_R(1) * C_R(1);
            break;
          case 1:
            dRnetdY[s] = kf * rho * invMsp[s] * 2. * C_R(1) * C_R(0);
            break;
          }
          break;
        default:
          switch (rc_stoich[sridx]) {
          case 1:
            dRnetdY[s] = kf * rho * invMsp[s];
            break;
          case 2:
            dRnetdY[s] = kf * rho * invMsp[s] * 2. * C_R(sridx);
            break;
          case 3:
            dRnetdY[s] = kf * rho * invMsp[s] * 3. * C_R(sridx) * C_R(sridx);
            break;
          }
          for (int i = 0; i < n_rc; ++i) {
            if (!(i == sridx)) {
              switch (rc_stoich[i]) {
              case 1:
                dRnetdY[s] *= C_R(i);
                break;
              case 2:
                dRnetdY[s] *= C_R( i ) * C_R(i);
                break;
              case 3:
                dRnetdY[s] *= C_R( i ) * C_R(i) * C_R(i);
                break;
              }
            }
          }
        }
      } else {
        nsIsReactant = true;
        nsReactantIdx = sridx;
      }
    }

    if (nsIsReactant) {
      switch (rxnData.forwardOrder) {
      case griffon::ReactionOrder::ONE:
        nsTmp = kf * rho * invMsp[ns - 1];
        break;
      case griffon::ReactionOrder::TWO:
        nsTmp = kf * rho * invMsp[ns - 1] * 2. * C_R(nsReactantIdx);
        break;
      case griffon::ReactionOrder::ONE_ONE:
        switch (nsReactantIdx) {
        case 0:
          nsTmp = kf * rho * invMsp[ns - 1] * C_R(1);
          break;
        case 1:
          nsTmp = kf * rho * invMsp[ns - 1] * C_R(0);
          break;
        }
        break;
      case griffon::ReactionOrder::ONE_ONE_ONE:
        switch (nsReactantIdx) {
        case 0:
          nsTmp = kf * rho * invMsp[ns - 1] * C_R(1) * C_R(2);
          break;
        case 1:
          nsTmp = kf * rho * invMsp[ns - 1] * C_R(0) * C_R(2);
          break;
        case 2:
          nsTmp = kf * rho * invMsp[ns - 1] * C_R(0) * C_R(1);
          break;
        }
        break;
      case griffon::ReactionOrder::TWO_ONE:
        switch (nsReactantIdx) {
        case 0:
          nsTmp = kf * rho * invMsp[ns - 1] * 2. * C_R(0) * C_R(1);
          break;
        case 1:
          nsTmp = kf * rho * invMsp[ns - 1] * C_R(0) * C_R(0);
          break;
        }
        break;
      case griffon::ReactionOrder::ONE_TWO:
        switch (nsReactantIdx) {
        case 0:
          nsTmp = kf * rho * invMsp[ns - 1] * C_R(1) * C_R(1);
          break;
        case 1:
          nsTmp = kf * rho * invMsp[ns - 1] * 2. * C_R(1) * C_R(0);
          break;
        }
        break;
      default:
        nsTmp = kf * rho * invMsp[ns - 1];
        switch (rc_stoich[nsReactantIdx]) {
        case 1:
          break;
        case 2:
          nsTmp *= 2. * C_R(nsReactantIdx);
          break;
        case 3:
          nsTmp *= 3. * C_R(nsReactantIdx) * C_R(nsReactantIdx);
          break;
        }
        for (int i = 0; i < n_rc; ++i) {
          if (!(i == nsReactantIdx)) {
            switch (rc_stoich[i]) {
            case 1:
              nsTmp *= C_R(i);
              break;
            case 2:
              nsTmp *= C_R( i ) * C_R(i);
              break;
            case 3:
              nsTmp *= C_R( i ) * C_R(i) * C_R(i);
              break;
            }
          }
        }
      }
      for (int s = 0; s < ns - 1; ++s) {
        dRnetdY[s] -= nsTmp;
      }
    }

    if (rxnData.reversible) {
      const int sumStoich = rxnData.sumStoich;
      switch (n_net) {
      case 3:
        Kc = std::exp(-(sumStoich * std::log(standardStatePressure * invT * invRu) - invT * invRu * (GRIFFON_SUM3(GIBBS))));
        dKcdToverKc = -GRIFFON_SUM3(DBDT);
        break;
      case 2:
        Kc = std::exp(-(sumStoich * std::log(standardStatePressure * invT * invRu) - invT * invRu * (GRIFFON_SUM2(GIBBS))));
        dKcdToverKc = -GRIFFON_SUM2(DBDT);
        break;
      case 4:
        Kc = std::exp(-(sumStoich * std::log(standardStatePressure * invT * invRu) - invT * invRu * (GRIFFON_SUM4(GIBBS))));
        dKcdToverKc = -GRIFFON_SUM4(DBDT);
        break;
      case 5:
        Kc = std::exp(-(sumStoich * std::log(standardStatePressure * invT * invRu) - invT * invRu * (GRIFFON_SUM5(GIBBS))));
        dKcdToverKc = -GRIFFON_SUM5(DBDT);
        break;
      }
      kr = kf / Kc;

      switch (rxnData.reverseOrder) {
      case griffon::ReactionOrder::ONE:
        Rr = kr * C_P(0);
        break;
      case griffon::ReactionOrder::TWO:
        Rr = kr * C_P(0) * C_P(0);
        break;
      case griffon::ReactionOrder::ONE_ONE:
        Rr = kr * C_P(0) * C_P(1);
        break;
      case griffon::ReactionOrder::ONE_ONE_ONE:
        Rr = kr * C_P(0) * C_P(1) * C_P(2);
        break;
      case griffon::ReactionOrder::TWO_ONE:
        Rr = kr * C_P(0) * C_P(0) * C_P(1);
        break;
      case griffon::ReactionOrder::ONE_TWO:
        Rr = kr * C_P(0) * C_P(1) * C_P(1);
        break;
      default:
        Rr = kr;
        for (int i = 0; i < n_pd; ++i) {
          switch (abs(pd_stoich[i])) {
          case 1:
            Rr *= C_P(i);
            break;
          case 2:
            Rr *= C_P( i ) * C_P(i);
            break;
          case 3:
            Rr *= C_P( i ) * C_P(i) * C_P(i);
            break;
          }
        }
        break;
      }

      Rnet -= Rr;
      dRnetdrho -= Rr / ct * invM * rxnData.sumProductStoich;
      dRnetdT -= Rr * (ARRHENIUS_SENS_OVER_K( kCoefs ) - dKcdToverKc);

      bool nsIsProduct = false;
      int nsProductIdx = -1;

      for (int sridx = 0; sridx < n_pd; ++sridx) {
        int s = pd_indices[sridx];

        if (s != ns - 1) {
          switch (rxnData.reverseOrder) {
          case griffon::ReactionOrder::ONE:
            dRnetdY[s] -= kr * rho * invMsp[s];
            break;
          case griffon::ReactionOrder::TWO:
            dRnetdY[s] -= kr * rho * invMsp[s] * 2. * C_P(sridx);
            break;
          case griffon::ReactionOrder::ONE_ONE:
            switch (sridx) {
            case 0:
              dRnetdY[s] -= kr * rho * invMsp[s] * C_P(1);
              break;
            case 1:
              dRnetdY[s] -= kr * rho * invMsp[s] * C_P(0);
              break;
            }
            break;
          case griffon::ReactionOrder::ONE_ONE_ONE:
            switch (sridx) {
            case 0:
              dRnetdY[s] -= kr * rho * invMsp[s] * C_P(1) * C_P(2);
              break;
            case 1:
              dRnetdY[s] -= kr * rho * invMsp[s] * C_P(0) * C_P(2);
              break;
            case 2:
              dRnetdY[s] -= kr * rho * invMsp[s] * C_P(0) * C_P(1);
              break;
            }
            break;
          case griffon::ReactionOrder::TWO_ONE:
            switch (sridx) {
            case 0:
              dRnetdY[s] -= kr * rho * invMsp[s] * 2. * C_P(0) * C_P(1);
              break;
            case 1:
              dRnetdY[s] -= kr * rho * invMsp[s] * C_P(0) * C_P(0);
              break;
            }
            break;
          case griffon::ReactionOrder::ONE_TWO:
            switch (sridx) {
            case 0:
              dRnetdY[s] -= kr * rho * invMsp[s] * C_P(1) * C_P(1);
              break;
            case 1:
              dRnetdY[s] -= kr * rho * invMsp[s] * 2. * C_P(1) * C_P(0);
              break;
            }
            break;
          default:
            // first: build the part sensitive to species s
            switch (std::abs(pd_stoich[sridx])) {
            case 1:
              nsTmp = kr * rho * invMsp[s];
              break;
            case 2:
              nsTmp = kr * rho * invMsp[s] * 2. * C_P(sridx);
              break;
            case 3:
              nsTmp = kr * rho * invMsp[s] * 3. * C_P(sridx) * C_P(sridx);
              break;
            default:
              nsTmp = kr * rho * invMsp[s] * std::abs(pd_stoich[sridx]) * std::pow(C_P(sridx), std::abs(pd_stoich[sridx]) - 1);
              break;
            }
            // second: build the rest of the rate law product
            for (int i = 0; i < n_pd; ++i) {
              if (!(i == sridx)) {
                switch (std::abs(pd_stoich[i])) {
                case 1:
                  nsTmp *= C_P(i);
                  break;
                case 2:
                  nsTmp *= C_P( i ) * C_P(i);
                  break;
                case 3:
                  nsTmp *= C_P( i ) * C_P(i) * C_P(i);
                  break;
                default:
                  nsTmp *= std::pow(C_P(i), std::abs(pd_stoich[i]));
                  break;
                }
              }
            }
            // finally: offset the sensitivity
            dRnetdY[s] -= nsTmp;
          }
        } else {
          nsIsProduct = true;
          nsProductIdx = sridx;
        }
      }
      if (nsIsProduct) {
        switch (rxnData.reverseOrder) {
        case griffon::ReactionOrder::ONE:
          nsTmp = kr * rho * invMsp[ns - 1];
          break;
        case griffon::ReactionOrder::TWO:
          nsTmp = kr * rho * invMsp[ns - 1] * 2. * C_P(nsProductIdx);
          break;
        case griffon::ReactionOrder::ONE_ONE:
          switch (nsProductIdx) {
          case 0:
            nsTmp = kr * rho * invMsp[ns - 1] * C_P(1);
            break;
          case 1:
            nsTmp = kr * rho * invMsp[ns - 1] * C_P(0);
            break;
          }
          break;
        case griffon::ReactionOrder::ONE_ONE_ONE:
          switch (nsProductIdx) {
          case 0:
            nsTmp = kr * rho * invMsp[ns - 1] * C_P(1) * C_P(2);
            break;
          case 1:
            nsTmp = kr * rho * invMsp[ns - 1] * C_P(0) * C_P(2);
            break;
          case 2:
            nsTmp = kr * rho * invMsp[ns - 1] * C_P(0) * C_P(1);
            break;
          }
          break;
        case griffon::ReactionOrder::TWO_ONE:
          switch (nsProductIdx) {
          case 0:
            nsTmp = kr * rho * invMsp[ns - 1] * 2. * C_P(0) * C_P(1);
            break;
          case 1:
            nsTmp = kr * rho * invMsp[ns - 1] * C_P(0) * C_P(0);
            break;
          }
          break;
        case griffon::ReactionOrder::ONE_TWO:
          switch (nsProductIdx) {
          case 0:
            nsTmp = kr * rho * invMsp[ns - 1] * C_P(1) * C_P(1);
            break;
          case 1:
            nsTmp = kr * rho * invMsp[ns - 1] * 2. * C_P(1) * C_P(0);
            break;
          }
          break;
        default:
          switch (std::abs(pd_stoich[nsProductIdx])) {
          case 1:
            nsTmp = kr * rho * invMsp[ns - 1];
            break;
          case 2:
            nsTmp = kr * rho * invMsp[ns - 1] * 2. * C_P(nsProductIdx);
            break;
          case 3:
            nsTmp *= kr * rho * invMsp[ns - 1] * 3. * C_P(nsProductIdx) * C_P(nsProductIdx);
            break;
          }
          for (int i = 0; i != n_pd; ++i) {
            if (!(i == nsProductIdx)) {
              switch (std::abs(pd_stoich[i])) {
              case 1:
                nsTmp *= C_P(i);
                break;
              case 2:
                nsTmp *= C_P( i ) * C_P(i);
                break;
              case 3:
                nsTmp *= C_P( i ) * C_P(i) * C_P(i);
                break;
              }
            }
          }
        }
        for (int s = 0; s < ns - 1; ++s) {
          dRnetdY[s] += nsTmp;
        }
      }
    }

    double t1exp;
    double t2exp;
    double t3exp;
    double log10pr;
    double log10fcent;
    double logfcent;
    double ln10;
    double rho_baseeff;
    double kp_over_kf;
    switch (rxnData.type) {
    case griffon::ReactionRateType::SIMPLE:
      Ctbaf = 1.0;
      break;
    case griffon::ReactionRateType::THIRD_BODY:

      Ctbaf = baseEff * ct;
      for (int i = 0; i < n_tb; ++i)
        Ctbaf += rho * THD_BDY(i);
      break;
    case griffon::ReactionRateType::LINDEMANN:
      flfConc = baseEff * ct;
      for (int i = 0; i < n_tb; ++i)
        flfConc = flfConc + rho * THD_BDY(i);

      kp_over_kf = ARRHENIUS( kPCoefs ) / kf;
      pr = kp_over_kf * flfConc;

      Ctbaf = pr / (1. + pr);
      break;
    case griffon::ReactionRateType::TROE:
      t1exp = std::exp(-T / troe[1]);
      t2exp = std::exp(-T / troe[2]);
      t3exp = std::exp(-invT * troe[3]);
      switch (rxnData.troeForm) {
      case griffon::TroeTermsPresent::T123:
        fCent = (1 - troe[0]) * t1exp + troe[0] * t2exp + t3exp;
        break;
      case griffon::TroeTermsPresent::T12:
        fCent = (1 - troe[0]) * t1exp + troe[0] * t2exp;
        break;
      case griffon::TroeTermsPresent::T1:
        fCent = (1 - troe[0]) * t1exp;
        break;
      case griffon::TroeTermsPresent::T23:
        fCent = troe[0] * t2exp + t3exp;
        break;
      case griffon::TroeTermsPresent::T2:
        fCent = troe[0] * t2exp;
        break;
      case griffon::TroeTermsPresent::T13:
        fCent = (1 - troe[0]) * t1exp + t3exp;
        break;
      case griffon::TroeTermsPresent::T3:
        fCent = t3exp;
        break;
      case griffon::TroeTermsPresent::NO_TROE_TERMS:
      default: {
        throw std::runtime_error("no troe terms flagged for evaluation");
      }
      }

      flfConc = baseEff * ct;
      for (int i = 0; i < n_tb; ++i)
        flfConc = flfConc + rho * THD_BDY(i);

      kp_over_kf = ARRHENIUS( kPCoefs ) / kf;
      pr = kp_over_kf * flfConc;

      log10pr = std::log10(std::max(pr, 1.e-300));
      log10fcent = std::log10(std::max(fCent, 1.e-300));
      logfcent = std::log(std::max(fCent, 1.e-300));
      ln10 = std::log(10.);

      aTroe = log10pr - 0.67 * log10fcent - 0.4;
      bTroe = -0.14 * log10pr - 1.1762 * log10fcent + 0.806;
      gTroe = 1 / (1 + (aTroe / bTroe) * (aTroe / bTroe));

      fTroe = std::pow(fCent, gTroe);
      Ctbaf = fTroe * pr / (1 + pr);
      break;
    default: {
      throw std::runtime_error("unidentified reaction");
    }
    }

    q = Rnet * Ctbaf;

    dqdrho = dRnetdrho * Ctbaf;
    dqdT = dRnetdT * Ctbaf;
    for (int s = 0; s < ns - 1; ++s) {
      dqdY[s] = dRnetdY[s] * Ctbaf;
    }

    for (int i = 0; i < n_net; ++i) {
      const int index = net_indices[i];
      const double factor = net_stoich[i] * net_mw[i];
      out_prodrates[index] -= factor * q;
      out_prodratessens[index] -= factor * dqdrho;
      out_prodratessens[index + (ns + 1)] -= factor * dqdT;
      for (int s = 0; s < ns - 1; ++s) {
        out_prodratessens[index + (ns + 1) * (2 + s)] -= factor * dqdY[s];
      }
    }
  }
}
}

