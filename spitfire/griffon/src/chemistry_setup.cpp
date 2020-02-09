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
#include <algorithm>
#include <stdexcept>
#include <cmath>
#include <tuple>

namespace griffon {

void CombustionKernels::mechanism_add_element(const std::string& element_name) {
  const auto& en = mechanismData.phaseData.elementNames;
  if (std::find(en.begin(), en.end(), element_name) == en.end()) {
    ++mechanismData.phaseData.nElements;
    mechanismData.phaseData.elementNames.push_back(element_name);
  }
}

void CombustionKernels::mechanism_add_species(const std::string& species_name, const std::map<std::string, double> atom_map) {
  const auto& sn = mechanismData.phaseData.speciesNames;
  if (std::find(sn.begin(), sn.end(), species_name) != sn.end()) {
    throw std::logic_error("species " + species_name + " cannot be added twice");
  }
  const auto& en = mechanismData.phaseData.elementNames;
  for (const auto& atom_num : atom_map) {
    if (std::find(en.begin(), en.end(), atom_num.first) == en.end()) {
      throw std::logic_error("cannot find atom " + atom_num.first);
    }
  }
  const auto element_mw_map = get_element_mw_map();
  double mw = 0.;
  for (const auto& atom_num : atom_map) {
    mw += element_mw_map.at(atom_num.first) * atom_num.second;
  }
  mechanismData.phaseData.speciesIndices.insert(std::make_pair(species_name, mechanismData.phaseData.nSpecies));
  mechanismData.phaseData.speciesNames.push_back(species_name);
  mechanismData.phaseData.molecularWeights.push_back(mw);
  mechanismData.phaseData.inverseMolecularWeights.push_back(1. / mw);
  ++mechanismData.phaseData.nSpecies;
}

void CombustionKernels::mechanism_set_ref_pressure(const double& p_ref) {
  mechanismData.phaseData.referencePressure = p_ref;
}

void CombustionKernels::mechanism_set_ref_temperature(const double& T_ref) {
  mechanismData.phaseData.referenceTemperature = T_ref;
}

void CombustionKernels::mechanism_resize_heat_capacity_data() {
  const auto ns = mechanismData.phaseData.nSpecies;
  mechanismData.heatCapacityData.coefficients.resize(ns);
  mechanismData.heatCapacityData.minTemperatures.resize(ns);
  mechanismData.heatCapacityData.maxTemperatures.resize(ns);
  mechanismData.heatCapacityData.types.resize(ns);
}

void CombustionKernels::mechanism_add_const_cp(const std::string& spec_name, const double& Tmin, const double& Tmax, const double& T0,
    const double& h0, const double& s0, const double& cp) {
  const auto i = mechanismData.phaseData.speciesIndices.at(spec_name);
  mechanismData.heatCapacityData.types[i] = griffon::HeatCapacityType::CONST;
  mechanismData.heatCapacityData.minTemperatures[i] = Tmin;
  mechanismData.heatCapacityData.maxTemperatures[i] = Tmax;
  mechanismData.heatCapacityData.coefficients[i][0] = T0;
  mechanismData.heatCapacityData.coefficients[i][1] = h0;
  mechanismData.heatCapacityData.coefficients[i][2] = s0;
  mechanismData.heatCapacityData.coefficients[i][3] = cp;
}

void CombustionKernels::mechanism_add_nasa7_cp(const std::string& spec_name, const double& Tmin, const double& Tmid, const double& Tmax,
    const std::vector<double>& low_coeffs, const std::vector<double>& high_coeffs) {
  const auto i = mechanismData.phaseData.speciesIndices.at(spec_name);
  mechanismData.heatCapacityData.types[i] = griffon::HeatCapacityType::NASA7;
  mechanismData.heatCapacityData.minTemperatures[i] = Tmin;
  mechanismData.heatCapacityData.maxTemperatures[i] = Tmax;

  auto& coeffs = mechanismData.heatCapacityData.coefficients[i];
  coeffs[0] = Tmid;
  const double R = mechanismData.phaseData.Ru;
  for (int i = 0; i < high_coeffs.size(); ++i) {
    coeffs[1 + i] = high_coeffs[i] * R;
  }
  for (int i = 0; i < low_coeffs.size(); ++i) {
    coeffs[8 + i] = low_coeffs[i] * R;
  }
  coeffs[2] /= 2.;
  coeffs[3] /= 6.;
  coeffs[4] /= 12.;
  coeffs[5] /= 20.;
  coeffs[9] /= 2.;
  coeffs[10] /= 6.;
  coeffs[11] /= 12.;
  coeffs[12] /= 20.;
}

void CombustionKernels::mechanism_add_reaction_simple(const std::map<std::string, int>& reactants_stoich,
    const std::map<std::string, int>& products_stoich, const bool reversible, const double& fwd_pre_exp_value,
    const double& fwd_temp_exponent, const double& fwd_act_energy) {
  typename griffon::ReactionData<6,52>::ReactionRateData rxnratedata;
  rxnratedata.type = griffon::ReactionRateType::SIMPLE;
  rxnratedata.reversible = reversible;
  rxnratedata.hasOrders = false;
  rxnratedata.set_fwd_pre_exponential(fwd_pre_exp_value);
  rxnratedata.set_fwd_temp_exponent(fwd_temp_exponent);
  rxnratedata.set_fwd_activation_energy(fwd_act_energy);
  rxnratedata.set_reactants(reactants_stoich, mechanismData.phaseData);
  rxnratedata.set_products(products_stoich, mechanismData.phaseData);
  rxnratedata.finalize(mechanismData.phaseData);
  mechanismData.reactionData.reactions.push_back(rxnratedata);
  ++mechanismData.reactionData.nReactions;
}

void CombustionKernels::mechanism_add_reaction_three_body(const std::map<std::string, int>& reactants_stoich,
    const std::map<std::string, int>& products_stoich, const bool reversible, const double& fwd_pre_exp_value,
    const double& fwd_temp_exponent, const double& fwd_act_energy, const std::map<std::string, double>& three_body_efficiencies,
    const double& default_efficiency) {
  typename griffon::ReactionData<6,52>::ReactionRateData rxnratedata;
  rxnratedata.type = griffon::ReactionRateType::THIRD_BODY;
  rxnratedata.reversible = reversible;
  rxnratedata.hasOrders = false;
  rxnratedata.set_fwd_pre_exponential(fwd_pre_exp_value);
  rxnratedata.set_fwd_temp_exponent(fwd_temp_exponent);
  rxnratedata.set_fwd_activation_energy(fwd_act_energy);
  rxnratedata.set_reactants(reactants_stoich, mechanismData.phaseData);
  rxnratedata.set_products(products_stoich, mechanismData.phaseData);
  rxnratedata.set_three_body_efficiencies(three_body_efficiencies, default_efficiency, mechanismData.phaseData);
  rxnratedata.finalize(mechanismData.phaseData);
  mechanismData.reactionData.reactions.push_back(rxnratedata);
  ++mechanismData.reactionData.nReactions;
}

void CombustionKernels::mechanism_add_reaction_Lindemann(const std::map<std::string, int>& reactants_stoich,
    const std::map<std::string, int>& products_stoich, const bool reversible, const double fwd_pre_exp_value,
    const double fwd_temp_exponent, const double fwd_act_energy, const std::map<std::string, double>& three_body_efficiencies,
    const double& default_efficiency, const double flf_pre_exp_value, const double flf_temp_exponent, const double flf_act_energy) {
  typename griffon::ReactionData<6,52>::ReactionRateData rxnratedata;
  rxnratedata.type = griffon::ReactionRateType::LINDEMANN;
  rxnratedata.reversible = reversible;
  rxnratedata.hasOrders = false;
  rxnratedata.set_fwd_pre_exponential(fwd_pre_exp_value);
  rxnratedata.set_fwd_temp_exponent(fwd_temp_exponent);
  rxnratedata.set_fwd_activation_energy(fwd_act_energy);
  rxnratedata.set_reactants(reactants_stoich, mechanismData.phaseData);
  rxnratedata.set_products(products_stoich, mechanismData.phaseData);
  rxnratedata.set_three_body_efficiencies(three_body_efficiencies, default_efficiency, mechanismData.phaseData);
  rxnratedata.set_falloff_pre_exponential(flf_pre_exp_value);
  rxnratedata.set_falloff_temp_exponent(flf_temp_exponent);
  rxnratedata.set_falloff_activation_energy(flf_act_energy);
  rxnratedata.finalize(mechanismData.phaseData);
  mechanismData.reactionData.reactions.push_back(rxnratedata);
  ++mechanismData.reactionData.nReactions;
}

void CombustionKernels::mechanism_add_reaction_Troe(const std::map<std::string, int>& reactants_stoich,
    const std::map<std::string, int>& products_stoich, const bool reversible, const double fwd_pre_exp_value,
    const double fwd_temp_exponent, const double fwd_act_energy, const std::map<std::string, double>& three_body_efficiencies,
    const double& default_efficiency, const double flf_pre_exp_value, const double flf_temp_exponent, const double flf_act_energy,
    const std::vector<double>& troe_parameters) {
  typename griffon::ReactionData<6,52>::ReactionRateData rxnratedata;
  rxnratedata.type = griffon::ReactionRateType::TROE;
  rxnratedata.reversible = reversible;
  rxnratedata.hasOrders = false;
  rxnratedata.set_fwd_pre_exponential(fwd_pre_exp_value);
  rxnratedata.set_fwd_temp_exponent(fwd_temp_exponent);
  rxnratedata.set_fwd_activation_energy(fwd_act_energy);
  rxnratedata.set_reactants(reactants_stoich, mechanismData.phaseData);
  rxnratedata.set_products(products_stoich, mechanismData.phaseData);
  rxnratedata.set_three_body_efficiencies(three_body_efficiencies, default_efficiency, mechanismData.phaseData);
  rxnratedata.set_falloff_pre_exponential(flf_pre_exp_value);
  rxnratedata.set_falloff_temp_exponent(flf_temp_exponent);
  rxnratedata.set_falloff_activation_energy(flf_act_energy);
  rxnratedata.set_troe_parameters(troe_parameters);
  rxnratedata.finalize(mechanismData.phaseData);
  mechanismData.reactionData.reactions.push_back(rxnratedata);
  ++mechanismData.reactionData.nReactions;
}

void CombustionKernels::mechanism_add_reaction_simple_with_special_orders(const std::map<std::string, int>& reactants_stoich,
    const std::map<std::string, int>& products_stoich, const bool reversible, const double& fwd_pre_exp_value,
    const double& fwd_temp_exponent, const double& fwd_act_energy, const std::map<std::string, double>& special_orders) {
  typename griffon::ReactionData<6,52>::ReactionRateData rxnratedata;
  rxnratedata.type = griffon::ReactionRateType::SIMPLE;
  rxnratedata.reversible = reversible;
  rxnratedata.hasOrders = true;
  rxnratedata.set_fwd_pre_exponential(fwd_pre_exp_value);
  rxnratedata.set_fwd_temp_exponent(fwd_temp_exponent);
  rxnratedata.set_fwd_activation_energy(fwd_act_energy);
  rxnratedata.set_reactants(reactants_stoich, mechanismData.phaseData);
  rxnratedata.set_products(products_stoich, mechanismData.phaseData);
  rxnratedata.set_special_orders(special_orders, mechanismData.phaseData);
  rxnratedata.finalize(mechanismData.phaseData);
  mechanismData.reactionData.reactions.push_back(rxnratedata);
  ++mechanismData.reactionData.nReactions;
}

void CombustionKernels::mechanism_add_reaction_three_body_with_special_orders(const std::map<std::string, int>& reactants_stoich,
    const std::map<std::string, int>& products_stoich, const bool reversible, const double& fwd_pre_exp_value,
    const double& fwd_temp_exponent, const double& fwd_act_energy, const std::map<std::string, double>& three_body_efficiencies,
    const double& default_efficiency, const std::map<std::string, double>& special_orders) {
  typename griffon::ReactionData<6,52>::ReactionRateData rxnratedata;
  rxnratedata.type = griffon::ReactionRateType::THIRD_BODY;
  rxnratedata.reversible = reversible;
  rxnratedata.hasOrders = true;
  rxnratedata.set_fwd_pre_exponential(fwd_pre_exp_value);
  rxnratedata.set_fwd_temp_exponent(fwd_temp_exponent);
  rxnratedata.set_fwd_activation_energy(fwd_act_energy);
  rxnratedata.set_reactants(reactants_stoich, mechanismData.phaseData);
  rxnratedata.set_products(products_stoich, mechanismData.phaseData);
  rxnratedata.set_special_orders(special_orders, mechanismData.phaseData);
  rxnratedata.set_three_body_efficiencies(three_body_efficiencies, default_efficiency, mechanismData.phaseData);
  rxnratedata.finalize(mechanismData.phaseData);
  mechanismData.reactionData.reactions.push_back(rxnratedata);
  ++mechanismData.reactionData.nReactions;
}

void CombustionKernels::mechanism_add_reaction_Lindemann_with_special_orders(const std::map<std::string, int>& reactants_stoich,
    const std::map<std::string, int>& products_stoich, const bool reversible, const double fwd_pre_exp_value,
    const double fwd_temp_exponent, const double fwd_act_energy, const std::map<std::string, double>& three_body_efficiencies,
    const double& default_efficiency, const double flf_pre_exp_value, const double flf_temp_exponent, const double flf_act_energy,
    const std::map<std::string, double>& special_orders) {
  typename griffon::ReactionData<6,52>::ReactionRateData rxnratedata;
  rxnratedata.type = griffon::ReactionRateType::LINDEMANN;
  rxnratedata.reversible = reversible;
  rxnratedata.hasOrders = true;
  rxnratedata.set_fwd_pre_exponential(fwd_pre_exp_value);
  rxnratedata.set_fwd_temp_exponent(fwd_temp_exponent);
  rxnratedata.set_fwd_activation_energy(fwd_act_energy);
  rxnratedata.set_reactants(reactants_stoich, mechanismData.phaseData);
  rxnratedata.set_products(products_stoich, mechanismData.phaseData);
  rxnratedata.set_special_orders(special_orders, mechanismData.phaseData);
  rxnratedata.set_three_body_efficiencies(three_body_efficiencies, default_efficiency, mechanismData.phaseData);
  rxnratedata.set_falloff_pre_exponential(flf_pre_exp_value);
  rxnratedata.set_falloff_temp_exponent(flf_temp_exponent);
  rxnratedata.set_falloff_activation_energy(flf_act_energy);
  rxnratedata.finalize(mechanismData.phaseData);
  mechanismData.reactionData.reactions.push_back(rxnratedata);
  ++mechanismData.reactionData.nReactions;
}

void CombustionKernels::mechanism_add_reaction_Troe_with_special_orders(const std::map<std::string, int>& reactants_stoich,
    const std::map<std::string, int>& products_stoich, const bool reversible, const double fwd_pre_exp_value,
    const double fwd_temp_exponent, const double fwd_act_energy, const std::map<std::string, double>& three_body_efficiencies,
    const double& default_efficiency, const double flf_pre_exp_value, const double flf_temp_exponent, const double flf_act_energy,
    const std::vector<double>& troe_parameters, const std::map<std::string, double>& special_orders) {
  typename griffon::ReactionData<6,52>::ReactionRateData rxnratedata;
  rxnratedata.type = griffon::ReactionRateType::TROE;
  rxnratedata.reversible = reversible;
  rxnratedata.hasOrders = true;
  rxnratedata.set_fwd_pre_exponential(fwd_pre_exp_value);
  rxnratedata.set_fwd_temp_exponent(fwd_temp_exponent);
  rxnratedata.set_fwd_activation_energy(fwd_act_energy);
  rxnratedata.set_reactants(reactants_stoich, mechanismData.phaseData);
  rxnratedata.set_products(products_stoich, mechanismData.phaseData);
  rxnratedata.set_special_orders(special_orders, mechanismData.phaseData);
  rxnratedata.set_three_body_efficiencies(three_body_efficiencies, default_efficiency, mechanismData.phaseData);
  rxnratedata.set_falloff_pre_exponential(flf_pre_exp_value);
  rxnratedata.set_falloff_temp_exponent(flf_temp_exponent);
  rxnratedata.set_falloff_activation_energy(flf_act_energy);
  rxnratedata.set_troe_parameters(troe_parameters);
  rxnratedata.finalize(mechanismData.phaseData);
  mechanismData.reactionData.reactions.push_back(rxnratedata);
  ++mechanismData.reactionData.nReactions;
}

template<int NSR, int NTB>
void ReactionData<NSR, NTB>::ReactionRateData::set_fwd_pre_exponential(const double& v) {
  kFwdCoefs[0] = v;
}
template<int NSR, int NTB>
void ReactionData<NSR, NTB>::ReactionRateData::set_fwd_temp_exponent(const double& v) {
  kFwdCoefs[1] = v;
}
template<int NSR, int NTB>
void ReactionData<NSR, NTB>::ReactionRateData::set_fwd_activation_energy(const double& v) {
  kFwdCoefs[2] = v;
}
template<int NSR, int NTB>
void ReactionData<NSR, NTB>::ReactionRateData::set_reactants(const std::map<std::string, int>& reactants_stoich, const PhaseData& pd) {
  set_reactants_or_products(reactants_stoich, pd, true);
}
template<int NSR, int NTB>
void ReactionData<NSR, NTB>::ReactionRateData::set_products(const std::map<std::string, int>& products_stoich, const PhaseData& pd) {
  set_reactants_or_products(products_stoich, pd, false);
}
template<int NSR, int NTB>
void ReactionData<NSR, NTB>::ReactionRateData::set_reactants_or_products(const std::map<std::string, int>& ss_map, const PhaseData& pd,
    const bool is_reactants) {
  int i = 0;
  for (const auto& species_stoich : ss_map) {
    const int speciesIndex = pd.speciesIndices.at(species_stoich.first);
    if (is_reactants) {
      reactant_indices[i] = speciesIndex;
      reactant_stoich[i] = species_stoich.second;
      reactant_invmw[i] = pd.inverseMolecularWeights[speciesIndex];
    } else {
      product_indices[i] = speciesIndex;
      product_stoich[i] = -species_stoich.second;
      product_invmw[i] = pd.inverseMolecularWeights[speciesIndex];
    }
    ++i;
  }
  if (is_reactants) {
    n_reactants = i;
  } else {
    n_products = i;
  }
}
template<int NSR, int NTB>
void ReactionData<NSR, NTB>::ReactionRateData::set_three_body_efficiencies(const std::map<std::string, double>& eff_map,
    const double& default_efficiency, const PhaseData& pd) {
  thdBdyDefault = default_efficiency;
  int i = 0;
  for (const auto& rs : eff_map) {
    const int speciesIndex = pd.speciesIndices.at(rs.first);
    tb_indices[i] = speciesIndex;
    tb_invmw[i] = pd.inverseMolecularWeights[speciesIndex];
    tb_efficiencies[i] = tb_invmw[i] * (rs.second - thdBdyDefault);
    ++i;
  }
  n_tb = i;
}
template<int NSR, int NTB>
void ReactionData<NSR, NTB>::ReactionRateData::set_special_orders(const std::map<std::string, double>& special_orders_inp,
    const PhaseData& pd) {
  int i = 0;
  for (const auto& rs : special_orders_inp) {
    const int speciesIndex = pd.speciesIndices.at(rs.first);
    special_indices[i] = speciesIndex;
    special_invmw[i] = pd.inverseMolecularWeights[speciesIndex];
    special_orders[i] = rs.second;
    special_nonzero[i] = std::abs(special_orders[i]) > 1.e-12;
    ++i;
  }
  n_special = i;
}
template<int NSR, int NTB>
void ReactionData<NSR, NTB>::ReactionRateData::set_falloff_pre_exponential(const double& v) {
  kPressureCoefs[0] = v;
}
template<int NSR, int NTB>
void ReactionData<NSR, NTB>::ReactionRateData::set_falloff_temp_exponent(const double& v) {
  kPressureCoefs[1] = v;
}
template<int NSR, int NTB>
void ReactionData<NSR, NTB>::ReactionRateData::set_falloff_activation_energy(const double& v) {
  kPressureCoefs[2] = v;
}
template<int NSR, int NTB>
void ReactionData<NSR, NTB>::ReactionRateData::set_troe_parameters(const std::vector<double>& troe_params) {
  troeParams[0] = troe_params[0];
  troeParams[1] = troe_params[1];
  troeParams[2] = troe_params[2];
  troeParams[3] = troe_params[3];
}
template<int NSR, int NTB>
void ReactionData<NSR, NTB>::ReactionRateData::finalize(const PhaseData& pd) {
  {
    sumStoich = 0;
    std::map<int, std::tuple<int, double>> net_indices_stoichs; // spec idx to stoich, mw, invmw
    std::array<int, 2 * NSR> temp_net_indices;
    std::array<int, 2 * NSR> temp_net_stoich;
    std::array<int, 2 * NSR> temp_net_mw;
    std::array<int, 2 * NSR> temp_net_invmw;

    for (int i = 0; i < n_reactants; ++i) {
      sumStoich += reactant_stoich[i];
      net_indices_stoichs.insert(std::make_pair(reactant_indices[i], std::make_tuple(reactant_stoich[i], reactant_invmw[i])));
    }
    for (int i = 0; i < n_products; ++i) {
      sumStoich += product_stoich[i];
      if (net_indices_stoichs.find(product_indices[i]) == net_indices_stoichs.end()) {
        net_indices_stoichs.insert(std::make_pair(product_indices[i], std::make_tuple(product_stoich[i], product_invmw[i])));
      } else {
        const auto found_tuple = net_indices_stoichs.at(product_indices[i]);
        const auto stoich = std::get<0>(found_tuple);
        const auto invmw = std::get<1>(found_tuple);
        net_indices_stoichs.erase(product_indices[i]);
        net_indices_stoichs.insert(std::make_pair(product_indices[i], std::make_tuple(stoich + product_stoich[i], invmw)));
      }
    }
    n_net = 0;
    for (const auto& net_index : net_indices_stoichs) {
      const auto index = net_index.first;
      const auto stoich = std::get<0>(net_index.second);
      const auto invmw = std::get<1>(net_index.second);
      if (std::abs(stoich) > 1.e-14) {
        net_indices[n_net] = index;
        net_stoich[n_net] = stoich;
        net_mw[n_net] = 1. / invmw;
        ++n_net;
      }
    }
    if (n_net < 2 or n_net > 6) {
      throw std::runtime_error("bad number of net reacting species, need 2 < n < 7");
    }

    is_dense = false;
    for(int i=0; i<n_net; ++i){
      if(net_indices[i] == pd.nSpecies-1){
        is_dense = true;
      }
    }
    for(int i=0; i<n_tb; ++i){
      if(tb_indices[i] == pd.nSpecies-1){
        is_dense = true;
      }
    }
    n_sens = 0;
    if(not is_dense){
      for(int i=0; i<NJR; ++i){
        sens_indices[i] = -1;
      }
      for(int i=0; i<n_reactants; ++i){
        const int idx = reactant_indices[i];
        if(std::find(sens_indices.end(),
                     sens_indices.end(),
                     idx) == sens_indices.end()){
          sens_indices[n_sens] = idx;
          n_sens++;
        }
      }
      for(int i=0; i<n_products; ++i){
        const int idx = product_indices[i];
        if(std::find(sens_indices.end(),
                     sens_indices.end(),
                     idx) == sens_indices.end()){
          sens_indices[n_sens] = idx;
          n_sens++;
        }
      }
      for(int i=0; i<n_tb; ++i){
        const int idx = tb_indices[i];
        if(std::find(sens_indices.end(),
                     sens_indices.end(),
                     idx) == sens_indices.end()){
          sens_indices[n_sens] = idx;
          n_sens++;
        }
      }
    }

    switch (n_reactants) {
    case 1:
      if (reactant_stoich[0] == 1)
        forwardOrder = griffon::ReactionOrder::ONE;
      else if (reactant_stoich[0] == 2)
        forwardOrder = griffon::ReactionOrder::TWO;
      else
        forwardOrder = griffon::ReactionOrder::OTHER;
      break;
    case 2:
      if (reactant_stoich[0] == 1 && reactant_stoich[1] == 1)
        forwardOrder = griffon::ReactionOrder::ONE_ONE;
      else if (reactant_stoich[0] == 1 && reactant_stoich[1] == 2)
        forwardOrder = griffon::ReactionOrder::ONE_TWO;
      else if (reactant_stoich[0] == 2 && reactant_stoich[1] == 1)
        forwardOrder = griffon::ReactionOrder::TWO_ONE;
      else
        forwardOrder = griffon::ReactionOrder::OTHER;
      break;
    case 3:
      if (reactant_stoich[0] == 1 && reactant_stoich[1] == 1 && reactant_stoich[2] == 1)
        forwardOrder = griffon::ReactionOrder::ONE_ONE_ONE;
      else
        forwardOrder = griffon::ReactionOrder::OTHER;
      break;
    default:
      forwardOrder = griffon::ReactionOrder::OTHER;
    }
    sumReactantStoich = 0;
    for (int s = 0; s < n_reactants; ++s) {
      sumReactantStoich += std::abs(reactant_stoich[s]);
    }

    switch (n_products) {
    case 1:
      if (product_stoich[0] == -1)
        reverseOrder = griffon::ReactionOrder::ONE;
      else if (product_stoich[0] == -2)
        reverseOrder = griffon::ReactionOrder::TWO;
      else
        reverseOrder = griffon::ReactionOrder::OTHER;
      break;
    case 2:
      if (product_stoich[0] == -1 && product_stoich[1] == -1)
        reverseOrder = griffon::ReactionOrder::ONE_ONE;
      else if (product_stoich[0] == -1 && product_stoich[1] == -2)
        reverseOrder = griffon::ReactionOrder::ONE_TWO;
      else if (product_stoich[0] == -2 && product_stoich[1] == -1)
        reverseOrder = griffon::ReactionOrder::TWO_ONE;
      else
        reverseOrder = griffon::ReactionOrder::OTHER;
      break;
    case 3:
      if (product_stoich[0] == -1 && product_stoich[1] == -1 && product_stoich[2] == -1)
        reverseOrder = griffon::ReactionOrder::ONE_ONE_ONE;
      else
        reverseOrder = griffon::ReactionOrder::OTHER;
      break;
    default:
      reverseOrder = griffon::ReactionOrder::OTHER;
    }
    sumProductStoich = 0;
    for (int s = 0; s < n_products; ++s) {
      sumProductStoich += std::abs(product_stoich[s]);
    }

    if (std::abs(kFwdCoefs[2]) < 1.e-6) { // if the activation energy is zero
      if (std::abs(kFwdCoefs[1]) < 1.e-6)
        kForm = griffon::RateConstantTemperatureForm::CONSTANT;   // b=0
      else if (std::abs(kFwdCoefs[1] - 1) < 1.e-6)
        kForm = griffon::RateConstantTemperatureForm::LINEAR;     // b=1
      else if (std::abs(kFwdCoefs[1] - 2) < 1.e-6)
        kForm = griffon::RateConstantTemperatureForm::QUADRATIC;  // b=2
      else if (std::abs(kFwdCoefs[1] + 1) < 1.e-6)
        kForm = griffon::RateConstantTemperatureForm::RECIPROCAL; // b=-1
      else
        kForm = griffon::RateConstantTemperatureForm::ARRHENIUS; // non-integer b
    } else
      kForm = griffon::RateConstantTemperatureForm::ARRHENIUS;

    switch (type) {
    case griffon::ReactionRateType::SIMPLE:
    case griffon::ReactionRateType::THIRD_BODY:
    case griffon::ReactionRateType::LINDEMANN:
      troeForm = griffon::TroeTermsPresent::NO_TROE_TERMS;
      break;
    case griffon::ReactionRateType::TROE:
      troeForm = griffon::TroeTermsPresent::NO_TROE_TERMS;
      if (std::abs(troeParams[1]) > 1.e-8) {
        if (std::abs(troeParams[2]) > 1.e-8) {
          if (std::abs(troeParams[3]) > 1.e-8)
            troeForm = griffon::TroeTermsPresent::T123;
          else
            troeForm = griffon::TroeTermsPresent::T12;
        } else if (std::abs(troeParams[3]) > 1.e-8)
          troeForm = griffon::TroeTermsPresent::T13;
        else
          troeForm = griffon::TroeTermsPresent::T1;
      } else {
        if (std::abs(troeParams[2]) > 1.e-8) {
          if (std::abs(troeParams[3]) > 1.e-8)
            troeForm = griffon::TroeTermsPresent::T23;
          else
            troeForm = griffon::TroeTermsPresent::T2;
        } else if (std::abs(troeParams[3]) > 1.e-8)
          troeForm = griffon::TroeTermsPresent::T3;
      }
      break;
    default: {
      throw std::runtime_error("unknown reaction type");
    }
    }
  }
}

}

