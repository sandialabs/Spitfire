/* 
 * Spitfire - a Python-C++ library for building tabulated chemistry models and solving differential equations                    
 * Copyright 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS). 
 *                      
 * You should have received a copy of the 3-clause BSD License                                        
 * along with this program.  If not, see <https://opensource.org/licenses/BSD-3-Clause>.   
 *                   
 * Questions? Contact Mike Hansen (mahanse@sandia.gov)    
 */

/*
 * This file includes Griffon's combustion chemistry kernels, predominantly methods
 * for reaction rates of large mechanisms and ODE/PDE methods needed for solving
 * reactor and flamelet models.
 */

#ifndef GRIFFON_COMBUSTION_H
#define GRIFFON_COMBUSTION_H

#include <map>
#include <string>
#include <vector>
#include <array>

namespace griffon
{

  /*
   * @class CombustionKernels
   * @brief objects of this class contain mechanism data and do thermodynamics, kinetics, reactor, and flamelet calculations
   */
  class CombustionKernels
  {
  public:

    /*
     * @enum CpType
     * @brief types of heat capacity polynomials
     *
     * - CONSTANT: constant heat capacity (calorically perfect gas)
     * - NASA7   : 7-coefficient polynomial form heat capacity
     */
    enum class CpType
    {
      UNKNOWN, CONST, NASA7
    };

    /*
     * @enum RateType
     * @brief types of reaction rates
     *
     * - SIMPLE: simple reactions with a canonical mass-action rate law
     * - THIRD_BODY: third-body-enhanced reactions
     * - LINDEMANN: pressure falloff reaction of a Lindemann form
     * - TROE: pressure falloff reaction with a Troe parameterization
     */
    enum class RateType
    {
      UNKNOWN, SIMPLE, THIRD_BODY, LINDEMANN, TROE
    };

    /*
     * @class RateConstantTForm
     * @brief type of rate constant temperature dependence
     * This is important in preventing unnecessary expensive exponential evaluations,
     * such as when the activation energy is zero.
     */
    enum class RateConstantTForm
    {
      CONSTANT, LINEAR, QUADRATIC, RECIPROCAL, ARRHENIUS
    };

    /*
     * @class TroeTermsPresent
     * @brief which of the Troe terms are present in a Troe falloff reaction rate
     */
    enum class TroeTermsPresent
    {
      NO_TROE_TERMS, T1, T2, T12, T3, T13, T23, T123
    };

    /*
     * @class ReactionOrder
     * @brief special versions of stoichiometry coefficients in a reaction
     * This is helpful in avoiding exponentiation of concentrations and simplifies Jacobian calculation as well.
     */
    enum class ReactionOrder
    {
      ONE, TWO, ONE_ONE, TWO_ONE, ONE_TWO, ONE_ONE_ONE, OTHER
    };

    /*
     * @brief get a map from element names to element molecular weights
     * This is used in computing species molecular weights.
     */
    std::map<std::string, double>
    get_element_mw_map ();

    /*
     * @class PhaseData
     * @brief data describing elements and species present in a gas model
     */
    struct PhaseData
    {
      int nElements;
      int nSpecies;
      std::vector<std::string> elementNames;
      std::vector<std::string> speciesNames;
      std::map<std::string, int> speciesIndices;
      std::vector<double> molecularWeights;
      std::vector<double> inverseMolecularWeights;
      double referenceTemperature;
      double referencePressure;
      static constexpr double Ru = 8314.4621; // universal gas constant
    };

    /*
     * @class HeatCapacityData
     * @brief heat capacity polynomial coefficients
     *
     * template parameters:
     *  NCP: maximum allowed number of coefficients
     */
    template<int NCP>
      struct HeatCapacityData
      {
        std::vector<std::array<double, NCP> > coefficients;
        std::vector<double> minTemperatures;
        std::vector<double> maxTemperatures;
        std::vector<CpType> types;
      };

    /*
     * @class ReactionData
     * @brief reaction data, just a container for ReactionRateData objects
     *
     * template parameters:
     *  NSR: maximum allowed number of reactants, products, net species in a reaction, or of species in a nonelementary rate expression
     */
    template<int NSR>
      struct ReactionData
      {
        /*
         * @class ReactionRateData
         * @brief data and setters for single chemical reactions
         *
         * reactant_*: data corresponding to reactants (consumed in the forward reaction)
         * product_*: data correspodning to products (consumed in the reverse reaction, if reversible)
         * net_*: data corresponding to species that do not appear equally on each side of the reaction (net production/consumption)
         * tb_*: data corresponding to species with specified three-body efficiencies
         * special_*: data corresponding to species with specified orders in a nonelementary reaction
         *
         * hasOrders: true if the reaction is nonelementary, false if elementary
         */
        struct ReactionRateData
        {
          std::array<int, NSR> reactant_indices;
          std::array<int, NSR> reactant_stoich;
          std::array<double, NSR> reactant_invmw;
          int n_reactants = 0;

          std::array<int, NSR> product_indices;
          std::array<int, NSR> product_stoich;
          std::array<double, NSR> product_invmw;
          int n_products = 0;

          std::array<int, NSR> net_indices;
          std::array<int, NSR> net_stoich;
          std::array<double, NSR> net_mw;
          int n_net = 0;

          std::vector<int> tb_indices;
          std::vector<double> tb_invmw;
          std::vector<double> tb_efficiencies;
          int n_tb = 0;

          bool hasOrders = false;
          std::array<int, NSR> special_indices;
          std::array<double, NSR> special_orders;
          std::array<bool, NSR> special_nonzero;
          std::array<double, NSR> special_invmw;
          int n_special = 0;

          bool is_dense = false;
          int n_sens = 0;
          std::vector<int> sens_indices;

          double thdBdyDefault;
          std::array<double, 3> kFwdCoefs;
          std::array<double, 3> kPressureCoefs;
          std::array<double, 4> troeParams;

          bool reversible;
          RateType type;
          RateConstantTForm kForm;
          TroeTermsPresent troeForm;

          ReactionOrder forwardOrder;
          ReactionOrder reverseOrder;
          int sumStoich;
          int sumReactantStoich;
          int sumProductStoich;

          void
          set_fwd_pre_exponential (const double& v);
          void
          set_fwd_temp_exponent (const double& v);
          void
          set_fwd_activation_energy (const double& v);
          void
          set_falloff_pre_exponential (const double& v);
          void
          set_falloff_temp_exponent (const double& v);
          void
          set_falloff_activation_energy (const double& v);
          void
          set_reactants (const std::map<std::string, int>& reactants_stoich, const PhaseData& pd);
          void
          set_products (const std::map<std::string, int>& products_stoich, const PhaseData& pd);
          void
          set_reactants_or_products (const std::map<std::string, int>& ss_map, const PhaseData& pd,
                                     const bool is_reactants);
          void
          set_special_orders (const std::map<std::string, double>& orders, const PhaseData& pd);
          void
          set_three_body_efficiencies (const std::map<std::string, double>& eff_map, const double& default_efficiency,
                                       const PhaseData& pd);
          void
          set_troe_parameters (const std::vector<double>& troe_params);
          void
          finalize (const PhaseData& phaseData);
        };

        int nReactions;
        std::vector<ReactionData::ReactionRateData> reactions;
      };

    /*
     * @class MechanismData
     * @brief collection of phase, heat capacity, and reaction data
     */
    template<int NSRn, int NCPn>
      struct MechanismData
      {
        PhaseData phaseData;
        HeatCapacityData<NCPn> heatCapacityData;
        ReactionData<NSRn> reactionData;

        static constexpr int NSR = NSRn;
        static constexpr int NCP = NCPn;
      };

  public:
    /*
     * mechanism data setter methods
     *
     * To set mechanism data:
     * 1. add elements (mechanism_add_element)
     * 2. set reference pressure and temperature (mechanism_set_ref_pressure, mechanism_set_ref_temperature)
     * 3. add species with atom maps (mechanism_add_species)
     * 4. call mechanism_resize_heat_capacity_data()
     * 5. add the heat capacity polynomials with either mechanism_add_const_cp or mechanism_add_nasa7_cp
     * 6. add reactions with the methods below. The *_with_special_orders methods are for nonelementary reactions.
     */
    void
    mechanism_add_element (const std::string& element_name);
    void
    mechanism_add_species (const std::string& species_name, const std::map<std::string, double> atom_map);
    void
    mechanism_set_ref_pressure (const double& p_ref);
    void
    mechanism_set_ref_temperature (const double& T_ref);
    void
    mechanism_resize_heat_capacity_data ();
    void
    mechanism_add_const_cp (const std::string& spec_name, const double& Tmin, const double& Tmax, const double& T0,
                            const double& h0, const double& s0, const double& cp);
    void
    mechanism_add_nasa7_cp (const std::string& spec_name, const double& Tmin, const double& Tmid, const double& Tmax,
                            const std::vector<double>& low_coeffs, const std::vector<double>& high_coeffs);
    void
    mechanism_add_reaction_simple (const std::map<std::string, int>& reactants_stoich,
                                   const std::map<std::string, int>& products_stoich, const bool reversible,
                                   const double& fwd_pre_exp_value, const double& fwd_temp_exponent,
                                   const double& fwd_act_energy);
    void
    mechanism_add_reaction_three_body (const std::map<std::string, int>& reactants_stoich,
                                       const std::map<std::string, int>& products_stoich, const bool reversible,
                                       const double& fwd_pre_exp_value, const double& fwd_temp_exponent,
                                       const double& fwd_act_energy,
                                       const std::map<std::string, double>& three_body_efficiencies,
                                       const double& default_efficiency);
    void
    mechanism_add_reaction_Lindemann (const std::map<std::string, int>& reactants_stoich,
                                      const std::map<std::string, int>& products_stoich, const bool reversible,
                                      const double fwd_pre_exp_value, const double fwd_temp_exponent,
                                      const double fwd_act_energy,
                                      const std::map<std::string, double>& three_body_efficiencies,
                                      const double& default_efficiency, const double flf_pre_exp_value,
                                      const double flf_temp_exponent, const double flf_act_energy);
    void
    mechanism_add_reaction_Troe (const std::map<std::string, int>& reactants_stoich,
                                 const std::map<std::string, int>& products_stoich, const bool reversible,
                                 const double fwd_pre_exp_value, const double fwd_temp_exponent,
                                 const double fwd_act_energy,
                                 const std::map<std::string, double>& three_body_efficiencies,
                                 const double& default_efficiency, const double flf_pre_exp_value,
                                 const double flf_temp_exponent, const double flf_act_energy,
                                 const std::vector<double>& troe_parameters);
    void
    mechanism_add_reaction_simple_with_special_orders (const std::map<std::string, int>& reactants_stoich,
                                                       const std::map<std::string, int>& products_stoich,
                                                       const bool reversible, const double& fwd_pre_exp_value,
                                                       const double& fwd_temp_exponent, const double& fwd_act_energy,
                                                       const std::map<std::string, double>& special_orders);
    void
    mechanism_add_reaction_three_body_with_special_orders (const std::map<std::string, int>& reactants_stoich,
                                                           const std::map<std::string, int>& products_stoich,
                                                           const bool reversible, const double& fwd_pre_exp_value,
                                                           const double& fwd_temp_exponent,
                                                           const double& fwd_act_energy,
                                                           const std::map<std::string, double>& three_body_efficiencies,
                                                           const double& default_efficiency,
                                                           const std::map<std::string, double>& special_orders);
    void
    mechanism_add_reaction_Lindemann_with_special_orders (const std::map<std::string, int>& reactants_stoich,
                                                          const std::map<std::string, int>& products_stoich,
                                                          const bool reversible, const double fwd_pre_exp_value,
                                                          const double fwd_temp_exponent, const double fwd_act_energy,
                                                          const std::map<std::string, double>& three_body_efficiencies,
                                                          const double& default_efficiency,
                                                          const double flf_pre_exp_value,
                                                          const double flf_temp_exponent, const double flf_act_energy,
                                                          const std::map<std::string, double>& special_orders);
    void
    mechanism_add_reaction_Troe_with_special_orders (const std::map<std::string, int>& reactants_stoich,
                                                     const std::map<std::string, int>& products_stoich,
                                                     const bool reversible, const double fwd_pre_exp_value,
                                                     const double fwd_temp_exponent, const double fwd_act_energy,
                                                     const std::map<std::string, double>& three_body_efficiencies,
                                                     const double& default_efficiency, const double flf_pre_exp_value,
                                                     const double flf_temp_exponent, const double flf_act_energy,
                                                     const std::vector<double>& troe_parameters,
                                                     const std::map<std::string, double>& special_orders);

    /*
     * general-purpose thermodynamics evaluations
     */
    double
    mixture_molecular_weight (const double *y) const;
    void
    mole_fractions (const double *y, double *x) const;
    double
    ideal_gas_density (const double &pressure, const double &temperature, const double *y) const;
    double
    ideal_gas_pressure (const double &density, const double &temperature, const double *y) const;
    double
    cp_mix (const double &temperature, const double *y) const;
    double
    cv_mix (const double &temperature, const double *y) const;
    double
    enthalpy_mix (const double &temperature, const double *y) const;
    double
    energy_mix (const double &temperature, const double *y) const;
    void
    species_cp (const double &temperature, double *out_cpspecies) const;
    void
    species_cv (const double &temperature, double *out_cvspecies) const;
    void
    species_enthalpies (const double &temperature, double *out_enthalpies) const;
    void
    species_energies (const double &temperature, double *out_energies) const;

    /*
     * general-purpose kinetics evaluations
     */
    void
    production_rates (const double &temperature, const double &density, const double *y, double *out_prodrates) const;
    void
    prod_rates_primitive_sensitivities (const double &density, const double &temperature, const double *y,
                                        int rates_sensitivity_option, double *out_prodratessens) const;

    /*
     * reactor RHS and Jacobian methods
     */
    void
    reactor_rhs_isobaric (const double *state, const double &pressure, const double &inflowTemperature,
                          const double *inflowY, const double &tau, const double &fluidTemperature,
                          const double &surfTemperature, const double &hConv, const double &epsRad,
                          const double &surfaceAreaOverVolume, const int heat_transfer_option, const bool open,
                          double *out_rhs) const;
    void
    reactor_jac_isobaric (const double *state, const double &pressure, const double &inflowTemperature,
                          const double *inflowY, const double &tau, const double &fluidTemperature,
                          const double &surfTemperature, const double &hConv, const double &epsRad,
                          const double &surfaceAreaOverVolume, const int heat_transfer_option, const bool open,
                          const int rates_sensitivity_option, const int sensitivity_transform_option, double *out_rhs,
                          double *out_jac) const;
    void
    reactor_rhs_isochoric (const double *state, const double &inflowDensity, const double &inflowTemperature,
                           const double *inflowY, const double &tau, const double &fluidTemperature,
                           const double &surfTemperature, const double &hConv, const double &epsRad,
                           const double &surfaceAreaOverVolume, const int heatTransferOption, const bool open,
                           double *out_rhs) const;
    void
    reactor_jac_isochoric (const double *state, const double &inflowDensity, const double &inflowTemperature,
                           const double *inflowY, const double &tau, const double &fluidTemperature,
                           const double &surfTemperature, const double &hConv, const double &epsRad,
                           const double &surfaceAreaOverVolume, const int heatTransferOption, const bool open,
                           const int rates_sensitivity_option, double *out_rhs, double *out_jac) const;

    /*
     * flamelet RHS, Jacobian, and post-processor methods
     */
    void
    flamelet_rhs (const double *state, const double &pressure, const double *oxyState, const double *fuelState,
                  const bool adiabatic, const double *T_convection, const double *h_convection,
                  const double *T_radiation, const double *h_radiation, const int &nzi, const double *cmajor,
                  const double *csub, const double *csup, const double *mcoeff, const double *ncoeff,
                  const double *dissipationRate, const bool include_enthalpy_flux, const bool include_variable_cp,
                  const bool use_scaled_heat_loss, double *out_rhs) const;
    void
    flamelet_jacobian (const double *state, const double &pressure, const double *oxyState, const double *fuelState,
                       const bool adiabatic, const double *T_convection, const double *h_convection,
                       const double *T_radiation, const double *h_radiation, const int &nzi, const double *cmajor,
                       const double *csub, const double *csup, const double *mcoeff, const double *ncoeff,
                       const double *chi, const bool compute_eigenvalues, const double diffterm,
                       const bool scale_and_offset, const double prefactor, const int &rates_sensitivity_option,
                       const int &sensitivity_transform_option, const bool include_enthalpy_flux,
                       const bool include_variable_cp, const bool use_scaled_heat_loss, double *out_expeig,
                       double *out_jac) const;
    void
    flamelet_stencils (const double *dz, const int &nzi, const double *dissipationRate, const double *invLewisNumbers,
                       double *out_cmajor, double *out_csub, double *out_csup, double *out_mcoeff,
                       double *out_ncoeff) const;
    void
    flamelet_jac_indices (const int &nzi, int *out_row_indices, int *out_col_indices) const;

    /*
     * 2d flamelet methods (preliminary, serial implementation for block-Jacobi PBiCGStab)
     */
    void
    flamelet2d_rhs (const double *state, const double &pressure, const int &nx, const int &ny, const double *xcp,
                    const double *xcl, const double *xcr, const double *ycp, const double *ycb, const double *yct,
                    double *out_rhs) const;

    void
    flamelet2d_factored_block_diag_jacobian (const double *state, const double &pressure, const int &nx, const int &ny,
                                             const double *xcp, const double *ycp, const double &prefactor,
                                             double *out_values, double *out_factors, int *out_pivots) const;

    void
    flamelet2d_offdiag_matvec (const double *vec, const int &nx, const int &ny, const double *xcp, const double *xcl,
                               const double *xcr, const double *ycp, const double *ycb, const double *yct,
                               const double &prefactor, double *out_matvec) const;

    void
    flamelet2d_matvec (const double *vec, const int &nx, const int &ny, const double *xcp, const double *xcl,
                       const double *xcr, const double *ycp, const double *ycb, const double *yct,
                       const double &prefactor, const double *block_diag_values, double *out_matvec) const;

    void
    flamelet2d_block_diag_solve (const int &nx, const int &ny, const double *factors, const int *pivots,
                                 const double *b, double* out_x) const;

  private:
    using MechData = MechanismData<8, 15>;
    MechData mechanismData; // note: moving the template integers to CombustionKernels causes Cython problems...

    static constexpr int NSR = MechData::NSR;
    static constexpr int NCP = MechData::NCP;

    inline void
    extract_y (const double* ynm1, const int n, double* y) const
    {
      const int nSpec = mechanismData.phaseData.nSpecies;
      y[nSpec - 1] = 1.;
      for (int j = 0; j < nSpec - 1; ++j)
      {
        y[j] = ynm1[j];
        y[nSpec - 1] -= y[j];
      }
    }

    inline void
    ideal_gas_density (const double &pressure, const double &temperature, const double &mmw, double *out_density) const
    {
      *out_density = pressure * mmw / (temperature * mechanismData.phaseData.Ru);
    }
    inline void
    ideal_gas_pressure (const double &density, const double &temperature, const double &mmw, double *out_pressure) const
    {
      *out_pressure = density * temperature * mechanismData.phaseData.Ru / mmw;
    }
    void
    cp_mix_and_species (const double &temperature, const double *y, double *out_cvmix, double *out_cvspecies) const;
    void
    cv_mix_and_species (const double &temperature, const double *y, const double &mmw, double *out_cvmix,
                        double *out_cvspecies) const;
    void
    cp_sens_T (const double &temperature, const double *y, double *out_cpmixsens, double *out_cpspeciessens) const;
    inline void
    cv_sens_T (const double &temperature, const double *y, double *out_cvmixsens, double *out_cvspeciessens) const
    {
      cp_sens_T (temperature, y, out_cvmixsens, out_cvspeciessens);
    }
    void
    production_rates (const double &temperature, const double &density, const double &mmw, const double *y,
                      double *out_prodrates) const;
    void
    prod_rates_sens_exact (const double &temperature, const double &density, const double &mmw, const double *y,
                           double *out_prodrates, double *out_prodratessens) const;
    void
    prod_rates_sens_no_tbaf (const double &temperature, const double &density, const double &mmw, const double *y,
                             double *out_prodrates, double *out_prodratessens) const;
    void
    prod_rates_sens_sparse (const double &temperature, const double &density, const double &mmw, const double *y,
                            double *out_prodrates, double *out_prodratessens) const;
    void
    chem_rhs_isobaric (const double &rho, const double &cp, const double *h, const double *w, double *out_rhs) const;
    void
    heat_rhs_isobaric (const double &temperature, const double &rho, const double &cp, const double &fluidTemperature,
                       const double &surfTemperature, const double &hConv, const double &epsRad,
                       const double &surfaceAreaOverVolume, double *out_heatTransferRate) const;
    void
    mass_rhs_isobaric (const double *y, const double *enthalpies, const double *inflowEnthalpies, const double &rho,
                       const double &cp, const double *inflowY, const double &tau, double *out_rhs) const;
    void
    chem_jacexactdense_isobaric (const double &pressure, const double &temperature, const double *y, const double &mmw,
                                 const double &rho, const double &cp, const double *cpi, const double &cpsensT,
                                 const double *h, const double *w, const double *wsens, double *out_rhs,
                                 double *out_primJac) const;
    void
    mass_jacexactdense_isobaric (const double &pressure, const double &temperature, const double *y, const double &rho,
                                 const double &cp, const double &cpsensT, const double *cpi, const double *enthalpies,
                                 const double *inflowEnthalpies, const double &inflowTemperature, const double *inflowY,
                                 const double &tau, double *out_rhs, double *out_primJac) const;
    void
    heat_jacexactdense_isobaric (const double &temperature, const double &rho, const double &cp, const double &cpsensT,
                                 const double *cpi, const double &convectionTemperature,
                                 const double &radiationTemperature, const double &convectionCoefficient,
                                 const double &radiativeEmissivity, const double &surfaceAreaOverVolume,
                                 double *out_heatTransferRate, double *out_heatTransferRatePrimJac) const;
    void
    transform_isobaric_primitive_jacobian (const double &rho, const double &pressure, const double &temperature,
                                           const double &mmw, const double *primJac, double *out_jac) const;
    void
    chem_rhs_isochoric (const double &rho, const double &cv, const double *e, const double *w, double *out_rhs) const;
    void
    heat_rhs_isochoric (const double &temperature, const double &rho, const double &cv, const double &fluidTemperature,
                        const double &surfTemperature, const double &hConv, const double &epsRad,
                        const double &surfaceAreaOverVolume, double *out_heatTransferRate) const;
    void
    mass_rhs_isochoric (const double *y, const double *energies, const double *inflowEnergies, const double &rho,
                        const double &inflowRho, const double &pressure, const double &inflowPressure, const double &cv,
                        const double *inflowY, const double &tau, double *out_rhs) const;
    void
    chem_jacexactdense_isochoric (const double &temperature, const double *y, const double &rho, const double &cv,
                                  const double *cvi, const double &cvsensT, const double *e, const double *w,
                                  const double *wsens, double *out_rhs, double *out_jac) const;
    void
    mass_jacexactdense_isochoric (const double &pressure, const double &inflowPressure, const double &temperature,
                                  const double *y, const double &rho, const double &inflowRho, const double &cv,
                                  const double &cvsensT, const double *cvi, const double *energies,
                                  const double *inflowEnergies, const double &inflowTemperature, const double *inflowY,
                                  const double &tau, double *out_rhs, double *out_jac) const;
    void
    heat_jacexactdense_isochoric (const double &temperature, const double &rho, const double &cv, const double &cvsensT,
                                  const double *cvi, const double &convectionTemperature,
                                  const double &radiationTemperature, const double &convectionCoefficient,
                                  const double &radiativeEmissivity, const double &surfaceAreaOverVolume,
                                  double *out_heatTransferRate, double *out_heatTransferRateJac) const;
  };

}

#endif // GRIFFON_CALCULATOR_H
