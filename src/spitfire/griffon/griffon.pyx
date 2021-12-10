# distutils: language = c++
# cython: language_level=3

# Spitfire - a Python-C++ library for building tabulated chemistry models and solving differential equations                    
# Copyright 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS). 
#                       
# You should have received a copy of the 3-clause BSD License                                        
# along with this program.  If not, see <https://opensource.org/licenses/BSD-3-Clause>.   
#                    
# Questions? Contact Mike Hansen (mahanse@sandia.gov)    

import numpy as np
cimport numpy as np

cimport cython

from libcpp.string cimport string
from libcpp.map cimport map
from libcpp.vector cimport vector

from libcpp cimport bool

# This is the C++-side interface to the CombustionKernels class defined in the griffon C++ code
cdef extern from "combustion_kernels.h" namespace "griffon":
  cdef cppclass CombustionKernels:

    CombustionKernels() except +

    # mechanism data setters
    void mechanism_set_element_mw_map(const map[string, double] element_mw_map)
    void mechanism_add_element(const string& element_name)
    void mechanism_add_species(const string& species_name, const map[string, double] atom_map)
    void mechanism_set_ref_pressure(const double& p_ref)
    void mechanism_set_ref_temperature(const double& T_ref)
    void mechanism_set_gas_constant(const double& Ru)
    void mechanism_resize_heat_capacity_data()
    void mechanism_add_const_cp(const string& spec_name,
      const double& Tmin,
      const double& Tmax,
      const double& T0,
      const double& h0,
      const double& s0,
      const double& cp)
    void mechanism_add_nasa7_cp(const string& spec_name,
      const double& Tmin,
      const double& Tmid,
      const double& Tmax,
      const vector[double]& low_coeffs,
      const vector[double]& high_coeffs)
    void mechanism_add_nasa9_cp(const string& spec_name,
      const double& Tmin,
      const double& Tmax,
      const vector[double]& coeffs)
    void mechanism_add_reaction_simple(const map[string, int]& reactants_stoich,
      const map[string, int]& products_stoich,
      const bool reversible,
      const double& fwd_pre_exp_value,
      const double& fwd_temp_exponent,
      const double& fwd_act_energy)
    void mechanism_add_reaction_three_body(const map[string, int]& reactants_stoich,
      const map[string, int]& products_stoich,
      const bool reversible,
      const double& fwd_pre_exp_value,
      const double& fwd_temp_exponent,
      const double& fwd_act_energy,
      const map[string, double]& three_body_efficiencies,
      const double& default_efficiency)
    void mechanism_add_reaction_Lindemann(const map[string, int]& reactants_stoich,
      const map[string, int]& products_stoich,
      const bool reversible,
      const double fwd_pre_exp_value,
      const double fwd_temp_exponent,
      const double fwd_act_energy,
      const map[string, double]& three_body_efficiencies,
      const double& default_efficiency,
      const double flf_pre_exp_value,
      const double flf_temp_exponent,
      const double flf_act_energy)
    void mechanism_add_reaction_Troe(const map[string, int]& reactants_stoich,
      const map[string, int]& products_stoich,
      const bool reversible,
      const double fwd_pre_exp_value,
      const double fwd_temp_exponent,
      const double fwd_act_energy,
      const map[string, double]& three_body_efficiencies,
      const double& default_efficiency,
      const double flf_pre_exp_value,
      const double flf_temp_exponent,
      const double flf_act_energy,
      const vector[double]& troe_parameters)
    void mechanism_add_reaction_simple_with_special_orders(const map[string, int]& reactants_stoich,
      const map[string, int]& products_stoich,
      const bool reversible,
      const double& fwd_pre_exp_value,
      const double& fwd_temp_exponent,
      const double& fwd_act_energy,
      const map[string, double]& special_orders)
    void mechanism_add_reaction_three_body_with_special_orders(const map[string, int]& reactants_stoich,
      const map[string, int]& products_stoich,
      const bool reversible,
      const double& fwd_pre_exp_value,
      const double& fwd_temp_exponent,
      const double& fwd_act_energy,
      const map[string, double]& three_body_efficiencies,
      const double& default_efficiency,
      const map[string, double]& special_orders)
    void mechanism_add_reaction_Lindemann_with_special_orders(const map[string, int]& reactants_stoich,
      const map[string, int]& products_stoich,
      const bool reversible,
      const double fwd_pre_exp_value,
      const double fwd_temp_exponent,
      const double fwd_act_energy,
      const map[string, double]& three_body_efficiencies,
      const double& default_efficiency,
      const double flf_pre_exp_value,
      const double flf_temp_exponent,
      const double flf_act_energy,
      const map[string, double]& special_orders)
    void mechanism_add_reaction_Troe_with_special_orders(const map[string, int]& reactants_stoich,
      const map[string, int]& products_stoich,
      const bool reversible,
      const double fwd_pre_exp_value,
      const double fwd_temp_exponent,
      const double fwd_act_energy,
      const map[string, double]& three_body_efficiencies,
      const double& default_efficiency,
      const double flf_pre_exp_value,
      const double flf_temp_exponent,
      const double flf_act_energy,
      const vector[double]& troe_parameters,
      const map[string, double]& special_orders)

    # thermodynamics methods
    double mixture_molecular_weight( const double* y)
    void mole_fractions( const double* y, double* x )
    double ideal_gas_density( const double p, const double T, const double *y)
    double ideal_gas_pressure( const double rho, const double T, const double *y)
    double cp_mix(const double T, const double* y)
    double cv_mix(const double T, const double* y)
    double enthalpy_mix(const double T, const double* y)
    double energy_mix(const double T, const double* y)
    void species_cp(const double T, double *cpspecies)
    void species_cv(const double T, double *cvspecies)
    void species_enthalpies(const double T, double *enthalpyspecies)
    void species_energies(const double T, double *energyspecies)
    void cp_sens_T(const double &temperature, const double *y, double *out_cpmixsens, double *out_cpspeciessens)

    # kinetics methods
    void production_rates(const double T, const double rho, const double* y, double *prodrates)
    void prod_rates_primitive_sensitivities(const double T, const double rho, const double* y, int rates_sensitivity_option, double* sens)

    # isobaric reactor methods
    void reactor_rhs_isobaric(const double* state, const double p, const double Tin, const double* yin, const double tau, const double Tinfty, const double Tsurf, const double hConv, const double epsRad, const double SoV, const int heatTransferOption, const bool open, double* rhs)
    void reactor_jac_isobaric(const double* state, const double p, const double Tin, const double* yin, const double tau, const double Tinfty, const double Tsurf, const double hConv, const double epsRad, const double SoV, const int heatTransferOption, const bool open, int rates_sensitivity_option, int sensitivity_transform_option, double* rhs, double* jac)
    void reactor_rhs_isochoric(const double* state, const double rhoin, const double Tin, const double* yin, const double tau, const double Tinfty, const double Tsurf, const double hConv, const double epsRad, const double SoV, const int heatTransferOption, const bool open, double* rhs)
    void reactor_jac_isochoric(const double* state, const double rhoin, const double Tin, const double* yin, const double tau, const double Tinfty, const double Tsurf, const double hConv, const double epsRad, const double SoV, const int heatTransferOption, const bool open, int rates_sensitivity_option, double* rhs, double* jac)

    # flamelet methods
    void flamelet_stencils(const double *dz, const int nzi, const double *dissipationRate, const double *invLewisNumbers, double *out_cmajor, double *out_csub, double *out_csup, double *out_mcoeff, double *out_ncoeff)
    void flamelet_jac_indices(const int nzi, int *out_row_indices, int *out_col_indices)
    void flamelet_rhs(const double *state, const double pressure, const double *oxyState, const double *fuelState, const bool adiabatic, const double *T_convection, const double *h_convection, const double *T_radiation, const double *h_radiation, const int nzi, const double *cmajor, const double *csub, const double *csup, const double *mcoeff, const double *ncoeff, const double *chi, const bool include_enthalpy_flux, const bool include_variable_cp, const bool use_scaled_heat_loss, double *out_rhs)
    void flamelet_jacobian(const double *state, const double &pressure, const double *oxyState, const double *fuelState, const bool adiabatic, const double *T_convection, const double *h_convection, const double *T_radiation, const double *h_radiation, const int &nzi, const double *cmajor, const double *csub, const double *csup, const double *mcoeff, const double *ncoeff, const double *chi, const bool compute_eigenvalues, const double diffterm, const bool scale_and_offset, const double prefactor, const int &rates_sensitivity_option, const int &sensitivity_transform_option, const bool include_enthalpy_flux, const bool include_variable_cp, const bool use_scaled_heat_loss, double *out_expeig, double *out_jac )


    void flamelet2d_rhs(const double *state,
                            const double &pressure,
                            const int nx,
                            const int ny,
                            const double *xcp,
                            const double *xcl,
                            const double *xcr,
                            const double *ycp,
                            const double *ycb,
                            const double *yct,
                            double *out_rhs)

    void flamelet2d_factored_block_diag_jacobian( const double *state,
                                                      const double pressure,
                                                      const int nx,
                                                      const int ny,
                                                      const double *xcp,
                                                      const double *ycp,
                                                      const double prefactor,
                                                      double *out_values,
                                                      double *out_factors,
                                                      int* out_pivots)

    void flamelet2d_offdiag_matvec( const double *vec,
                                        const int nx,
                                        const int ny,
                                        const double *xcp,
                                        const double *xcl,
                                        const double *xcr,
                                        const double *ycp,
                                        const double *ycb,
                                        const double *yct,
                                        const double prefactor,
                                        double *out_vec)
    void flamelet2d_matvec( const double *vec,
                                const int &nx,
                                const int &ny,
                                const double *xcp,
                                const double *xcl,
                                const double *xcr,
                                const double *ycp,
                                const double *ycb,
                                const double *yct,
                                const double &prefactor,
                                const double *block_diag_values,
                                double *out_matvec )

    void flamelet2d_block_diag_solve( const int nx,
                                          const int ny,
                                          const double *factors,
                                          const int *pivots,
                                          const double *b,
                                          double* out_x)


cdef class PyCombustionKernels:
    """This is the Python-side interface to the CombustionKernels class defined in the griffon C++ code.
      """
    cdef CombustionKernels* c_calculator

    def __cinit__(self):
      self.c_calculator = new CombustionKernels()

    def __dealloc__(self):
      del self.c_calculator

    @cython.nonecheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def mechanism_set_element_mw_map(self, element_mw_map):
      cdef map[string, double] am
      for a in element_mw_map:
        am[a.encode()] = element_mw_map[a]
      self.c_calculator.mechanism_set_element_mw_map(am)

    @cython.nonecheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def mechanism_add_element(self, str element_name):
      self.c_calculator.mechanism_add_element(element_name.encode())

    @cython.nonecheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def mechanism_add_species(self, str species_name, dict atom_map):
      cdef map[string, double] am
      for a in atom_map:
        am[a.encode()] = atom_map[a]
      self.c_calculator.mechanism_add_species(species_name.encode(), am)

    @cython.nonecheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def mechanism_set_ref_pressure(self, double p):
      self.c_calculator.mechanism_set_ref_pressure(p)

    @cython.nonecheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def mechanism_set_ref_temperature(self, double T):
      self.c_calculator.mechanism_set_ref_temperature(T)

    @cython.nonecheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def mechanism_set_gas_constant(self, double Ru):
      self.c_calculator.mechanism_set_gas_constant(Ru)

    @cython.nonecheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def mechanism_resize_heat_capacity_data(self):
      self.c_calculator.mechanism_resize_heat_capacity_data()

    @cython.nonecheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def mechanism_add_const_cp(self,
        str spec_name,
        double Tmin,
        double Tmax,
        double T0,
        double h0,
        double s0,
        double cp):
      self.c_calculator.mechanism_add_const_cp(spec_name.encode(), Tmin, Tmax, T0, h0, s0, cp)

    @cython.nonecheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def mechanism_add_nasa7_cp(self,
        str spec_name,
        double Tmin,
        double Tmid,
        double Tmax,
        list low_coeffs,
        list high_coeffs):
      cdef vector[double] low_coeffs_vec
      cdef vector[double] high_coeffs_vec
      for l in low_coeffs:
        low_coeffs_vec.push_back(l)
      for h in high_coeffs:
        high_coeffs_vec.push_back(h)
      self.c_calculator.mechanism_add_nasa7_cp(spec_name.encode(), Tmin, Tmid, Tmax, low_coeffs_vec, high_coeffs_vec)

    @cython.nonecheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def mechanism_add_nasa9_cp(self,
        str spec_name,
        double Tmin,
        double Tmax,
        list coeffs):
      cdef vector[double] coeffs_vec
      for c in coeffs:
        coeffs_vec.push_back(c)
      self.c_calculator.mechanism_add_nasa9_cp(spec_name.encode(), Tmin, Tmax, coeffs_vec)

    @cython.nonecheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def mechanism_add_reaction_simple(self,
        dict reactants_stoich,
        dict products_stoich,
        bool reversible,
        double fwd_pre_exp_value,
        double fwd_temp_exponent,
        double fwd_act_energy):
      cdef map[string, int] reac_map
      cdef map[string, int] prod_map
      for spec_name in reactants_stoich:
        reac_map[spec_name.encode()] = reactants_stoich[spec_name]
      for spec_name in products_stoich:
        prod_map[spec_name.encode()] = products_stoich[spec_name]
      self.c_calculator.mechanism_add_reaction_simple(reac_map, prod_map, reversible,
        fwd_pre_exp_value, fwd_temp_exponent, fwd_act_energy)

    @cython.nonecheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def mechanism_add_reaction_three_body(self,
        dict reactants_stoich,
        dict products_stoich,
        bool reversible,
        double fwd_pre_exp_value,
        double fwd_temp_exponent,
        double fwd_act_energy,
        dict three_body_efficiencies,
        double default_efficiency):
      cdef map[string, int] reac_map
      cdef map[string, int] prod_map
      cdef map[string, double] three_body_map
      for spec_name in reactants_stoich:
        reac_map[spec_name.encode()] = reactants_stoich[spec_name]
      for spec_name in products_stoich:
        prod_map[spec_name.encode()] = products_stoich[spec_name]
      for spec_name in three_body_efficiencies:
        three_body_map[spec_name.encode()] = three_body_efficiencies[spec_name]
      self.c_calculator.mechanism_add_reaction_three_body(reac_map, prod_map, reversible,
      	fwd_pre_exp_value, fwd_temp_exponent, fwd_act_energy, three_body_map, default_efficiency)

    @cython.nonecheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def mechanism_add_reaction_Lindemann(self,
        dict reactants_stoich,
        dict products_stoich,
        bool reversible,
        double fwd_pre_exp_value,
        double fwd_temp_exponent,
        double fwd_act_energy,
        dict three_body_efficiencies,
        double default_efficiency,
        double flf_pre_exp_value,
        double flf_temp_exponent,
        double flf_act_energy):
      cdef map[string, int] reac_map
      cdef map[string, int] prod_map
      cdef map[string, double] three_body_map
      for spec_name in reactants_stoich:
        reac_map[spec_name.encode()] = reactants_stoich[spec_name]
      for spec_name in products_stoich:
        prod_map[spec_name.encode()] = products_stoich[spec_name]
      for spec_name in three_body_efficiencies:
        three_body_map[spec_name.encode()] = three_body_efficiencies[spec_name]
      self.c_calculator.mechanism_add_reaction_Lindemann(reac_map, prod_map, reversible,
      	fwd_pre_exp_value, fwd_temp_exponent, fwd_act_energy, three_body_map, default_efficiency,
      	flf_pre_exp_value, flf_temp_exponent, flf_act_energy)

    @cython.nonecheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def mechanism_add_reaction_Troe(self,
        dict reactants_stoich,
        dict products_stoich,
        bool reversible,
        double fwd_pre_exp_value,
        double fwd_temp_exponent,
        double fwd_act_energy,
        dict three_body_efficiencies,
        double default_efficiency,
        double flf_pre_exp_value,
        double flf_temp_exponent,
        double flf_act_energy,
        list troe_parameters):
      cdef vector[double] troe_parameters_vec
      troe_parameters_vec.push_back(0.)
      troe_parameters_vec.push_back(0.)
      troe_parameters_vec.push_back(0.)
      troe_parameters_vec.push_back(0.)
      for i, l in enumerate(troe_parameters):
        troe_parameters_vec[i] = l
      cdef map[string, int] reac_map
      cdef map[string, int] prod_map
      cdef map[string, double] three_body_map
      for spec_name in reactants_stoich:
        reac_map[spec_name.encode()] = reactants_stoich[spec_name]
      for spec_name in products_stoich:
        prod_map[spec_name.encode()] = products_stoich[spec_name]
      for spec_name in three_body_efficiencies:
        three_body_map[spec_name.encode()] = three_body_efficiencies[spec_name]
      self.c_calculator.mechanism_add_reaction_Troe(reac_map, prod_map, reversible,
      	fwd_pre_exp_value, fwd_temp_exponent, fwd_act_energy, three_body_map, default_efficiency,
      	flf_pre_exp_value, flf_temp_exponent, flf_act_energy, troe_parameters_vec)

    @cython.nonecheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def mechanism_add_reaction_simple_with_special_orders(self,
        dict reactants_stoich,
        dict products_stoich,
        bool reversible,
        double fwd_pre_exp_value,
        double fwd_temp_exponent,
        double fwd_act_energy,
        dict special_orders):
      cdef map[string, int] reac_map
      cdef map[string, int] prod_map
      cdef map[string, double] ord_map
      for spec_name in reactants_stoich:
        reac_map[spec_name.encode()] = reactants_stoich[spec_name]
      for spec_name in products_stoich:
        prod_map[spec_name.encode()] = products_stoich[spec_name]
      for spec_name in special_orders:
        ord_map[spec_name.encode()] = special_orders[spec_name]
      self.c_calculator.mechanism_add_reaction_simple_with_special_orders(reac_map, prod_map, reversible,
      	fwd_pre_exp_value, fwd_temp_exponent, fwd_act_energy, ord_map)

    @cython.nonecheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def mechanism_add_reaction_three_body_with_special_orders(self,
        dict reactants_stoich,
        dict products_stoich,
        bool reversible,
        double fwd_pre_exp_value,
        double fwd_temp_exponent,
        double fwd_act_energy,
        dict three_body_efficiencies,
        double default_efficiency,
        dict special_orders):
      cdef map[string, int] reac_map
      cdef map[string, int] prod_map
      cdef map[string, double] ord_map
      cdef map[string, double] three_body_map
      for spec_name in reactants_stoich:
        reac_map[spec_name.encode()] = reactants_stoich[spec_name]
      for spec_name in products_stoich:
        prod_map[spec_name.encode()] = products_stoich[spec_name]
      for spec_name in special_orders:
        ord_map[spec_name.encode()] = special_orders[spec_name]
      for spec_name in three_body_efficiencies:
        three_body_map[spec_name.encode()] = three_body_efficiencies[spec_name]
      self.c_calculator.mechanism_add_reaction_three_body_with_special_orders(reac_map, prod_map, reversible,
      	fwd_pre_exp_value, fwd_temp_exponent, fwd_act_energy, three_body_map, default_efficiency, ord_map)

    @cython.nonecheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def mechanism_add_reaction_Lindemann_with_special_orders(self,
        dict reactants_stoich,
        dict products_stoich,
        bool reversible,
        double fwd_pre_exp_value,
        double fwd_temp_exponent,
        double fwd_act_energy,
        dict three_body_efficiencies,
        double default_efficiency,
        double flf_pre_exp_value,
        double flf_temp_exponent,
        double flf_act_energy,
        dict special_orders):
      cdef map[string, int] reac_map
      cdef map[string, int] prod_map
      cdef map[string, double] ord_map
      cdef map[string, double] three_body_map
      for spec_name in reactants_stoich:
        reac_map[spec_name.encode()] = reactants_stoich[spec_name]
      for spec_name in products_stoich:
        prod_map[spec_name.encode()] = products_stoich[spec_name]
      for spec_name in special_orders:
        ord_map[spec_name.encode()] = special_orders[spec_name]
      for spec_name in three_body_efficiencies:
        three_body_map[spec_name.encode()] = three_body_efficiencies[spec_name]
      self.c_calculator.mechanism_add_reaction_Lindemann_with_special_orders(reac_map, prod_map, reversible,
      	fwd_pre_exp_value, fwd_temp_exponent, fwd_act_energy, three_body_map, default_efficiency,
      	flf_pre_exp_value, flf_temp_exponent, flf_act_energy, ord_map)

    @cython.nonecheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def mechanism_add_reaction_Troe_with_special_orders(self,
        dict reactants_stoich,
        dict products_stoich,
        bool reversible,
        double fwd_pre_exp_value,
        double fwd_temp_exponent,
        double fwd_act_energy,
        dict three_body_efficiencies,
        double default_efficiency,
        double flf_pre_exp_value,
        double flf_temp_exponent,
        double flf_act_energy,
        list troe_parameters,
        dict special_orders):
      cdef vector[double] troe_parameters_vec
      troe_parameters_vec.push_back(0.)
      troe_parameters_vec.push_back(0.)
      troe_parameters_vec.push_back(0.)
      troe_parameters_vec.push_back(0.)
      for i, l in enumerate(troe_parameters):
        troe_parameters_vec[i] = l
      cdef map[string, int] reac_map
      cdef map[string, int] prod_map
      cdef map[string, double] ord_map
      cdef map[string, double] three_body_map
      for spec_name in reactants_stoich:
        reac_map[spec_name.encode()] = reactants_stoich[spec_name]
      for spec_name in products_stoich:
        prod_map[spec_name.encode()] = products_stoich[spec_name]
      for spec_name in special_orders:
        ord_map[spec_name.encode()] = special_orders[spec_name]
      for spec_name in three_body_efficiencies:
        three_body_map[spec_name.encode()] = three_body_efficiencies[spec_name]
      self.c_calculator.mechanism_add_reaction_Troe_with_special_orders(reac_map, prod_map, reversible,
      	fwd_pre_exp_value, fwd_temp_exponent, fwd_act_energy, three_body_map, default_efficiency,
      	flf_pre_exp_value, flf_temp_exponent, flf_act_energy, troe_parameters_vec, ord_map)

    @cython.nonecheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def flamelet_stencils(self,
        np.ndarray[np.double_t, ndim=1] dz,
        int nzi,
        np.ndarray[np.double_t, ndim=1] dissipationRate,
        np.ndarray[np.double_t, ndim=1] invLewisNumbers,
        np.ndarray[np.double_t, ndim=1] out_cmajor,
        np.ndarray[np.double_t, ndim=1] out_csub,
        np.ndarray[np.double_t, ndim=1] out_csup,
        np.ndarray[np.double_t, ndim=1] out_mcoeff,
        np.ndarray[np.double_t, ndim=1] out_ncoeff):
      self.c_calculator.flamelet_stencils(&dz[0], nzi, &dissipationRate[0], &invLewisNumbers[0], &out_cmajor[0], &out_csub[0], &out_csup[0], &out_mcoeff[0], &out_ncoeff[0])

    @cython.nonecheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def flamelet_jac_indices(self,
        int nzi,
        np.ndarray[int, ndim=1] out_row_indices,
        np.ndarray[int, ndim=1] out_col_indices):
      self.c_calculator.flamelet_jac_indices(nzi, &out_row_indices[0], &out_col_indices[0])

    @cython.nonecheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def flamelet_rhs(self,
        np.ndarray[np.double_t, ndim=1] state,
        np.float_t pressure,
        np.ndarray[np.double_t, ndim=1] oxyState,
        np.ndarray[np.double_t, ndim=1] fuelState,
        bool adiabatic,
        np.ndarray[np.double_t, ndim=1] T_convection,
        np.ndarray[np.double_t, ndim=1] T_radiation,
        np.ndarray[np.double_t, ndim=1] h_convection,
        np.ndarray[np.double_t, ndim=1] h_radiation,
        int nzi,
        np.ndarray[np.double_t, ndim=1] cmajor,
        np.ndarray[np.double_t, ndim=1] csub,
        np.ndarray[np.double_t, ndim=1] csup,
        np.ndarray[np.double_t, ndim=1] mcoeff,
        np.ndarray[np.double_t, ndim=1] ncoeff,
        np.ndarray[np.double_t, ndim=1] chi,
        bool include_enthalpy_flux,
        bool include_variable_cp,
        bool use_scaled_heat_loss,
        np.ndarray[np.double_t, ndim=1] out_rhs):
      self.c_calculator.flamelet_rhs(&state[0],
      pressure,
      &oxyState[0],
      &fuelState[0],
      adiabatic,
      &T_convection[0],
      &h_convection[0],
      &T_radiation[0],
      &h_radiation[0],
      nzi,
      &cmajor[0],
      &csub[0],
      &csup[0],
      &mcoeff[0],
      &ncoeff[0],
      &chi[0],
      include_enthalpy_flux,
      include_variable_cp,
      use_scaled_heat_loss,
      &out_rhs[0])

    @cython.nonecheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def flamelet_jacobian(self,
        np.ndarray[np.double_t, ndim=1] state,
        np.float_t pressure,
        np.ndarray[np.double_t, ndim=1] oxyState,
        np.ndarray[np.double_t, ndim=1] fuelState,
        bool adiabatic,
        np.ndarray[np.double_t, ndim=1] T_convection,
        np.ndarray[np.double_t, ndim=1] T_radiation,
        np.ndarray[np.double_t, ndim=1] h_convection,
        np.ndarray[np.double_t, ndim=1] h_radiation,
        int nzi,
        np.ndarray[np.double_t, ndim=1] cmajor,
        np.ndarray[np.double_t, ndim=1] csub,
        np.ndarray[np.double_t, ndim=1] csup,
        np.ndarray[np.double_t, ndim=1] mcoeff,
        np.ndarray[np.double_t, ndim=1] ncoeff,
        np.ndarray[np.double_t, ndim=1] chi,
        bool compute_eigenvalues,
        np.float_t diffterm,
        bool scale_and_offset,
        np.float_t prefactor,
        int rates_sensitivity_option,
        int sensitivity_transform_option,
        bool include_enthalpy_flux,
        bool include_variable_cp,
        bool use_scaled_heat_loss,
        np.ndarray[np.double_t, ndim=1] out_expeig,
        np.ndarray[np.double_t, ndim=1] out_jac):
      self.c_calculator.flamelet_jacobian(&state[0],
      pressure,
      &oxyState[0],
      &fuelState[0],
      adiabatic,
      &T_convection[0],
      &h_convection[0],
      &T_radiation[0],
      &h_radiation[0],
      nzi,
      &cmajor[0],
      &csub[0],
      &csup[0],
      &mcoeff[0],
      &ncoeff[0],
      &chi[0],
      compute_eigenvalues,
      diffterm,
      scale_and_offset,
      prefactor,
      rates_sensitivity_option,
      sensitivity_transform_option,
      include_enthalpy_flux,
      include_variable_cp,
      use_scaled_heat_loss,
      &out_expeig[0],
      &out_jac[0])

    @cython.nonecheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def mixture_molecular_weight(self, np.ndarray[np.double_t, ndim=1] y):
      return self.c_calculator.mixture_molecular_weight(&y[0])

    @cython.nonecheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def mole_fractions(self, np.ndarray[np.double_t, ndim=1] y, np.ndarray[np.double_t, ndim=1] x):
      return self.c_calculator.mole_fractions(&y[0], &x[0])

    @cython.nonecheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def ideal_gas_density(self, np.double_t p, np.double_t T, np.ndarray[np.double_t, ndim=1] y):
      return self.c_calculator.ideal_gas_density(p, T, &y[0])

    @cython.nonecheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def ideal_gas_pressure(self, np.double_t rho, np.double_t T, np.ndarray[np.double_t, ndim=1] y):
      return self.c_calculator.ideal_gas_pressure(rho, T, &y[0])

    @cython.nonecheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def cp_mix(self, np.double_t T, np.ndarray[np.double_t, ndim=1] y):
      return self.c_calculator.cp_mix(T, &y[0])

    @cython.nonecheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def cv_mix(self, np.double_t T, np.ndarray[np.double_t, ndim=1] y):
      return self.c_calculator.cv_mix(T, &y[0])

    @cython.nonecheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def species_cp(self, np.double_t T, np.ndarray[np.double_t, ndim=1] cpspecies):
      self.c_calculator.species_cp(T, &cpspecies[0])

    @cython.nonecheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def species_cv(self, np.double_t T, np.ndarray[np.double_t, ndim=1] cvspecies):
      self.c_calculator.species_cv(T, &cvspecies[0])

    @cython.nonecheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def enthalpy_mix(self, np.double_t T, np.ndarray[np.double_t, ndim=1] y):
      return self.c_calculator.enthalpy_mix(T, &y[0])

    @cython.nonecheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def energy_mix(self, np.double_t T, np.ndarray[np.double_t, ndim=1] y):
      return self.c_calculator.energy_mix(T, &y[0])

    @cython.nonecheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def species_enthalpies(self, np.double_t T, np.ndarray[np.double_t, ndim=1] enthalpyspecies):
      self.c_calculator.species_enthalpies(T, &enthalpyspecies[0])

    @cython.nonecheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def species_energies(self, np.double_t T, np.ndarray[np.double_t, ndim=1] energyspecies):
      self.c_calculator.species_energies(T, &energyspecies[0])

    @cython.nonecheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def dcpdT_species(self, np.double_t T, np.ndarray[np.double_t, ndim=1] mass_fractions, np.ndarray[np.double_t, ndim=1] dcpdT_spec):
      dcpdT = 0.0
      self.c_calculator.cp_sens_T(T, &mass_fractions[0], &dcpdT, &dcpdT_spec[0])

    @cython.nonecheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def production_rates(self,
        np.double_t density,
        np.double_t temperature,
        np.ndarray[np.double_t, ndim=1] mass_fractions,
        np.ndarray[np.double_t, ndim=1] prodrates):
      self.c_calculator.production_rates(density, temperature, &mass_fractions[0], &prodrates[0])

    @cython.nonecheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def prod_rates_primitive_sensitivities(self,
        np.double_t density,
        np.double_t temperature,
        np.ndarray[np.double_t, ndim=1] mass_fractions,
        int rates_sensitivity_option,
        np.ndarray[np.double_t, ndim=1] prod_rates_sens):
      self.c_calculator.prod_rates_primitive_sensitivities(density,
                                                           temperature,
                                                           &mass_fractions[0],
                                                           rates_sensitivity_option,
                                                           &prod_rates_sens[0])

    @cython.nonecheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def reactor_rhs_isobaric(self,
        np.ndarray[np.double_t, ndim=1] state,
        np.double_t pressure,
        np.double_t inflowTemperature,
        np.ndarray[np.double_t, ndim=1] inflowY,
        np.double_t tau,
        np.double_t fluidTemperature,
        np.double_t surfTemperature,
        np.double_t hConv,
        np.double_t epsRad,
        np.double_t surfaceAreaOverVolume,
        int heatTransferOption,
        bool open,
        np.ndarray[np.double_t, ndim=1] rightHandSide):
      self.c_calculator.reactor_rhs_isobaric(&state[0], pressure, inflowTemperature, &inflowY[0], tau, fluidTemperature, surfTemperature, hConv, epsRad, surfaceAreaOverVolume, heatTransferOption, open, &rightHandSide[0])

    @cython.nonecheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def reactor_jac_isobaric(self,
        np.ndarray[np.double_t, ndim=1] state,
        np.double_t pressure,
        np.double_t inflowTemperature,
        np.ndarray[np.double_t, ndim=1] inflowY,
        np.double_t tau,
        np.double_t fluidTemperature,
        np.double_t surfTemperature,
        np.double_t hConv,
        np.double_t epsRad,
        np.double_t surfaceAreaOverVolume,
        int heatTransferOption,
        bool open,
        int rates_sensitivity_option,
        int sensitivity_transform_option,
        np.ndarray[np.double_t, ndim=1] rightHandSide,
        np.ndarray[np.double_t, ndim=1] jacobian):
      self.c_calculator.reactor_jac_isobaric(&state[0], pressure, inflowTemperature, &inflowY[0], tau, fluidTemperature, surfTemperature, hConv, epsRad, surfaceAreaOverVolume, heatTransferOption, open, rates_sensitivity_option, sensitivity_transform_option, &rightHandSide[0], &jacobian[0])



    @cython.nonecheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def reactor_rhs_isochoric(self,
        np.ndarray[ np.double_t, ndim=1] state,
        np.double_t inflowDensity,
        np.double_t inflowTemperature,
        np.ndarray[ np.double_t, ndim=1] inflowY,
        np.double_t tau,
        np.double_t fluidTemperature,
        np.double_t surfTemperature,
        np.double_t hConv,
        np.double_t epsRad,
        np.double_t surfaceAreaOverVolume,
        int heatTransferOption,
        bool open,
        np.ndarray[ np.double_t, ndim=1] rightHandSide):
      self.c_calculator.reactor_rhs_isochoric(&state[0], inflowDensity, inflowTemperature, &inflowY[0], tau, fluidTemperature, surfTemperature, hConv, epsRad, surfaceAreaOverVolume, heatTransferOption, open, &rightHandSide[0])

    @cython.nonecheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def reactor_jac_isochoric(self,
        np.ndarray[ np.double_t, ndim=1] state,
        np.double_t inflowDensity,
        np.double_t inflowTemperature,
        np.ndarray[ np.double_t, ndim=1] inflowY,
        np.double_t tau,
        np.double_t fluidTemperature,
        np.double_t surfTemperature,
        np.double_t hConv,
        np.double_t epsRad,
        np.double_t surfaceAreaOverVolume,
        int heatTransferOption,
        bool open,
        int rates_sensitivity_option,
        np.ndarray[ np.double_t, ndim=1] rightHandSide,
        np.ndarray[ np.double_t, ndim=1] jacobian):
      self.c_calculator.reactor_jac_isochoric(&state[0], inflowDensity, inflowTemperature, &inflowY[0], tau, fluidTemperature, surfTemperature, hConv, epsRad, surfaceAreaOverVolume, heatTransferOption, open, rates_sensitivity_option, &rightHandSide[0], &jacobian[0])

    @cython.nonecheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def flamelet2d_rhs(self,
                       np.ndarray[np.double_t, ndim=1] state,
                       np.float_t pressure,
                       int nx,
                       int ny,
                       np.ndarray[np.double_t, ndim=1] xcp,
                       np.ndarray[np.double_t, ndim=1] xcl,
                       np.ndarray[np.double_t, ndim=1] xcr,
                       np.ndarray[np.double_t, ndim=1] ycp,
                       np.ndarray[np.double_t, ndim=1] ycb,
                       np.ndarray[np.double_t, ndim=1] yct,
                       np.ndarray[np.double_t, ndim=1] out_rhs):
        self.c_calculator.flamelet2d_rhs(&state[0],
                                         pressure,
                                         nx,
                                         ny,
                                         &xcp[0],
                                         &xcl[0],
                                         &xcr[0],
                                         &ycp[0],
                                         &ycb[0],
                                         &yct[0],
                                         &out_rhs[0])
    @cython.nonecheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def flamelet2d_factored_block_diag_jacobian(self,
                       np.ndarray[np.double_t, ndim=1] state,
                       np.float_t pressure,
                       int nx,
                       int ny,
                       np.ndarray[np.double_t, ndim=1] xcp,
                       np.ndarray[np.double_t, ndim=1] ycp,
                       np.float_t prefactor,
                       np.ndarray[np.double_t, ndim=1] out_values,
                       np.ndarray[np.double_t, ndim=1] out_factors,
                       np.ndarray[int, ndim=1] out_pivots):
        self.c_calculator.flamelet2d_factored_block_diag_jacobian(&state[0],
                                                                  pressure,
                                                                  nx,
                                                                  ny,
                                                                  &xcp[0],
                                                                  &ycp[0],
                                                                  prefactor,
                                                                  &out_values[0],
                                                                  &out_factors[0],
                                                                  &out_pivots[0])
    @cython.nonecheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def flamelet2d_offdiag_matvec(self,
                                  np.ndarray[np.double_t, ndim=1] vec,
                                  int nx,
                                  int ny,
                                  np.ndarray[np.double_t, ndim=1] xcp,
                                  np.ndarray[np.double_t, ndim=1] xcl,
                                  np.ndarray[np.double_t, ndim=1] xcr,
                                  np.ndarray[np.double_t, ndim=1] ycp,
                                  np.ndarray[np.double_t, ndim=1] ycb,
                                  np.ndarray[np.double_t, ndim=1] yct,
                                  np.float_t prefactor,
                                  np.ndarray[np.double_t, ndim=1] out_vec):
        self.c_calculator.flamelet2d_offdiag_matvec(&vec[0],
                                                    nx,
                                                    ny,
                                                    &xcp[0],
                                                    &xcl[0],
                                                    &xcr[0],
                                                    &ycp[0],
                                                    &ycb[0],
                                                    &yct[0],
                                                    prefactor,
                                                    &out_vec[0])
    @cython.nonecheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def flamelet2d_matvec(self,
                          np.ndarray[np.double_t, ndim=1] vec,
                          int nx,
                          int ny,
                          np.ndarray[np.double_t, ndim=1] xcp,
                          np.ndarray[np.double_t, ndim=1] xcl,
                          np.ndarray[np.double_t, ndim=1] xcr,
                          np.ndarray[np.double_t, ndim=1] ycp,
                          np.ndarray[np.double_t, ndim=1] ycb,
                          np.ndarray[np.double_t, ndim=1] yct,
                          np.float_t prefactor,
                          np.ndarray[np.double_t, ndim=1] block_diag_values,
                          np.ndarray[np.double_t, ndim=1] out_vec):
        self.c_calculator.flamelet2d_matvec(&vec[0],
                                            nx,
                                            ny,
                                            &xcp[0],
                                            &xcl[0],
                                            &xcr[0],
                                            &ycp[0],
                                            &ycb[0],
                                            &yct[0],
                                            prefactor,
                                            &block_diag_values[0],
                                            &out_vec[0])
    @cython.nonecheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def flamelet2d_block_diag_solve(self,
                                    int nx,
                                    int ny,
                                    np.ndarray[np.double_t, ndim=1] factors,
                                    np.ndarray[int, ndim=1] pivots,
                                    np.ndarray[np.double_t, ndim=1] b,
                                    np.ndarray[np.double_t, ndim=1] out_x):
        self.c_calculator.flamelet2d_block_diag_solve(nx,
                                                      ny,
                                                      &factors[0],
                                                      &pivots[0],
                                                      &b[0],
                                                      &out_x[0])



# block-tridiagonal scalar-off-diagonal (BTDSOD) matrix operations

cdef extern from "btddod_matrix_kernels.h" namespace "griffon::btddod":
  void btddod_full_matvec(const double *matrix_values, const double *vec, const int num_blocks, const int block_size, double *out_matvec)
  void btddod_blockdiag_matvec(const double *matrix_values, const double *vec, const int num_blocks, const int block_size, double *out_matvec)
  void btddod_blockdiag_factorize(const double *matrix_values, const int num_blocks, const int block_size, int *out_pivots, double *out_factors)
  void btddod_blockdiag_solve(const int *pivots, const double *factors, const double *rhs, const int num_blocks, const int block_size, double *out_solution)
  void btddod_lowerfulltriangle_solve(const int *pivots, const double *factors, const double *matrix_values, const double *rhs, const int num_blocks, const int block_size, double *out_solution)
  void btddod_upperfulltriangle_solve(const int *pivots, const double *factors, const double *matrix_values, const double *rhs, const int num_blocks, const int block_size, double *out_solution)
  void btddod_full_factorize( double *out_dfactors, const int num_blocks, const int block_size, double *out_l_values, int *out_d_pivots )
  void btddod_full_solve( const double *d_factors, const double *l_values, const int *d_pivots, const double *rhs, const int num_blocks, const int block_size, double *out_solution )
  void btddod_scale_and_add_scaled_block_diagonal( double *in_out_matrix_values, const double matrix_scale, const double *block_diag, const double diag_scale, const int num_blocks, const int block_size )
  void btddod_scale_and_add_diagonal( double *in_out_matrix_values, const double matrix_scale, const double *diagonal, const double diag_scale, const int num_blocks, const int block_size );


@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
def py_btddod_full_matvec(np.ndarray[np.double_t, ndim=1] matrix_values,
    np.ndarray[np.double_t, ndim=1] vec,
    int num_blocks,
    int block_size,
    np.ndarray[np.double_t, ndim=1] out_matvec):
  btddod_full_matvec(&matrix_values[0], &vec[0], num_blocks, block_size, &out_matvec[0])

@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
def py_btddod_blockdiag_matvec(np.ndarray[np.double_t, ndim=1] matrix_values,
    np.ndarray[np.double_t, ndim=1] vec,
    int num_blocks,
    int block_size,
    np.ndarray[np.double_t, ndim=1] out_matvec):
  btddod_blockdiag_matvec(&matrix_values[0], &vec[0], num_blocks, block_size, &out_matvec[0])

@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
def py_btddod_blockdiag_factorize(np.ndarray[np.double_t, ndim=1] matrix_values,
    int num_blocks,
    int block_size,
    np.ndarray[int, ndim=1] out_pivots,
    np.ndarray[np.double_t, ndim=1] out_factors):
  btddod_blockdiag_factorize(&matrix_values[0], num_blocks, block_size, &out_pivots[0], &out_factors[0])

@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
def py_btddod_blockdiag_solve(np.ndarray[int, ndim=1] pivots,
    np.ndarray[np.double_t, ndim=1] factors,
    np.ndarray[np.double_t, ndim=1] rhs,
    int num_blocks,
    int block_size,
    np.ndarray[np.double_t, ndim=1] out_solution):
  btddod_blockdiag_solve(&pivots[0], &factors[0], &rhs[0], num_blocks, block_size, &out_solution[0])

@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
def py_btddod_lowerfulltriangle_solve(np.ndarray[int, ndim=1] pivots,
    np.ndarray[np.double_t, ndim=1] factors,
    np.ndarray[np.double_t, ndim=1] matrix_values,
    np.ndarray[np.double_t, ndim=1] rhs,
    int num_blocks,
    int block_size,
    np.ndarray[np.double_t, ndim=1] out_solution):
  btddod_lowerfulltriangle_solve(&pivots[0], &factors[0], &matrix_values[0], &rhs[0], num_blocks, block_size, &out_solution[0])

@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
def py_btddod_upperfulltriangle_solve(np.ndarray[int, ndim=1] pivots,
    np.ndarray[np.double_t, ndim=1] factors,
    np.ndarray[np.double_t, ndim=1] matrix_values,
    np.ndarray[np.double_t, ndim=1] rhs,
    int num_blocks,
    int block_size,
    np.ndarray[np.double_t, ndim=1] out_solution):
  btddod_upperfulltriangle_solve(&pivots[0], &factors[0], &matrix_values[0], &rhs[0], num_blocks, block_size, &out_solution[0])

@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
def py_btddod_full_factorize(np.ndarray[np.double_t, ndim=1] out_d_factors,
    int num_blocks,
    int block_size,
    np.ndarray[np.double_t, ndim=1] out_l_values,
    np.ndarray[int, ndim=1] out_d_pivots):
  btddod_full_factorize(&out_d_factors[0], num_blocks, block_size, &out_l_values[0], &out_d_pivots[0])

@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
def py_btddod_full_solve(np.ndarray[np.double_t, ndim=1] d_factors,
    np.ndarray[np.double_t, ndim=1] l_values,
    np.ndarray[int, ndim=1] d_pivots,
    np.ndarray[np.double_t, ndim=1] rhs,
    int num_blocks,
    int block_size,
    np.ndarray[np.double_t, ndim=1] out_solution):
  btddod_full_solve(&d_factors[0], &l_values[0], &d_pivots[0], &rhs[0], num_blocks, block_size, &out_solution[0])

@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
def py_btddod_scale_and_add_scaled_block_diagonal(np.ndarray[np.double_t, ndim=1] in_out_matrix_values,
    np.double_t matrix_scale,
    np.ndarray[np.double_t, ndim=1] block_diag,
    np.double_t diag_scale,
    int num_blocks,
    int block_size):
  btddod_scale_and_add_scaled_block_diagonal(&in_out_matrix_values[0], matrix_scale, &block_diag[0], diag_scale, num_blocks, block_size)

@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
def py_btddod_scale_and_add_diagonal(np.ndarray[np.double_t, ndim=1] in_out_matrix_values,
    np.double_t matrix_scale,
    np.ndarray[np.double_t, ndim=1] diagonal,
    np.double_t diag_scale,
    int num_blocks,
    int block_size):
  btddod_scale_and_add_diagonal(&in_out_matrix_values[0], matrix_scale, &diagonal[0], diag_scale, num_blocks, block_size)
