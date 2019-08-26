from spitfire.chemistry.mechanism import ChemicalMechanismSpec as Mechanism
from spitfire.chemistry.tabulation import build_adiabatic_slfm_library
import cantera as ct
import numpy as np
import matplotlib.pyplot as plt


def make_one_step_mechanism(pre_exp_factor=5.1e11, activation_energy=30.0, ord_fuel=0.25, ord_oxy=1.5):
    species_data = ct.Species.listFromFile('mechanisms/heptane-liu-hewson-chen-pitsch-highT.xml')
    species_in_model = ['NXC7H16', 'O2', 'H2O', 'CO2', 'CO', 'N2']
    species_list = list()
    for sp in species_data:
        if sp.name in species_in_model:
            species_list.append(sp)
    reaction_list = [ct.Reaction.fromCti(f'''
reaction(
'2 NXC7H16 + 22 O2 => 14 CO2 + 16 H2O',
[{pre_exp_factor}, 0, ({activation_energy}, 'kcal/mol')],
order='NXC7H16:{ord_fuel} O2:{ord_oxy}')''')]
    #     reaction_list = [
    #         ct.Reaction.fromCti(f'''
    # reaction(
    # '2 NXC7H16 + 15 O2 => 14 CO + 16 H2O',
    # [{pre_exp_factor}, 0, ({activation_energy}, 'kcal/mol')],
    # order='NXC7H16:{ord_fuel} O2:{ord_oxy}')'''),
    #         ct.Reaction.fromCti(f'''
    # reaction(
    # '2 CO + O2 => 2 CO2',
    # [3.98107170553497e14, 0, (40, 'kcal/mol')],
    # order='CO:1 O2:0.25 H2O:0.5',
    # options=['nonreactant_orders'])'''),
    #         ct.Reaction.fromCti(f'''
    # reaction(
    # '2 CO2 => 2 CO + O2',
    # [5e8, 0, (40, 'kcal/mol')],
    # order='CO2:1')'''),
    #     ]
    s = ct.Solution(thermo='IdealGas',
                    kinetics='GasKinetics',
                    species=species_list,
                    reactions=reaction_list)
    return Mechanism.from_solution(s)


# def make_two_step_mechanism(pre_exp_factor=5.1e11, activation_energy=30.0, ord_fuel=0.25, ord_oxy=1.5):
#     species_data = ct.Species.listFromFile('mechanisms/heptane-liu-hewson-chen-pitsch-highT.xml')
#     species_in_model = ['NXC7H16', 'O2', 'H2O', 'CO2', 'CO', 'N2']
#     species_list = list()
#     for sp in species_data:
#         if sp.name in species_in_model:
#             species_list.append(sp)
#     reaction_list = [
#         ct.Reaction.fromCti(f'''
#     reaction(
#     '2 NXC7H16 + 15 O2 => 14 CO + 16 H2O',
#     [{pre_exp_factor}, 0, ({activation_energy}, 'kcal/mol')],
#     order='NXC7H16:{ord_fuel} O2:{ord_oxy}')'''),
#         ct.Reaction.fromCti(f'''
#     reaction(
#     '2 CO + O2 => 2 CO2',
#     [3.98107170553497e14, 0, (40, 'kcal/mol')],
#     order='CO:1 O2:0.1',
#     options=['nonreactant_orders'])'''),  # order='CO:1 O2:0.25 H2O:0.5'
#         ct.Reaction.fromCti(f'''
#     reaction(
#     '2 CO2 => 2 CO + O2',
#     [5e8, 0, (40, 'kcal/mol')],
#     order='CO2:1')'''),
#     ]
#     s = ct.Solution(thermo='IdealGas',
#                     kinetics='GasKinetics',
#                     species=species_list,
#                     reactions=reaction_list)
#     return Mechanism.from_solution(s)


def build_library(m):
    pressure = 101325.

    air = m.stream(stp_air=True)
    air.TP = 300., pressure

    fuel = m.stream('TPX', (300., pressure, 'NXC7H16:1'))

    flamelet_specs = {'mech_spec': m,
                      'pressure': pressure,
                      'oxy_stream': air,
                      'fuel_stream': fuel,
                      'grid_points': 98,
                      'include_variable_cp': True,
                      'include_enthalpy_flux': True,
                      'initial_condition': 'equilibrium'}

    quantities = ['temperature', 'density']

    library = build_adiabatic_slfm_library(flamelet_specs,
                                           quantities,
                                           diss_rate_ref='stoichiometric',
                                           diss_rate_values=np.logspace(-3, 3, 60),
                                           verbose=True,
                                           solver_verbose=False)

    chi = library.dissipation_rate_stoich_grid
    z = library.mixture_fraction_grid

    return z, chi, library


z_r, chi_r, library_one_step = build_library(make_one_step_mechanism(ord_fuel=1, ord_oxy=1))
z_t, chi_t, library_two_step = build_library(make_one_step_mechanism(ord_fuel=1, ord_oxy=2))
# z_t, chi_t, library_two_step = build_library(make_two_step_mechanism(ord_fuel=1, ord_oxy=1))
z_d, chi_d, library_detailed = build_library(Mechanism('mechanisms/heptane-liu-hewson-chen-pitsch-highT.xml', 'gas'))

fig2, axarray2 = plt.subplots(1, 3)

levels = np.linspace(300., np.min([np.max(library_one_step['temperature']),
                                   np.max(library_two_step['temperature']),
                                   np.max(library_detailed['temperature'])]), 20)

axarray2[0].contourf(z_r, chi_r, library_one_step['temperature'], levels=levels)
axarray2[1].contourf(z_t, chi_t, library_two_step['temperature'], levels=levels)
axarray2[2].contourf(z_d, chi_d, library_detailed['temperature'], levels=levels)

for icol in range(3):
    axarray2[icol].set_yscale('log')
    axarray2[icol].set_xlabel('mixture fraction')
axarray2[0].set_ylabel('$\\chi_{{max}}$')
for icol in range(2):
    axarray2[1 + icol].set_ylabel('')
    axarray2[1 + icol].set_yticks([])
plt.show()
