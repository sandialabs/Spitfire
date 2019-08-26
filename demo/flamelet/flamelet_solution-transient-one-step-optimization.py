from spitfire.chemistry.mechanism import ChemicalMechanismSpec as Mechanism
from spitfire.chemistry.flamelet import Flamelet
from spitfire.chemistry.tabulation import build_adiabatic_slfm_library
import numpy as np
import cantera as ct
import matplotlib.pyplot as plt


def make_one_step_mechanism(pre_exp_factor=5.1e11, activation_energy=30.0, ord_fuel=0.25, ord_oxy=1.5):
    species_data = ct.Species.listFromFile('mechanisms/heptane-liu-hewson-chen-pitsch-highT.xml')
    species_in_model = ['NXC7H16', 'O2', 'H2O', 'CO2', 'N2']
    species_list = list()
    for sp in species_data:
        if sp.name in species_in_model:
            species_list.append(sp)
    reaction_list = [ct.Reaction.fromCti(
        f'''
    reaction(
    '2 NXC7H16 + 22 O2 => 14 CO2 + 16 H2O', 
    [{pre_exp_factor}, 0, ({activation_energy}, 'kcal/mol')], 
    order='NXC7H16:{ord_fuel} O2:{ord_oxy}')
        ''')]
    s = ct.Solution(thermo='IdealGas',
                    kinetics='GasKinetics',
                    species=species_list,
                    reactions=reaction_list)
    return Mechanism.from_solution(s)


def compute_flamelet_ign_delay(m):
    pressure = 101325.
    air = m.stream(stp_air=True)
    air.TP = 1200., pressure
    fuel = m.stream('X', f'NXC7H16:1')
    fuel.TP = 300., pressure

    chi_max = 1.e1

    ft = Flamelet(mech_spec=m,
                  pressure=pressure,
                  oxy_stream=air,
                  fuel_stream=fuel,
                  max_dissipation_rate=chi_max,
                  grid_points=128,
                  grid_cluster_intensity=6.,
                  include_variable_cp=True,
                  include_enthalpy_flux=True,
                  initial_condition='unreacted')

    ft.insitu_process_quantity('temperature')
    tau_ign = ft.compute_ignition_delay(delta_temperature_ignition=600., first_time_step=1.e-6)

    t = ft.solution_times
    z = ft.mixfrac_grid
    temperature = ft.trajectory_data('temperature')

    return tau_ign, t, z, temperature


detailed_mech = Mechanism('mechanisms/heptane-liu-hewson-chen-pitsch-highT.xml', 'gas')
reduced_mech1 = make_one_step_mechanism(pre_exp_factor=2.e7, activation_energy=30.)
reduced_mech2 = make_one_step_mechanism(pre_exp_factor=1.2e9, activation_energy=40.)


tau_r, t_r, z_r, T_r = compute_flamelet_ign_delay(reduced_mech1)
print('one-step 1 complete')
tau_r2, t_r2, z_r2, T_r2 = compute_flamelet_ign_delay(reduced_mech2)
print('one-step 2 complete')
tau_d, t_d, z_d, T_d = compute_flamelet_ign_delay(detailed_mech)
print('detailed complete')

print(f'Ignition delay (ms)\n--------------------\ndetailed = {tau_d * 1e3:.1f},\n'
      f'one-step (A=2e7, Ea=30 kcal/mol) = {tau_r * 1e3:.1f},\n'
      f'one-step (A=1.2e9, Ea=40 kcal/mol) = {tau_r2 * 1e3:.1f}\n')

for it in range(0, t_d.size, 10):
    plt.plot(z_d, T_d[it, :], 'b:')
for it in range(0, t_r.size, 6):
    plt.plot(z_r, T_r[it, :], 'r:')
for it in range(0, t_r2.size, 6):
    plt.plot(z_r2, T_r2[it, :], 'g:')
plt.plot(z_d, T_d[-1, :], 'b-', label='detailed')
plt.plot(z_r, T_r[-1, :], 'r-', label='one-step (A=2e7, Ea=30 kcal/mol)')
plt.plot(z_r2, T_r2[-1, :], 'g-', label='one-step (A=1.2e9, Ea=40 kcal/mol)')
plt.grid()
plt.legend(loc='best')
plt.ylabel('T (K)')
plt.xlabel('Z')
plt.xlim([0, 0.1])
plt.ylim([800, 1800])
plt.show()


# def build_library(m, fuel_name):
#     pressure = 101325.
#     air = m.stream(stp_air=True)
#     air.TP = 1200., pressure
#     fuel = m.stream('X', f'{fuel_name}:1')
#     fuel.TP = 300., pressure
#
#     flamelet_specs = \
#         {
#             'mech_spec': m,
#             'pressure': pressure,
#             'oxy_stream': air,
#             'fuel_stream': fuel,
#             'grid_points': 128,
#             'grid_cluster_intensity': 10.,
#             'grid_cluster_point': 0.1,
#             'include_enthalpy_flux': True,
#             'include_variable_cp': True,
#             'initial_condition': 'equilibrium'
#         }
#
#     return build_adiabatic_slfm_library(flamelet_specs=flamelet_specs,
#                                         tabulated_quantities=['temperature'],
#                                         diss_rate_values=np.logspace(-3, 2, 32),
#                                         diss_rate_ref='stoichiometric')
#
#
# lib_r = build_library(reduced_mech1, 'C7H16')
# lib_r2 = build_library(reduced_mech2, 'C7H16')
#
# plt.semilogx(lib_r.dissipation_rate_values, np.max(lib_r['temperature'], axis=0))
# plt.semilogx(lib_r2.dissipation_rate_values, np.max(lib_r2['temperature'], axis=0))
# plt.show()
