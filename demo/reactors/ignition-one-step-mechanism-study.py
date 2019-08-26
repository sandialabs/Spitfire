from spitfire.chemistry.mechanism import ChemicalMechanismSpec as Mechanism
from spitfire.chemistry.reactors import HomogeneousReactor
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


def integrate_reactor(m):
    fuel = m.stream('X', 'NXC7H16:1')
    air = m.stream(stp_air=True)

    mix = m.mix_for_equivalence_ratio(phi=1., fuel=fuel, oxy=air)
    mix.TP = 1000, 101325

    r = HomogeneousReactor(m, mix,
                           configuration='isobaric',
                           heat_transfer='adiabatic',
                           mass_transfer='closed')
    r.insitu_process_quantity('temperature')
    r.integrate_to_time(0.15)
    return r.solution_times * 1.e3, r.trajectory_data('temperature')


reduced_mech1 = make_one_step_mechanism(pre_exp_factor=2e7)
reduced_mech2 = make_one_step_mechanism(pre_exp_factor=1.2e9, activation_energy=40.)
detailed_mech = Mechanism('mechanisms/heptane-liu-hewson-chen-pitsch-highT.xml', 'gas')

plt.plot(*integrate_reactor(reduced_mech1), label='one-step (A=2e7, Ea=30 kcal/mol)')
plt.plot(*integrate_reactor(reduced_mech2), label='one-step (A=1.2e9, Ea=40 kcal/mol)')
plt.plot(*integrate_reactor(detailed_mech), label='detailed')
plt.grid()
plt.legend(loc='best')
plt.ylabel('T (K)')
plt.xlabel('t (ms)')
plt.show()
