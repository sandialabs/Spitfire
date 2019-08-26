import unittest


class Test(unittest.TestCase):
    def test(self):
        import cantera as ct
        from spitfire.chemistry.mechanism import ChemicalMechanismSpec as Mechanism
        from spitfire.chemistry.flamelet import Flamelet
        import matplotlib.pyplot as plt
        import numpy as np
        from os.path import abspath, join

        xml = abspath(join('spitfire_test', 'test_mechanisms', 'h2-burke.xml'))
        sol = ct.Solution(xml, 'h2-burke')

        Tair = 1200.
        pressure = 101325.
        zstoich = 0.1
        chi_max = 1.e3
        npts_interior = 32

        k1mult_list = [0.01, 0.1, 0.5, 1.0, 2., 5.]
        sol_dict = dict()
        fig, axarray = plt.subplots(1, len(k1mult_list), sharex=True, sharey=True)

        max_time = 0.
        max_temp = 0.

        A0_original = np.copy(sol.reaction(0).rate.pre_exponential_factor)

        for i, k1mult in enumerate(k1mult_list):
            r0 = sol.reaction(0)
            new_rate = ct.Arrhenius(k1mult * A0_original,
                                    r0.rate.temperature_exponent,
                                    r0.rate.activation_energy)
            new_rxn = ct.ElementaryReaction(r0.reactants, r0.products)
            new_rxn.rate = new_rate
            sol.modify_reaction(0, new_rxn)

            m = Mechanism.from_solution(sol)
            air = m.stream(stp_air=True)
            air.TP = Tair, pressure
            fuel = m.mix_fuels_for_stoich_mixture_fraction(m.stream('X', 'H2:1'), m.stream('X', 'N2:1'), zstoich, air)
            fuel.TP = 300., pressure

            ft = Flamelet(mech_spec=m,
                          pressure=pressure,
                          oxy_stream=air,
                          fuel_stream=fuel,
                          max_dissipation_rate=chi_max,
                          grid_points=npts_interior + 2,
                          grid_cluster_intensity=4.,
                          initial_condition='unreacted')

            ft.insitu_process_quantity('temperature')
            ft.integrate_to_steady(first_time_step=1.e-9)

            t = ft.solution_times
            z = ft.mixfrac_grid
            T = ft.trajectory_data('temperature')
            max_time = max([max_time, np.max(t)])
            max_temp = max([max_temp, np.max(T)])

            sol_dict[k1mult] = (i, t, z, T)

        for k1mult in k1mult_list:
            sol = sol_dict[k1mult]
            axarray[sol[0]].contourf(sol[2], sol[1] * 1.e3, sol[3],
                                     cmap=plt.get_cmap('magma'),
                                     levels=np.linspace(300., max_temp, 20))
            axarray[sol[0]].set_title(f'{k1mult:.1f}x k1')
            axarray[sol[0]].set_xlim([0, 1])
            axarray[sol[0]].set_ylim([1.e-3, max_time * 1.e3])
            axarray[sol[0]].set_yscale('log')
            axarray[sol[0]].set_xlabel('Z')

        axarray[0].set_ylabel('t (ms)')

if __name__ == '__main__':
    unittest.main()
