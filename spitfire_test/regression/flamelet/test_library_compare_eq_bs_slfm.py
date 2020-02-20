import unittest


class Test(unittest.TestCase):
    def test(self):
        from spitfire.chemistry.mechanism import ChemicalMechanismSpec
        from spitfire.chemistry.tabulation import build_adiabatic_eq_library, \
            build_adiabatic_bs_library, \
            build_adiabatic_slfm_library
        import matplotlib.pyplot as plt
        import numpy as np
        from os.path import abspath, join

        xml = abspath(join('spitfire_test', 'test_mechanisms', 'heptane-liu-hewson-chen-pitsch-highT.xml'))
        m = ChemicalMechanismSpec(cantera_xml=xml, group_name='gas')

        pressure = 101325.

        air = m.stream(stp_air=True)
        air.TP = 1200., pressure

        fuel = m.stream('TPY', (485., pressure, 'NXC7H16:1'))
        fuel.TP = 300., pressure

        flamelet_specs = {'mech_spec': m,
                          'pressure': pressure,
                          'oxy_stream': air,
                          'fuel_stream': fuel,
                          'grid_points': 34,
                          'include_enthalpy_flux': True,
                          'include_variable_cp': True}

        quantities = ['enthalpy', 'temperature', 'mass fraction OH', 'mass fraction C2H2']

        l = build_adiabatic_eq_library(flamelet_specs, quantities, verbose=False)
        h_eq = l['enthalpy']
        T_eq = l['temperature']
        YOH_eq = l['mass fraction OH']
        YC2H2_eq = l['mass fraction C2H2']

        l = build_adiabatic_bs_library(flamelet_specs, quantities, verbose=False)
        h_bs = l['enthalpy']
        T_bs = l['temperature']
        YOH_bs = l['mass fraction OH']
        YC2H2_bs = l['mass fraction C2H2']

        l = build_adiabatic_slfm_library(flamelet_specs, quantities,
                                         diss_rate_values=np.logspace(-2, 3, 12),
                                         diss_rate_ref='maximum',
                                         verbose=False)

        chi_indices_plot = [0, 7, 9, 11]
        chi_values = l.dim('dissipation_rate_stoich').values

        plt.subplot(221)
        for ix, marker in zip(chi_indices_plot, ['s', 'o', 'd', '^']):
            z = l.dim('mixture_fraction').values
            plt.plot(z, l['temperature'][:, ix], 'c-',
                     marker=marker, markevery=4, markersize=5, markerfacecolor='w',
                     label='SLFM, $\\chi_{\\mathrm{max}}$=' + '{:.0e} 1/s'.format(chi_values[ix]))

        plt.plot(z, T_eq, 'P-', markevery=4, markersize=5, markerfacecolor='w', label='EQ')
        plt.plot(z, T_bs, 'H-', markevery=4, markersize=5, markerfacecolor='w', label='BS')

        plt.xlabel('mixture fraction')
        plt.ylabel('T (K)')
        plt.grid(True)
        plt.legend(loc=True)

        plt.subplot(222)
        for ix, marker in zip(chi_indices_plot, ['s', 'o', 'd', '^']):
            z = l.dim('mixture_fraction').values
            plt.plot(z, l['enthalpy'][:, ix] / 1.e6, 'c-',
                     marker=marker, markevery=4, markersize=5, markerfacecolor='w',
                     label='SLFM, $\\chi_{\\mathrm{max}}$=' + '{:.0e} 1/s'.format(chi_values[ix]))

        plt.plot(z, h_eq / 1.e6, 'P-', markevery=4, markersize=5, markerfacecolor='w', label='EQ')
        plt.plot(z, h_bs / 1.e6, 'H-', markevery=4, markersize=5, markerfacecolor='w', label='BS')

        plt.xlabel('mixture fraction')
        plt.ylabel('h (MJ/kg)')
        plt.grid(True)
        plt.legend(loc=True)

        plt.subplot(223)
        for ix, marker in zip(chi_indices_plot, ['s', 'o', 'd', '^']):
            z = l.dim('mixture_fraction').values
            plt.plot(z, l['mass fraction OH'][:, ix], 'c-',
                     marker=marker, markevery=4, markersize=5, markerfacecolor='w',
                     label='SLFM, $\\chi_{\\mathrm{max}}$=' + '{:.0e} 1/s'.format(chi_values[ix]))

        plt.plot(z, YOH_eq, 'P-', markevery=4, markersize=5, markerfacecolor='w', label='EQ')
        plt.plot(z, YOH_bs, 'H-', markevery=4, markersize=5, markerfacecolor='w', label='BS')

        plt.xlabel('mixture fraction')
        plt.ylabel('Y OH')
        plt.grid(True)
        plt.legend(loc=True)

        plt.subplot(224)
        for ix, marker in zip(chi_indices_plot, ['s', 'o', 'd', '^']):
            z = l.dim('mixture_fraction').values
            plt.plot(z, l['mass fraction C2H2'][:, ix], 'c-',
                     marker=marker, markevery=4, markersize=5, markerfacecolor='w',
                     label='SLFM, $\\chi_{\\mathrm{max}}$=' + '{:.0e} 1/s'.format(chi_values[ix]))

        plt.plot(z, YC2H2_eq, 'P-', markevery=4, markersize=5, markerfacecolor='w', label='EQ')
        plt.plot(z, YC2H2_bs, 'H-', markevery=4, markersize=5, markerfacecolor='w', label='BS')

        plt.xlabel('mixture fraction')
        plt.ylabel('Y C2H2')
        plt.grid(True)
        plt.legend(loc=True)


if __name__ == '__main__':
    unittest.main()
