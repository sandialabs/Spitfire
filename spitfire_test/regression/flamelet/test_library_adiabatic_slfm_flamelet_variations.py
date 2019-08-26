import unittest


class Test(unittest.TestCase):
    def test(self):
        from spitfire.chemistry.mechanism import ChemicalMechanismSpec
        from spitfire.chemistry.tabulation import build_adiabatic_slfm_library
        import matplotlib.pyplot as plt
        import numpy as np

        from os.path import abspath, join

        xml = abspath(join('spitfire_test', 'test_mechanisms', 'h2-burke.xml'))
        m = ChemicalMechanismSpec(cantera_xml=xml, group_name='h2-burke')

        pressure = 101325.

        air = m.stream(stp_air=True)
        air.TP = 1400., pressure

        fuel = m.mix_fuels_for_stoich_mixture_fraction(m.stream('X', 'H2:1'), m.stream('X', 'N2:1'), 0.2, air)
        fuel.TP = 300., pressure

        fig, axarr = plt.subplots(3, 1, sharex=True)
        Tax = axarr[0]
        hax = axarr[1]
        Yax = axarr[2]

        for include_flux, include_vcp, line_style, legend in \
                [(False, False, 'b-', 'classical'),
                 (True, False, 'r--', 'w/h flux'),
                 (True, True, 'g-.', 'w/h flux + var. cp')]:
            flamelet_specs = {'mech_spec': m,
                              'pressure': pressure,
                              'oxy_stream': air,
                              'fuel_stream': fuel,
                              'grid_points': 34,
                              'include_enthalpy_flux': include_flux,
                              'include_variable_cp': include_vcp}

            library = build_adiabatic_slfm_library(flamelet_specs,
                                                   ['temperature', 'enthalpy', 'mass fraction OH'],
                                                   diss_rate_ref='stoichiometric',
                                                   diss_rate_values=np.logspace(0, 4, 4),
                                                   verbose=False)

            x_values = library.dim('dissipation_rate_stoich').values
            z = library.dim('mixture_fraction').values

            for ix, (chi_st, marker) in enumerate(zip(x_values, ['s', 'o', 'd', '^'])):
                Tax.plot(z, library['temperature'][:, ix], line_style,
                         marker=marker, markevery=4, markersize=5, markerfacecolor='w', label=legend)
                hax.plot(z, library['enthalpy'][:, ix] / 1.e6, line_style,
                         marker=marker, markevery=4, markersize=5, markerfacecolor='w', label=legend)
                Yax.plot(z, library['mass fraction OH'][:, ix], line_style,
                         marker=marker, markevery=4, markersize=5, markerfacecolor='w', label=legend)
                Tmax = np.max(library['temperature'][:, ix])
                YOHmax = np.max(library['mass fraction OH'][:, ix])

        Yax.set_xlabel('mixture fraction')
        Tax.set_ylabel('T (K)')
        hax.set_ylabel('h (MJ/kg)')
        Yax.set_ylabel('Y OH')
        for ax in axarr:
            ax.grid(True)
        Tax.legend(loc='best')


if __name__ == '__main__':
    unittest.main()
