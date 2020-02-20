import unittest


class Test(unittest.TestCase):
    def test(self):
        import numpy as np
        from spitfire.chemistry.mechanism import ChemicalMechanismSpec
        from spitfire.chemistry.flamelet import Flamelet
        from spitfire.chemistry.tabulation import Dimension, Library
        import pickle
        from os.path import abspath, join

        def get_fuel_stream(coal_fuels, alpha, mechanism, pressure):
            """
            Fuel streams representative of coal combustion, spanning char to volatiles.
            Personal communication with Josh McConnell, University of Utah, 2018.
            """

            volatiles = mechanism.stream('TPY', (coal_fuels['volatiles']['T'], pressure, coal_fuels['volatiles']['Y']))
            char = mechanism.stream('TPY', (coal_fuels['char']['T'], pressure, coal_fuels['char']['Y']))

            return mechanism.mix_streams([(volatiles, alpha), (char, 1. - alpha)], 'mass', 'HP')

        xml = abspath(join('spitfire_test', 'test_mechanisms', 'methane-gri30.xml'))
        mechanism = ChemicalMechanismSpec(cantera_xml=xml, group_name='methane-gri30')

        pressure = 101325.

        oxy = mechanism.stream(stp_air=True)
        oxy.TP = 350., pressure

        particle_temperature = 350.

        bcs_file = abspath(join('spitfire_test', 'regression', 'flamelet', 'coalflamelet_bcs.pkl'))
        with open(bcs_file, 'rb') as bcs_src:
            coal_fuels = pickle.load(bcs_src)

        alpha_vec = np.array([0.7])
        chist_vec = np.logspace(0., 2., 3)
        h_vec = np.hstack([0., np.logspace(-3, -1, 6)])

        npts_interior = 32

        base_specs = {'mech_spec': mechanism,
                      'pressure': pressure,
                      'oxy_stream': oxy,
                      'grid_points': npts_interior + 2,
                      'grid_type': 'uniform',
                      'heat_transfer': 'nonadiabatic',
                      'convection_temperature': particle_temperature,
                      'radiation_temperature': particle_temperature,
                      'radiative_emissivity': 0.}

        quantities = ['temperature', 'enthalpy', 'mass fraction C2H2']

        base_specs0 = dict(base_specs)
        base_specs0.update({'fuel_stream': get_fuel_stream(coal_fuels, 0., mechanism, pressure),
                            'initial_condition': 'linear-TY',
                            'convection_coefficient': 0.})
        f0 = Flamelet(**base_specs0)
        zdim = Dimension('mixture_fraction', f0.mixfrac_grid)
        xdim = Dimension('dissipation_rate_stoich', chist_vec)
        hdim = Dimension('heat_transfer_coefficient', h_vec)
        adim = Dimension('alpha', alpha_vec)

        l = Library(adim, hdim, xdim, zdim)

        values_dict = dict({q: l.get_empty_dataset() for q in quantities})

        for ia, alpha in enumerate(alpha_vec):
            base_specs.update({'fuel_stream': get_fuel_stream(coal_fuels, alpha, mechanism, pressure)})

            for ichi, chist in enumerate(chist_vec):
                base_specs.update({'stoich_dissipation_rate': chist})

                for ih, h in enumerate(h_vec):
                    base_specs.update({'convection_coefficient': h})
                    base_specs.update(
                        {'initial_condition': 'equilibrium' if ichi == 0 and ih == 0 else f.final_interior_state})
                    f = Flamelet(**base_specs)

                    f.compute_steady_state()

                    data_dict = f.process_quantities_on_state(f.final_state, quantities)

                    for quantity in quantities:
                        values_dict[quantity][ia, ih, ichi, :] = data_dict[quantity].ravel()

        for q in quantities:
            l[q] = values_dict[q]


if __name__ == '__main__':
    unittest.main()
