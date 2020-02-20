import unittest


class Test(unittest.TestCase):
    def test(self):
        from spitfire.chemistry.tabulation import build_nonadiabatic_defect_transient_slfm_library
        from spitfire.chemistry.mechanism import ChemicalMechanismSpec
        import numpy as np
        from os.path import abspath, join

        xml = abspath(join('spitfire_test', 'test_mechanisms', 'h2-burke.xml'))
        mech = ChemicalMechanismSpec(cantera_xml=xml, group_name='h2-burke')
        pressure = 101325.
        fuel = mech.stream('TPX', (300., pressure, 'H2:1'))
        air = mech.stream(stp_air=True)
        air.TP = 300., pressure

        flamelet_specs = {'mech_spec': mech,
                          'pressure': pressure,
                          'oxy_stream': air,
                          'fuel_stream': fuel,
                          'grid_points': 34,
                          'grid_cluster_intensity': 8.}

        quantities = ['temperature', 'mass fraction OH', 'enthalpy']

        build_nonadiabatic_defect_transient_slfm_library(flamelet_specs, quantities,
                                                         diss_rate_values=np.logspace(-1, 0, 2),
                                                         diss_rate_ref='stoichiometric',
                                                         verbose=False, n_defect_st=4, num_procs=1)


if __name__ == '__main__':
    unittest.main()
