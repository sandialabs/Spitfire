import unittest
from os.path import join, abspath
from spitfire.chemistry.mechanism import ChemicalMechanismSpec
from spitfire.chemistry.flamelet import Flamelet


def insitu_processing(heat_transfer):
    test_xml = abspath(join('spitfire_test', 'test_mechanisms', 'h2-burke.xml'))
    mechanism = ChemicalMechanismSpec(cantera_xml=test_xml, group_name='h2-burke')
    air = mechanism.stream(stp_air=True)
    fuel = mechanism.stream('X', 'H2:1')

    flamelet_specs = {'mech_spec': mechanism,
                      'pressure': air.P,
                      'oxy_stream': air,
                      'fuel_stream': fuel,
                      'grid_points': 8,
                      'grid_cluster_intensity': 4.,
                      'initial_condition': 'equilibrium',
                      'max_dissipation_rate': 0.}

    if heat_transfer == 'nonadiabatic':
        flamelet_specs.update({'heat_transfer': 'nonadiabatic',
                               'convection_temperature': 350.,
                               'convection_coefficient': 0.,
                               'radiation_temperature': 350.,
                               'radiative_emissivity': 0.})

    try:
        f = Flamelet(**flamelet_specs)
        f.insitu_process_cantera_method('enthalpy_mass')
        f.insitu_process_quantity(
            ['temperature', 'density', 'pressure', 'energy', 'enthalpy', 'heat capacity cv', 'heat capacity cp',
             'mass fractions', 'mole fractions', 'production rates', 'heat release rate'])
        f.insitu_process_cema(True, True, True)
        data_dict = f.process_quantities_on_state(f.initial_interior_state)
        return True
    except Exception as e:
        print(e)
        return False


def create_test(h):
    def test(self):
        self.assertTrue(insitu_processing(h))

    return test


class InSituProcessing(unittest.TestCase):
    pass


for heat_transfer in ['adiabatic', 'nonadiabatic']:
    testname = 'test_insitu_processing_coverage_' + heat_transfer
    setattr(InSituProcessing, testname, create_test(heat_transfer))

if __name__ == '__main__':
    unittest.main()
