import unittest
import numpy as np
import pickle
from os.path import join, abspath
from spitfire import ChemicalMechanismSpec, Flamelet, FlameletSpec


def construct_adiabatic_flamelet(initialization, grid_type, diss_rate_form):
    test_xml = abspath(join('tests', 'test_mechanisms', 'h2-burke.xml'))
    mechanism = ChemicalMechanismSpec(cantera_xml=test_xml, group_name='h2-burke')
    air = mechanism.stream(stp_air=True)
    fuel = mechanism.stream('X', 'H2:1')
    fuel.TP = 300, air.P

    if grid_type == 'uniform':
        grid_specs = {'grid_points': 8}
    elif grid_type == 'clustered-1args':
        grid_specs = {'grid_points': 8}
    elif grid_type == 'clustered-2args':
        grid_specs = {'grid_points': 8, 'grid_cluster_intensity': 4.}
    elif grid_type == 'clustered-3args':
        grid_specs = {'grid_points': 8, 'grid_cluster_intensity': 4., 'grid_cluster_point': 0.4}
    elif grid_type == 'custom':
        grid_specs = {'grid': np.linspace(0., 1., 8)}

    if diss_rate_form == 'Peters':
        drf_specs = {'max_dissipation_rate': 1., 'dissipation_rate_form': diss_rate_form}
    elif diss_rate_form == 'uniform':
        drf_specs = {'max_dissipation_rate': 1., 'dissipation_rate_form': diss_rate_form}
    elif diss_rate_form == 'custom':
        drf_specs = {'dissipation_rate': np.linspace(0., 1., 8)}

    flamelet_specs = {'mech_spec': mechanism,
                      'oxy_stream': air,
                      'fuel_stream': fuel,
                      'initial_condition': initialization}
    flamelet_specs.update(grid_specs)
    flamelet_specs.update(drf_specs)

    try:
        Flamelet(**flamelet_specs)
        Flamelet(flamelet_specs=flamelet_specs)
        fso = FlameletSpec(**flamelet_specs)
        f = Flamelet(fso)
        fso_pickle = pickle.dumps(fso)
        fso2 = pickle.loads(fso_pickle)
        f = Flamelet(fso2)

        lib = f.make_library_from_interior_state(f.initial_interior_state)
        Flamelet(library_slice=lib)

        lib_pickle = pickle.dumps(lib)
        lib2 = pickle.loads(lib_pickle)
        Flamelet(library_slice=lib2)
        return True
    except:
        return False


def construct_nonadiabatic_flamelet(initialization, grid_type, diss_rate_form):
    test_xml = abspath(join('tests', 'test_mechanisms', 'h2-burke.xml'))
    mechanism = ChemicalMechanismSpec(cantera_xml=test_xml, group_name='h2-burke')
    air = mechanism.stream(stp_air=True)
    fuel = mechanism.stream('X', 'H2:1')
    fuel.TP = 300, air.P

    if grid_type == 'uniform':
        grid_specs = {'grid_points': 8}
    elif grid_type == 'clustered-1args':
        grid_specs = {'grid_points': 8}
    elif grid_type == 'clustered-2args':
        grid_specs = {'grid_points': 8, 'grid_cluster_intensity': 4.}
    elif grid_type == 'clustered-3args':
        grid_specs = {'grid_points': 8, 'grid_cluster_intensity': 4., 'grid_cluster_point': 0.4}
    elif grid_type == 'custom':
        grid_specs = {'grid': np.linspace(0., 1., 8)}

    if diss_rate_form == 'Peters':
        drf_specs = {'max_dissipation_rate': 1., 'dissipation_rate_form': diss_rate_form}
    elif diss_rate_form == 'uniform':
        drf_specs = {'max_dissipation_rate': 1., 'dissipation_rate_form': diss_rate_form}
    elif diss_rate_form == 'custom':
        drf_specs = {'dissipation_rate': np.linspace(0., 1., 8)}

    try:
        flamelet_specs = {'mech_spec': mechanism,
                          'oxy_stream': air,
                          'fuel_stream': fuel,
                          'initial_condition': initialization,
                          'heat_transfer': 'nonadiabatic',
                          'convection_temperature': 350.,
                          'convection_coefficient': 0.,
                          'radiation_temperature': 350.,
                          'radiative_emissivity': 0.}
        flamelet_specs.update(grid_specs)
        flamelet_specs.update(drf_specs)
        Flamelet(**flamelet_specs)
        Flamelet(flamelet_specs=flamelet_specs)
        fso = FlameletSpec(**flamelet_specs)
        f = Flamelet(fso)

        fso_pickle = pickle.dumps(fso)
        fso2 = pickle.loads(fso_pickle)
        f = Flamelet(fso2)

        lib = f.make_library_from_interior_state(f.initial_interior_state)
        Flamelet(library_slice=lib)

        lib_pickle = pickle.dumps(lib)
        lib2 = pickle.loads(lib_pickle)
        Flamelet(library_slice=lib2)

        flamelet_specs = {'mech_spec': mechanism,
                          'oxy_stream': air,
                          'fuel_stream': fuel,
                          'initial_condition': initialization,
                          'heat_transfer': 'nonadiabatic',
                          'scale_heat_loss_by_temp_range': True,
                          'scale_convection_by_dissipation': True,
                          'use_linear_ref_temp_profile': True,
                          'convection_coefficient': 1.e7,
                          'radiative_emissivity': 0.}
        flamelet_specs.update(grid_specs)
        flamelet_specs.update(drf_specs)

        Flamelet(**flamelet_specs)
        Flamelet(flamelet_specs=flamelet_specs)
        fso = FlameletSpec(**flamelet_specs)
        f = Flamelet(fso)

        fso_pickle = pickle.dumps(fso)
        fso2 = pickle.loads(fso_pickle)
        f = Flamelet(fso2)

        lib = f.make_library_from_interior_state(f.initial_interior_state)
        Flamelet(library_slice=lib)

        lib_pickle = pickle.dumps(lib)
        lib2 = pickle.loads(lib_pickle)
        Flamelet(library_slice=lib2)

        flamelet_specs = {'mech_spec': mechanism,
                          'oxy_stream': air,
                          'fuel_stream': fuel,
                          'initial_condition': initialization,
                          'heat_transfer': 'nonadiabatic',
                          'scale_heat_loss_by_temp_range': False,
                          'scale_convection_by_dissipation': False,
                          'use_linear_ref_temp_profile': True,
                          'convection_coefficient': 1.e7,
                          'radiative_emissivity': 0.}
        flamelet_specs.update(grid_specs)
        flamelet_specs.update(drf_specs)

        Flamelet(**flamelet_specs)
        Flamelet(flamelet_specs=flamelet_specs)
        fso = FlameletSpec(**flamelet_specs)
        f = Flamelet(fso)

        fso_pickle = pickle.dumps(fso)
        fso2 = pickle.loads(fso_pickle)
        f = Flamelet(fso2)

        lib = f.make_library_from_interior_state(f.initial_interior_state)
        Flamelet(library_slice=lib)

        lib_pickle = pickle.dumps(lib)
        lib2 = pickle.loads(lib_pickle)
        Flamelet(library_slice=lib2)

        flamelet_specs = {'mech_spec': mechanism,
                          'oxy_stream': air,
                          'fuel_stream': fuel,
                          'initial_condition': initialization,
                          'heat_transfer': 'nonadiabatic',
                          'scale_heat_loss_by_temp_range': False,
                          'scale_convection_by_dissipation': True,
                          'use_linear_ref_temp_profile': True,
                          'convection_coefficient': 1.e7,
                          'radiative_emissivity': 0.}
        flamelet_specs.update(grid_specs)
        flamelet_specs.update(drf_specs)

        Flamelet(**flamelet_specs)
        Flamelet(flamelet_specs=flamelet_specs)
        fso = FlameletSpec(**flamelet_specs)
        f = Flamelet(fso)

        fso_pickle = pickle.dumps(fso)
        fso2 = pickle.loads(fso_pickle)
        f = Flamelet(fso2)

        lib = f.make_library_from_interior_state(f.initial_interior_state)
        Flamelet(library_slice=lib)

        lib_pickle = pickle.dumps(lib)
        lib2 = pickle.loads(lib_pickle)
        Flamelet(library_slice=lib2)

        return True
    except:
        return False


def create_test(ht, ic, gt, drf):
    if ht == 'adiabatic':
        def test(self):
            self.assertTrue(construct_adiabatic_flamelet(ic, gt, drf))
    elif ht == 'nonadiabatic':
        def test(self):
            self.assertTrue(construct_nonadiabatic_flamelet(ic, gt, drf))
    return test


class Construction(unittest.TestCase):
    pass


test_xml = abspath(join('tests', 'test_mechanisms', 'h2-burke.xml'))
mechanism = ChemicalMechanismSpec(cantera_xml=test_xml, group_name='h2-burke')
air = mechanism.stream(stp_air=True)
fuel = mechanism.stream('X', 'H2:1')
fuel.TP = 300, air.P

flamelet_specs = {'mech_spec': mechanism,
                  'oxy_stream': air,
                  'fuel_stream': fuel,
                  'grid_points': 8,
                  'grid_cluster_intensity': 4.,
                  'initial_condition': 'equilibrium',
                  'max_dissipation_rate': 0.}

temp_flamelet = Flamelet(**flamelet_specs)

for ht in temp_flamelet._heat_transfers:
    for ic in ['equilibrium', 'linear-TY', 'unreacted', temp_flamelet.initial_interior_state]:
        for gt in ['uniform', 'clustered-1args', 'clustered-2args', 'clustered-3args', 'custom']:
            for drf in ['uniform', 'Peters', 'custom']:
                ic_str = 'icstate' if isinstance(ic, np.ndarray) else ic
                testname = 'test_construct_flamelet_' + ht + '_' + ic_str + '_' + gt + '_' + drf
                setattr(Construction, testname, create_test(ht, ic, gt, drf))

if __name__ == '__main__':
    unittest.main()
