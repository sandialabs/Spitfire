import unittest
import numpy as np
from os.path import join, abspath
import cantera as ct
from spitfire import ChemicalMechanismSpec as Mechanism, HomogeneousReactor
import spitfire.chemistry.analysis as sca

xml = abspath(join('spitfire_test', 'test_mechanisms', 'hydrogen_one_step.xml'))
sol = ct.Solution(thermo='IdealGas',
                  kinetics='GasKinetics',
                  species=ct.Species.listFromFile(xml),
                  reactions=ct.Reaction.listFromFile(xml))
mechanism = Mechanism.from_solution(sol)


def construct_reactor(configuration, mass_transfer, heat_transfer, shape):
    air = mechanism.stream(stp_air=True)
    fuel = mechanism.stream('X', 'H2:1')

    mix = mechanism.mix_for_equivalence_ratio(1.0, fuel, air)
    mix.TP = 1200., 101325.

    feed = mechanism.copy_stream(mix)

    tau = 1.e-3

    extra_args = dict()
    if mass_transfer == 'open':
        if configuration == 'isobaric':
            extra_args['feed_temperature'] = feed.T
            extra_args['feed_mass_fractions'] = feed.Y
            extra_args['mixing_tau'] = tau
        elif configuration == 'isochoric':
            extra_args['feed_temperature'] = feed.T
            extra_args['feed_mass_fractions'] = feed.Y
            extra_args['feed_density'] = feed.density
            extra_args['mixing_tau'] = tau
    if heat_transfer == 'diathermal':
        extra_args['convection_coefficient'] = 1.
        extra_args['convection_temperature'] = 300.
        extra_args['radiative_emissivity'] = 1.
        extra_args['radiation_temperature'] = 300.
        extra_args['shape_dimension_dict'] = {'shape': shape, 'char. length': 1.e-3}

    try:
        HomogeneousReactor(mechanism, mix,
                           configuration=configuration,
                           heat_transfer=heat_transfer,
                           mass_transfer=mass_transfer,
                           **extra_args)
        return True
    except:
        return False


def integrate_a_few_steps(configuration, mass_transfer, heat_transfer, shape):
    air = mechanism.stream(stp_air=True)
    fuel = mechanism.stream('X', 'H2:1')

    mix = mechanism.mix_for_equivalence_ratio(1.0, fuel, air)
    mix.TP = 1200., 101325.

    feed = mechanism.copy_stream(mix)

    tau = 1.e-3

    extra_args = dict()
    if mass_transfer == 'open':
        if configuration == 'isobaric':
            extra_args['feed_temperature'] = feed.T
            extra_args['feed_mass_fractions'] = feed.Y
            extra_args['mixing_tau'] = tau
        elif configuration == 'isochoric':
            extra_args['feed_temperature'] = feed.T
            extra_args['feed_mass_fractions'] = feed.Y
            extra_args['feed_density'] = feed.density
            extra_args['mixing_tau'] = tau
    if heat_transfer == 'diathermal':
        extra_args['convection_coefficient'] = 1.
        extra_args['convection_temperature'] = 300.
        extra_args['radiative_emissivity'] = 1.
        extra_args['radiation_temperature'] = 300.
        extra_args['shape_dimension_dict'] = {'shape': shape, 'char. length': 1.e-3}

    try:
        reactor = HomogeneousReactor(mechanism, mix,
                                     configuration=configuration,
                                     heat_transfer=heat_transfer,
                                     mass_transfer=mass_transfer,
                                     **extra_args)
        reactor.integrate_to_time(final_time=1.e-6,
                                  first_time_step=1.e-7,
                                  minimum_time_step_count=1)
        return True
    except:
        return False


def integrate_steady(configuration, mass_transfer, heat_transfer, shape):
    air = mechanism.stream(stp_air=True)
    fuel = mechanism.stream('X', 'H2:1')

    mix = mechanism.mix_for_equivalence_ratio(1.0, fuel, air)
    mix.TP = 1200., 101325.

    feed = mechanism.copy_stream(mix)

    tau = 1.e-3

    extra_args = dict()
    if mass_transfer == 'open':
        if configuration == 'isobaric':
            extra_args['feed_temperature'] = feed.T
            extra_args['feed_mass_fractions'] = feed.Y
            extra_args['mixing_tau'] = tau
        elif configuration == 'isochoric':
            extra_args['feed_temperature'] = feed.T
            extra_args['feed_mass_fractions'] = feed.Y
            extra_args['feed_density'] = feed.density
            extra_args['mixing_tau'] = tau
    if heat_transfer == 'diathermal':
        extra_args['convection_coefficient'] = 1.
        extra_args['convection_temperature'] = 300.
        extra_args['radiative_emissivity'] = 1.
        extra_args['radiation_temperature'] = 300.
        extra_args['shape_dimension_dict'] = {'shape': shape, 'char. length': 1.e-3}

    try:
        reactor = HomogeneousReactor(mechanism, mix,
                                     configuration=configuration,
                                     heat_transfer=heat_transfer,
                                     mass_transfer=mass_transfer,
                                     **extra_args)
        output = reactor.integrate_to_steady(steady_tolerance=1.e-4,
                                             transient_tolerance=1.e-8)
        num_time_steps = output.time_npts
        return 20 < num_time_steps < 300
    except:
        return False


def adiabatic_closed_ignition_delay(configuration):
    air = mechanism.stream(stp_air=True)
    fuel = mechanism.stream('X', 'H2:1')

    try:
        mix = mechanism.mix_for_equivalence_ratio(1.0, fuel, air)
        mix.TP = 1200., 101325.
        reactor = HomogeneousReactor(mechanism, mix,
                                     configuration=configuration,
                                     heat_transfer='adiabatic',
                                     mass_transfer='closed',
                                     rates_sensitivity_type='dense')
        t_ignition_1 = reactor.compute_ignition_delay(transient_tolerance=1.e-8, return_solution=False)
        reactor = HomogeneousReactor(mechanism, mix,
                                     configuration=configuration,
                                     heat_transfer='adiabatic',
                                     mass_transfer='closed',
                                     rates_sensitivity_type='dense')
        t_ignition_2 = reactor.compute_ignition_delay(transient_tolerance=1.e-10, return_solution=False)
        reactor = HomogeneousReactor(mechanism, mix,
                                     configuration=configuration,
                                     heat_transfer='adiabatic',
                                     mass_transfer='closed',
                                     rates_sensitivity_type='sparse')
        t_ignition_3 = reactor.compute_ignition_delay(transient_tolerance=1.e-10, return_solution=False)

        success = (t_ignition_1 - t_ignition_2) / t_ignition_2 < 0.05 and \
                  (t_ignition_3 - t_ignition_2) / t_ignition_2 < 1e-6
        if not success:
            print(configuration, t_ignition_1, t_ignition_2, t_ignition_3)
        return success
    except Exception as e:
        print(configuration, e)
        return False


def adiabatic_closed_steady_after_ignition(configuration):
    air = mechanism.stream(stp_air=True)
    fuel = mechanism.stream('X', 'H2:1')

    mix = mechanism.mix_for_equivalence_ratio(1.0, fuel, air)
    mix.TP = 1200., 101325.

    try:
        reactor = HomogeneousReactor(mechanism, mix,
                                     configuration=configuration,
                                     heat_transfer='adiabatic',
                                     mass_transfer='closed')
        reactor.integrate_to_steady_after_ignition()
        return True
    except:
        return False


def post_processing(configuration, mass_transfer, heat_transfer):
    air = mechanism.stream(stp_air=True)
    fuel = mechanism.stream('X', 'H2:1')

    mix = mechanism.mix_for_equivalence_ratio(1.0, fuel, air)
    mix.TP = 1200., 101325.

    feed = mechanism.copy_stream(mix)

    tau = 1.e-3

    extra_args = dict()
    if mass_transfer == 'open':
        if configuration == 'isobaric':
            extra_args['feed_temperature'] = feed.T
            extra_args['feed_mass_fractions'] = feed.Y
            extra_args['mixing_tau'] = tau
        elif configuration == 'isochoric':
            extra_args['feed_temperature'] = feed.T
            extra_args['feed_mass_fractions'] = feed.Y
            extra_args['feed_density'] = feed.density
            extra_args['mixing_tau'] = tau
    if heat_transfer == 'diathermal':
        extra_args['convection_coefficient'] = 1.
        extra_args['convection_temperature'] = 300.
        extra_args['radiative_emissivity'] = 1.
        extra_args['radiation_temperature'] = 300.
        extra_args['shape_dimension_dict'] = {'shape': 'sphere', 'char. length': 1.e-3}

    try:
        reactor = HomogeneousReactor(mechanism, mix,
                                     configuration=configuration,
                                     heat_transfer=heat_transfer,
                                     mass_transfer=mass_transfer,
                                     **extra_args)

        tol = np.sqrt(np.finfo(float).eps)
        test_success = True

        output_library = reactor.integrate_to_time(1e-16, minimum_time_step_count=0)
        output_library = sca.compute_specific_enthalpy(mechanism, output_library)
        output_library = sca.compute_density(mechanism, output_library)
        output_library = sca.compute_pressure(mechanism, output_library)
        output_library = sca.compute_isobaric_specific_heat(mechanism, output_library)
        output_library = sca.compute_isochoric_specific_heat(mechanism, output_library)
        output_library = sca.explosive_mode_analysis(mechanism, output_library,
                                                     configuration, heat_transfer,
                                                     True, True, True)

        test_success = test_success and np.abs(mix.T - output_library['temperature'][-1]) / mix.T < tol
        test_success = test_success and np.abs(mix.P - output_library['pressure'][-1]) / mix.P < tol
        test_success = test_success and np.abs(mix.density - output_library['density'][-1]) / mix.density < tol
        test_success = test_success and np.abs(mix.enthalpy - output_library['enthalpy'][-1]) / mix.enthalpy_mass < tol
        test_success = test_success and np.abs(mix.cv_mass - output_library['heat capacity cv'][-1]) / mix.cv_mass < tol
        test_success = test_success and np.abs(mix.cp_mass - output_library['heat capacity cp'][-1]) / mix.cp_mass < tol

        return test_success
    except:
        return False


def create_test(test_method, c, m=None, h=None, s=None):
    if test_method == 'construct':
        def test(self):
            self.assertTrue(construct_reactor(c, m, h, s))
    elif test_method == 'integrate_a_few_steps':
        def test(self):
            self.assertTrue(integrate_a_few_steps(c, m, h, s))
    elif test_method == 'integrate_steady':
        def test(self):
            self.assertTrue(integrate_steady(c, m, h, s))
    elif test_method == 'ignition_delay':
        def test(self):
            self.assertTrue(adiabatic_closed_ignition_delay(c))
    elif test_method == 'steady_after_ignition':
        def test(self):
            self.assertTrue(adiabatic_closed_steady_after_ignition(c))
    elif test_method == 'post_processing_coverage':
        def test(self):
            self.assertTrue(post_processing(c, m, h))

    return test


class Construction(unittest.TestCase):
    pass


shape = list(HomogeneousReactor.get_supported_reactor_shapes())[0]
for configuration in ['isobaric', 'isochoric']:
    for mass_transfer in ['closed', 'open']:
        for heat_transfer in ['adiabatic', 'isothermal', 'diathermal']:
            testname = 'test_construct_reactor_' + configuration + \
                       '_' + mass_transfer + \
                       '_' + heat_transfer + \
                       '_' + shape
            setattr(Construction, testname, create_test('construct',
                                                        configuration,
                                                        mass_transfer,
                                                        heat_transfer,
                                                        shape))
            testname = 'test_integrate_a_few_steps_' + configuration + \
                       '_' + mass_transfer + \
                       '_' + heat_transfer + \
                       '_' + shape
            setattr(Construction, testname, create_test('integrate_a_few_steps',
                                                        configuration,
                                                        mass_transfer,
                                                        heat_transfer,
                                                        shape))
            testname = 'test_integrate_steady_' + configuration + \
                       '_' + mass_transfer + \
                       '_' + heat_transfer + \
                       '_' + shape
            setattr(Construction, testname, create_test('integrate_steady',
                                                        configuration,
                                                        mass_transfer,
                                                        heat_transfer,
                                                        'cube'))

            testname = 'test_post_processing_coverage_' + configuration + '_' + mass_transfer + '_' + heat_transfer
            setattr(Construction, testname,
                    create_test('post_processing_coverage', configuration, mass_transfer, heat_transfer))

    testname = 'test_compute_ignition_delay_adiabatic_closed_' + configuration
    setattr(Construction, testname, create_test('ignition_delay', configuration))

    testname = 'test_compute_steady_after_ignition_adiabatic_closed_' + configuration
    setattr(Construction, testname, create_test('steady_after_ignition', configuration))

if __name__ == '__main__':
    unittest.main()
