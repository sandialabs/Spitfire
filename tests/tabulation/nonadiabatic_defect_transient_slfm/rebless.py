from os.path import abspath, join


def run(num_procs):
    from spitfire.chemistry.mechanism import ChemicalMechanismSpec
    from spitfire.chemistry.tabulation import build_nonadiabatic_defect_transient_slfm_library
    import spitfire.chemistry.analysis as sca
    from spitfire.data.get import datafile
    import numpy as np

    test_xml = datafile('burke-hydrogen.yaml')
    m = ChemicalMechanismSpec(cantera_input=test_xml, group_name='gas')
    pressure = 101325.
    air = m.stream(stp_air=True)
    air.TP = 1200., pressure
    fuel = m.stream('TPY', (300., pressure, 'H2:1'))

    flamelet_specs = {'mech_spec': m, 'oxy_stream': air, 'fuel_stream': fuel, 'grid_points': 34}

    l = build_nonadiabatic_defect_transient_slfm_library(flamelet_specs, verbose=False,
                                                         diss_rate_values=np.logspace(0, 1, 4),
                                                         integration_args={'transient_tolerance': 1e-10},
                                                         num_procs=num_procs)
    l = sca.compute_specific_enthalpy(m, l)
    l = sca.compute_isochoric_specific_heat(m, l)
    l = sca.compute_isobaric_specific_heat(m, l)
    l = sca.compute_density(m, l)
    l = sca.compute_pressure(m, l)
    l = sca.compute_viscosity(m, l)

    return l


if __name__ == '__main__':
    gold_pkl = abspath(join('tests', 'tabulation', 'nonadiabatic_defect_transient_slfm', 'gold.pkl'))
    output_library = run(num_procs=1)
    output_library.save_to_file(gold_pkl)
