from os.path import abspath, join


def run():
    from spitfire.chemistry.mechanism import ChemicalMechanismSpec
    from spitfire.chemistry.tabulation import build_adiabatic_bs_library
    import spitfire.chemistry.analysis as sca

    test_xml = abspath(join('tests', 'test_mechanisms', 'h2-burke.xml'))
    m = ChemicalMechanismSpec(cantera_xml=test_xml, group_name='h2-burke')
    pressure = 101325.
    air = m.stream(stp_air=True)
    air.TP = 1200., pressure
    fuel = m.stream('TPY', (300., pressure, 'H2:1'))

    flamelet_specs = {'mech_spec': m, 'oxy_stream': air, 'fuel_stream': fuel, 'grid_points': 34}

    l = build_adiabatic_bs_library(flamelet_specs, verbose=False)
    l = sca.compute_specific_enthalpy(m, l)
    l = sca.compute_isochoric_specific_heat(m, l)
    l = sca.compute_isobaric_specific_heat(m, l)
    l = sca.compute_density(m, l)
    l = sca.compute_pressure(m, l)
    l = sca.compute_viscosity(m, l)

    return l


if __name__ == '__main__':
    gold_pkl = abspath(join('tests', 'tabulation', 'adiabatic_burke_schumann', 'gold.pkl'))
    output_library = run()
    output_library.save_to_file(gold_pkl)
