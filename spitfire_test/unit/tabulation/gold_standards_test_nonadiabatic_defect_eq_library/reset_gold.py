from spitfire.chemistry.mechanism import ChemicalMechanismSpec
from spitfire.chemistry.tabulation import build_nonadiabatic_defect_eq_library

m = ChemicalMechanismSpec(cantera_xml='h2-burke.xml', group_name='h2-burke')
pressure = 101325.
air = m.stream(stp_air=True)
air.TP = 1200., pressure
fuel = m.stream('TPY', (300., pressure, 'H2:1'))

flamelet_specs = {'mech_spec': m, 'pressure': pressure,
                  'oxy_stream': air, 'fuel_stream': fuel,
                  'grid_points': 34}

l = build_nonadiabatic_defect_eq_library(flamelet_specs, verbose=False)

l.save_to_file('library_gold.pkl')
