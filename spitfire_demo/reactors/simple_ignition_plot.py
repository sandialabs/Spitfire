from spitfire.chemistry.mechanism import ChemicalMechanismSpec as Mechanism
from spitfire.chemistry.reactors import HomogeneousReactor

sm = Mechanism(cantera_xml='h2-burke.xml', group_name='h2-burke')

h2 = sm.stream('TPX', (300, 101325, 'H2:1'))
air = sm.stream(stp_air=True)

mix = sm.mix_for_equivalence_ratio(1.0, h2, air)
mix.TP = 1200, 101325

r = HomogeneousReactor(sm, mix,
                       configuration='isochoric',
                       heat_transfer='adiabatic',
                       mass_transfer='closed')

r.integrate_to_steady_after_ignition(plot=['H2', 'O2', 'H2O', 'OH', 'H'])
