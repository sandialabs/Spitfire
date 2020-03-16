from spitfire.chemistry.mechanism import ChemicalMechanismSpec as Mechanism
from spitfire.chemistry.reactors import HomogeneousReactor
from spitfire.chemistry.analysis import get_ct_solution_array, explosive_mode_analysis
import matplotlib.pyplot as plt

sm = Mechanism(cantera_xml='h2-burke.xml', group_name='h2-burke')

h2 = sm.stream('X', 'H2:1')
air = sm.stream(stp_air=True)

mix = sm.mix_for_equivalence_ratio(1.0, h2, air)
mix.TP = 1200, 101325

r = HomogeneousReactor(sm, mix,
                       configuration='isobaric',
                       heat_transfer='adiabatic',
                       mass_transfer='closed')

output = r.integrate_to_steady_after_ignition(write_log=True)

t = output.time_values
T = output['temperature']

solarray, lib_shape = get_ct_solution_array(sm, output)

output['prod_rate_H'] = solarray.net_production_rates[:, sm.species_index('H')].reshape(lib_shape)
output['rate_cb'] = solarray.net_rates_of_progress[:, 0].reshape(lib_shape)

output = explosive_mode_analysis(sm, output)

plt.plot(T, output['prod_rate_H'])
plt.grid()
plt.xlabel('T (K)')
plt.ylabel('prod. rate H (kg/m3/s)')
plt.show()

plt.figure()
plt.subplot(211)
plt.semilogx(t, output['rate_cb'])
plt.grid()
plt.ylabel('net rate (mol/m3/s)')
plt.title('H + O2 <-> O + OH')
plt.subplot(212)
plt.semilogx(t, T)
plt.grid()
plt.xlabel('t (us)')
plt.ylabel('T (K)')
plt.show()

plt.figure()
plt.subplot(211)
plt.semilogx(t, output['cema-lexp1'] / 1.e3)
plt.grid()
plt.ylabel('explosive eigenvalue (kHz)')
plt.subplot(212)
plt.semilogx(t, T)
plt.grid()
plt.xlabel('t (us)')
plt.ylabel('T (K)')
plt.show()
