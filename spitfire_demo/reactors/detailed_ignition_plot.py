from spitfire.chemistry.mechanism import ChemicalMechanismSpec as Mechanism
from spitfire.chemistry.reactors import HomogeneousReactor
from os.path import abspath, join
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

r.insitu_process_quantity(['temperature', 'mass fractions', 'production rates'])
r.insitu_process_cantera_method('cp_mass')
r.insitu_process_cantera_method(label='cpm', method='cp_mass')
r.insitu_process_cantera_method(label='qCB', method='net_rates_of_progress', index=0)
r.insitu_process_cantera_method(label='cH', method='concentrations', index='H')
r.insitu_process_cema()

r.integrate_to_steady_after_ignition()

t = r.solution_times * 1.e6  # scale to microseconds
T = r.trajectory_data('temperature')
yH = r.trajectory_data('mass fraction H')
wH = r.trajectory_data('production rate H')
qCB = r.trajectory_data('qCB')

plt.plot(T, wH)
plt.grid()
plt.xlabel('T (K)')
plt.ylabel('prod. rate H (kg/m3/s)')
plt.show()

plt.plot(T, r.trajectory_data('cH'))
plt.grid()
plt.xlabel('T (K)')
plt.ylabel('concentration of H (mol/m3)')
plt.show()

plt.figure()
plt.subplot(211)
plt.semilogx(t, qCB)
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
plt.semilogx(t, r.trajectory_data('cema-lexp1') / 1.e3)
plt.grid()
plt.ylabel('explosive eigenvalue (kHz)')
plt.subplot(212)
plt.semilogx(t, T)
plt.grid()
plt.xlabel('t (us)')
plt.ylabel('T (K)')
plt.show()
