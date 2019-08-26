"""
Copyright (c) 2018-2019 Michael Alan Hansen - All Rights Reserved
You may use, distribute and modify this code under the terms of the MIT license.

You should have received a copy of the MIT license with this file.
If not, please write to mahanse@sandia.gov or mike.hansen@chemeng.utah.edu
"""

from spitfire.chemistry.mechanism import ChemicalMechanismSpec
from spitfire.chemistry.flamelet import Flamelet
import matplotlib.pyplot as plt
import numpy as np

m = ChemicalMechanismSpec(cantera_xml='mechanisms/h2-burke.xml', group_name='h2-burke')
fuel = m.stream('X', 'h2:5, n2:1')

pressure = 101325.

air = m.stream(stp_air=True)
air.TP = 1400., pressure

fuel.TP = 300., pressure

chi_max = 1.e2

npts_interior = 32

f = Flamelet(m, 'unreacted', pressure, air, fuel, chi_max,
             grid_points=npts_interior + 2, grid_cluster_intensity=4.)

f.insitu_process_quantity(['temperature', 'density', 'heat capacity cp',
                           'mass fractions', 'production rates', 'heat release rate',
                           'mole fractions', 'enthalpy', 'energy', 'heat capacity cv'])
f.insitu_process_cema()

f.integrate_to_time(final_time=5.e-3,
                    write_log=True,
                    log_rate=40,
                    first_time_step=1.e-8,
                    linear_solver='block thomas',
                    transient_tolerance=1.e-10,
                    nonlinear_solve_tolerance=1.e-10)

t_ms = f.solution_times * 1.e3
z = f.mixfrac_grid
T = f.trajectory_data('temperature')
wT = f.trajectory_data('heat release rate')
wH = f.trajectory_data('production rate H')
YOH = f.trajectory_data('mass fraction OH')
lexp_kHz = f.trajectory_data('cema-lexp1') / 1.e3

# you can also get post-processed quantities with the process_quantities_on_state(state) method:
final_state = f.final_interior_state
final_state_data = f.process_quantities_on_state(final_state)

# the following quantities are the last time step of the insitu-processed temperature,
# and the temperature computed with the process_quantities_on_state method called on the final flamelet state.
# they are identical
Tfinal_insitu = T[-1, :]
Tfinal_processed = final_state_data['temperature']

plt.contour(z, t_ms, lexp_kHz, levels=[100., 300., 500.], colors=['k'], linewidths=1.)
plt.contourf(z, t_ms, T, cmap=plt.get_cmap('rainbow'), levels=np.linspace(300., 3000., 17))
plt.xlabel('mixture fraction')
plt.ylabel('t (ms)')
plt.title('Temperature (K)')
plt.ylim([0., 0.1 * t_ms[-1]])
plt.colorbar()
plt.show()

plt.contour(z, t_ms, lexp_kHz, levels=[100., 300., 500.], colors=['k'], linewidths=1.)
plt.contourf(z, t_ms, wT, cmap=plt.get_cmap('rainbow'))
plt.xlabel('mixture fraction')
plt.ylabel('t (ms)')
plt.title('Heat release rate (K/s)')
plt.ylim([0., 0.1 * t_ms[-1]])
plt.colorbar()
plt.show()

plt.contour(z, t_ms, lexp_kHz, levels=[100., 300., 500.], colors=['k'], linewidths=1.)
plt.contourf(z, t_ms, wH, cmap=plt.get_cmap('bwr'))
plt.xlabel('mixture fraction')
plt.ylabel('t (ms)')
plt.title('Production rate H (kg/m3/s)')
plt.ylim([0., 0.1 * t_ms[-1]])
plt.colorbar()
plt.show()

plt.contour(z, t_ms, lexp_kHz, levels=[100., 300., 500.], colors=['k'], linewidths=1.)
plt.contourf(z, t_ms, YOH, cmap=plt.get_cmap('BuPu'))
plt.xlabel('mixture fraction')
plt.ylabel('t (ms)')
plt.title('mass fraction OH')
plt.ylim([0., 0.1 * t_ms[-1]])
plt.colorbar()
plt.show()
