"""
Copyright (c) 2018-2019 Michael Alan Hansen - All Rights Reserved
You may use, distribute and modify this code under the terms of the MIT license.

You should have received a copy of the MIT license with this file.
If not, please write to mahanse@sandia.gov or mike.hansen@chemeng.utah.edu
"""

from spitfire.chemistry.mechanism import ChemicalMechanismSpec
from spitfire.chemistry.reactors import HomogeneousReactor
import matplotlib.pyplot as plt
from numpy import linspace, zeros_like
from cantera import one_atm as atm
from time import perf_counter

b = ChemicalMechanismSpec(cantera_xml='mechanisms/prf-curran.xml', group_name='prf-curran')

air = b.stream(stp_air=True)
h2 = b.stream('X', 'ic8h18:1 nc7h15:1')
phi = 1.0
blend = b.mix_for_equivalence_ratio(phi, h2, air)

print(blend.n_species, blend.n_reactions)

temperature_list = linspace(600., 1400., 10)
pressure = 10. * atm

t0 = perf_counter()
print('computing ignition delay profile for {:.1f} atm'.format(pressure / atm))
tau_list = zeros_like(temperature_list)

integrate_kwargs = {'linear_solver': 'superlu',
                    'write_log': False,
                    'minimum_time_step_count': 20,
                    'minimum_allowable_residual': 1.e-12,
                    'maximum_steps_per_jacobian': 20}

for idx, temperature in enumerate(temperature_list):
    print('computing temperature {:3} of {:2}'.format(idx + 1, temperature_list.size))
    failure = True
    first_time_step = 1.e-8
    while failure:
        try:
            t00 = perf_counter()
            print('  - trying with first time step =', first_time_step)
            integrate_kwargs['first_time_step'] = first_time_step
            mix = b.copy_stream(blend)
            mix.TP = temperature, pressure
            r = HomogeneousReactor(b, mix,
                                   'isochoric',
                                   'adiabatic',
                                   'closed')
            tau_list[idx] = r.compute_ignition_delay(**integrate_kwargs)
            failure = False
            break
        except:
            print('  - failure! reducing first time step')
            first_time_step *= 0.1
            if first_time_step < 1.e-10:
                raise ValueError('failure to converge!')
    print('  - converged in', perf_counter() - t00, 's')

plt.semilogy(temperature_list, tau_list * 1.e6, '-o', label='{:.1f} atm'.format(pressure / atm))

print('completed in', perf_counter() - t0, 's')
plt.xlabel('T (K)')
plt.ylabel('ignition delay (us)')
plt.legend()
plt.grid()
plt.show()
