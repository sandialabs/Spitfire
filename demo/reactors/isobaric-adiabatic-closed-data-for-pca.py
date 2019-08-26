"""
Copyright (c) 2018-2019 Michael Alan Hansen - All Rights Reserved
You may use, distribute and modify this code under the terms of the MIT license.

You should have received a copy of the MIT license with this file.
If not, please write to mahanse@sandia.gov or mike.hansen@chemeng.utah.edu
"""

from spitfire.chemistry.mechanism import ChemicalMechanismSpec
from spitfire.chemistry.reactors import HomogeneousReactor
import numpy as np

b = ChemicalMechanismSpec(cantera_xml='mechanisms/h2-burke.xml', group_name='h2-burke')

air = b.stream(stp_air=True)
h2 = b.stream('X', 'H2:1')

mix = b.mix_for_equivalence_ratio(1.0, h2, air)
mix.TP = 1200., 101325.

nspec = mix.n_species
species_names = mix.species_names
ty_names = ['T'] + species_names[:-1]

r = HomogeneousReactor(b, mix, 'isobaric', 'adiabatic', 'closed')

r.insitu_process_quantity(['temperature', 'mass fractions', 'density', 'heat release rate', 'production rates'])
r.integrate_to_steady_after_ignition()

nsteps = r.solution_times.size

ty_state_data = np.zeros([nsteps, nspec])
ty_source_data = np.zeros([nsteps, nspec])

ty_state_data[:, 0] = r.trajectory_data('temperature')
ty_source_data[:, 0] = r.trajectory_data('heat release rate')
rho = r.trajectory_data('density')
for i in range(nspec - 1):
    ty_state_data[:, 1 + i] = r.trajectory_data('mass fraction ' + species_names[i])
    ty_source_data[:, 1 + i] = r.trajectory_data('production rate ' + species_names[i]) / rho

print('States:')
header = ''
for name in ty_names:
    header += '\t{:>8}'.format('T' if name == 'T' else 'Y ' + name)
print(header)
for i in range(nsteps):
    row = ''
    for j in range(nspec):
        row += '\t{:>8.1e}'.format(ty_state_data[i, j])
    print(row)

print('Source terms:')
header = ''
for name in ty_names:
    header += '\t{:>8}'.format('S T' if name == 'T' else 'S Y ' + name)
print(header)
for i in range(nsteps):
    row = ''
    for j in range(nspec):
        row += '\t{:>8.1e}'.format(ty_source_data[i, j])
    print(row)
