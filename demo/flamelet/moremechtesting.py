from spitfire.chemistry.mechanism import ChemicalMechanismSpec
import numpy as np
import cantera as ct

species_decl = list([ct.Species('C8H18', 'C:8,H:18'),
                     ct.Species('O2', 'O:2'),
                     ct.Species('H2O', 'H:2,O:1'),
                     ct.Species('CO2', 'C:1,O:2'),
                     ct.Species('N2', 'N:2')])
species_dict = dict({'const': list(), 'nasa7': list()})
for s in species_decl:
    species_dict['const'].append(ct.Species(s.name, s.composition))
    species_dict['const'][-1].thermo = ct.ConstantCp(300., 3000., 101325., (300., 0., 0., 1.e4))
    species_dict['nasa7'].append(ct.Species(s.name, s.composition))
    coeffs = [1.e0, 1.e-2, 1.e-4, 1.e-6, 1.e-8, 1.e-10, 1.e-12]
    species_dict['nasa7'][-1].thermo = ct.NasaPoly2(300., 3000., 101325., np.hstack([1200.] + coeffs + coeffs))

reaction_list = [ct.Reaction.fromCti(
    '''
reaction(
'2 C8H18 + 25 O2 => 16 CO2 + 18 H2O', 
[4.6e11, 0, (30.0, 'kcal/mol')], 
order='C8H18:0.25 O2:1.5 CO2:0.1',
options=['nonreactant_orders'])
    ''')]
s = ct.Solution(thermo='IdealGas',
                kinetics='GasKinetics',
                species=species_dict['nasa7'],
                reactions=reaction_list)
g = ChemicalMechanismSpec.from_solution(s).griffon

ns = s.n_species
T = 1000.
rho = 1.
s.TDY = T, rho, np.array([0.2, 0.2, 0.2, 0.2, 0.2])
w_ct = s.net_production_rates * s.molecular_weights
w_gr = np.zeros(ns)
g.production_rates(T, rho, s.Y, w_gr)
print('w:   ', np.max(np.abs(w_gr - w_ct)))

jacGR = np.zeros((ns + 1) * (ns + 1))
g.prod_rates_primitive_sensitivities(rho, T, s.Y, jacGR)
jacGR = jacGR.reshape((ns + 1, ns + 1), order='F')[:-1, :]

rhsGR1 = np.zeros(ns)
rhsGR2 = np.zeros(ns)
dT = 1.e-4
g.production_rates(T + dT, rho, s.Y, rhsGR1)
g.production_rates(T - dT, rho, s.Y, rhsGR2)
print('T:   ', np.max(np.abs((((rhsGR1 - rhsGR2) / dT * 0.5) - jacGR[:, 1]) / (jacGR[:, 1] + 1.e-4))))

rhsGR1 = np.zeros(ns)
rhsGR2 = np.zeros(ns)
dr = 1.e-4
g.production_rates(T, rho + dr, s.Y, rhsGR1)
g.production_rates(T, rho - dr, s.Y, rhsGR2)
print('rho: ', np.max(np.abs((((rhsGR1 - rhsGR2) / dr * 0.5) - jacGR[:, 0]) / (jacGR[:, 0] + 1.e-4))))

for spec_idx in range(ns - 1):
    rhsGR1 = np.zeros(ns)
    rhsGR2 = np.zeros(ns)
    dY = 1.e-4
    Yp = np.copy(s.Y)
    Yp[spec_idx] += dY
    Yp[-1] -= dY
    g.production_rates(T, rho, Yp, rhsGR1)
    g.production_rates(T, rho, s.Y, rhsGR2)
    print(f'Y_{spec_idx}: ', (rhsGR1 - rhsGR2) / dY, jacGR[:, 2 + spec_idx])
    print(f'Y_{spec_idx}: ',
          np.max(np.abs((((rhsGR1 - rhsGR2) / dY) - jacGR[:, 2 + spec_idx]) / (jacGR[:, 2 + spec_idx] + 1.e-4))))
    # print(jacGR[:, 2 + spec_idx])
    # print((rhsGR1 - rhsGR2) / dY * 0.5)
