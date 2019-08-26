from spitfire.chemistry.mechanism import ChemicalMechanismSpec
import numpy as np
import cantera as ct

species_decl = list([ct.Species('A', 'H:2'),
                     ct.Species('B', 'H:1'),
                     ct.Species('C', 'O:1'),
                     ct.Species('D', 'O:2'),
                     ct.Species('E', 'H:1, O:2'),
                     ct.Species('N', 'H:1, O:1')])
species_dict = dict({'const': list(), 'nasa7': list()})
for s in species_decl:
    species_dict['const'].append(ct.Species(s.name, s.composition))
    species_dict['const'][-1].thermo = ct.ConstantCp(300., 3000., 101325., (300., 0., 0., 1.e4))
    species_dict['nasa7'].append(ct.Species(s.name, s.composition))
    coeffs = [1.e0, 1.e-2, 1.e-4, 1.e-6, 1.e-8, 1.e-10, 1.e-12]
    species_dict['nasa7'][-1].thermo = ct.NasaPoly2(300., 3000., 101325., np.hstack([1200.] + coeffs + coeffs))

reactions_dict = dict(
    {
        'elem-arborder, irr, 11_11, constant':
            [ct.Reaction.fromCti('''reaction('A + D => B + E', [1, 0, 0], order='A:0.5 D:1.5')''')],
        'elem, irr, 11_11, constant': [ct.Reaction.fromCti('''reaction('A + D => B + E', [1, 0, 0])''')],
        # 'elem, irr, 11_11, linear': [ct.Reaction.fromCti('''reaction('A + D => B + E', [1, 1, 0])''')],
        # 'elem, irr, 11_11, quadratic': [ct.Reaction.fromCti('''reaction('A + D => B + E', [1, 2, 0])''')],
        # 'elem, irr, 11_11, reciprocal': [ct.Reaction.fromCti('''reaction('A + D => B + E', [1, -1, 0])''')],
        # 'elem, irr, 11_11, arrhenius': [ct.Reaction.fromCti('''reaction('A + D => B + E', [1, 1, 1])''')],
        # 'elem, rev, 11_11, constant': [ct.Reaction.fromCti('''reaction('A + D <=> B + E', [1, 0, 0])''')],
        # 'elem, rev, 2_11, constant': [ct.Reaction.fromCti('''reaction('2 D <=> D + 2 C', [1, 0, 0])''')],
        # 'elem, rev, 11_11, linear': [ct.Reaction.fromCti('''reaction('A + D <=> B + E', [1, 1, 0])''')],
        # 'elem, rev, 11_11, quadratic': [ct.Reaction.fromCti('''reaction('A + D <=> B + E', [1, 2, 0])''')],
        # 'elem, rev, 11_11, reciprocal': [ct.Reaction.fromCti('''reaction('A + D <=> B + E', [1, -1, 0])''')],
        # 'elem, rev, 11_11, arrhenius': [ct.Reaction.fromCti('''reaction('A + D <=> B + E', [1, 1, 1])''')],
        # 'elem, rev, 2_11, arrhenius': [ct.Reaction.fromCti('''reaction('2 D <=> D + 2 C', [1, 1, 1])''')],
        # '3bdy-noN, irr, 11_11, constant':
        #     [ct.Reaction.fromCti('''three_body_reaction('A + D => B + E', [1, 0, 0], efficiencies = 'C:2')''')],
        # '3bdy-wN, irr, 11_11, constant':
        #     [ct.Reaction.fromCti('''three_body_reaction('A + D => B + E', [1, 0, 0], efficiencies = 'C:2, N:2')''')],
    }
)

s = ct.Solution(thermo='IdealGas',
                kinetics='GasKinetics',
                species=species_dict['nasa7'],
                reactions=reactions_dict['elem, irr, 11_11, constant'])
g = ChemicalMechanismSpec.from_solution(s).griffon

# ns = s.n_species
# T = 1000.
# rho = 1.
# s.TDY = T, rho, np.ones(ns)
# w_ct = s.net_production_rates * s.molecular_weights
# w_gr = np.zeros(ns)
# g.production_rates(T, rho, s.Y, w_gr)
# print('w:   ', np.max(np.abs(w_gr - w_ct)))
#
# jacGR = np.zeros((ns + 1) * (ns + 1))
# g.prod_rates_primitive_sensitivities(rho, T, s.Y, jacGR)
# jacGR = jacGR.reshape((ns + 1, ns + 1), order='F')[:-1, :]
#
# rhsGR1 = np.zeros(ns)
# rhsGR2 = np.zeros(ns)
# dT = 1.e-4
# g.production_rates(T + dT, rho, s.Y, rhsGR1)
# g.production_rates(T - dT, rho, s.Y, rhsGR2)
# print('T:   ', np.max(np.abs((((rhsGR1 - rhsGR2) / dT * 0.5) - jacGR[:, 1]) / (jacGR[:, 1] + 1.e-4))))
#
# rhsGR1 = np.zeros(ns)
# rhsGR2 = np.zeros(ns)
# dr = 1.e-4
# g.production_rates(T, rho + dr, s.Y, rhsGR1)
# g.production_rates(T, rho - dr, s.Y, rhsGR2)
# print('rho: ', np.max(np.abs((((rhsGR1 - rhsGR2) / dr * 0.5) - jacGR[:, 0]) / (jacGR[:, 0] + 1.e-4))))
#
# for spec_idx in range(ns - 1):
#     rhsGR1 = np.zeros(ns)
#     rhsGR2 = np.zeros(ns)
#     dY = 1.e-4
#     Yp = np.copy(s.Y)
#     Yp[spec_idx] += dY
#     Yp[-1] -= dY
#     Ym = np.copy(s.Y)
#     Ym[spec_idx] -= dY
#     Ym[-1] += dY
#     g.production_rates(T, rho, Yp, rhsGR1)
#     g.production_rates(T, rho, Ym, rhsGR2)
#     print(f'Y_{spec_idx}: ',
#           np.max(np.abs((((rhsGR1 - rhsGR2) / dY * 0.5) - jacGR[:, 2 + spec_idx]) / (jacGR[:, 2 + spec_idx] + 1.e-4))))
#
# rho = s.density_mass
# state = np.hstack((rho, s.T, s.Y[:-1]))
# rhsGR = np.empty(ns + 1)
# rhsGRTemporary = np.empty(ns + 1)
# jacGR = np.empty((ns + 1) * (ns + 1))
# g.reactor_rhs_isochoric(state, 0, 0, np.ndarray(1), 0, 0, 0, 0, 0, 0, 0, False, rhsGR)
# g.reactor_jac_isochoric(state, 0, 0, np.ndarray(1), 0, 0, 0, 0, 0, 0, 0, False, 0, rhsGRTemporary, jacGR)
# jacGR = jacGR.reshape((ns + 1, ns + 1), order='F')
#
# print(np.max(np.abs(rhsGR - rhsGRTemporary)))
# print(rhsGR)
#
# jacFD = np.zeros((ns + 1, ns + 1))
# wm1 = np.zeros(ns + 1)
# wp1 = np.zeros(ns + 1)
# drho = 1.e-4
# dT = 1.e-2
# dY = 1.e-6
#
# state_m = np.hstack((rho - drho, s.T, s.Y[:-1]))
# state_p = np.hstack((rho + drho, s.T, s.Y[:-1]))
# g.reactor_rhs_isochoric(state_m, 0, 0, np.ndarray(1), 0, 0, 0, 0, 0, 0, 0, False, wm1)
# g.reactor_rhs_isochoric(state_p, 0, 0, np.ndarray(1), 0, 0, 0, 0, 0, 0, 0, False, wp1)
# jacFD[:, 0] = (- wm1 + wp1) / (2. * drho)
#
# state_m = np.hstack((rho, s.T - dT, s.Y[:-1]))
# state_p = np.hstack((rho, s.T + dT, s.Y[:-1]))
# g.reactor_rhs_isochoric(state_m, 0, 0, np.ndarray(1), 0, 0, 0, 0, 0, 0, 0, False, wm1)
# g.reactor_rhs_isochoric(state_p, 0, 0, np.ndarray(1), 0, 0, 0, 0, 0, 0, 0, False, wp1)
# jacFD[:, 1] = (- wm1 + wp1) / (2. * dT)
#
# for i in range(ns - 1):
#     y_m1, y_p1 = np.copy(s.Y), np.copy(s.Y)
#     y_m1[i] += - dY
#     y_m1[-1] -= - dY
#     y_p1[i] += dY
#     y_p1[-1] -= dY
#     state_m = np.hstack((rho, s.T, y_m1[:-1]))
#     state_p = np.hstack((rho, s.T, y_p1[:-1]))
#     g.reactor_rhs_isochoric(state_m, 0, 0, np.ndarray(1), 0, 0, 0, 0, 0, 0, 0, False, wm1)
#     g.reactor_rhs_isochoric(state_p, 0, 0, np.ndarray(1), 0, 0, 0, 0, 0, 0, 0, False, wp1)
#     jacFD[:, 2 + i] = (- wm1 + wp1) / (2. * dY)
# #
# for i in range(ns + 1):
#     for j in range(ns + 1):
#         if np.abs(jacFD[i, j] - jacGR[i, j]) / (jacGR[i, j] + 1.e-4) > 1.e-8:
#             print(i, j, jacGR[i, j], jacFD[i, j], jacFD[i, j] - jacGR[i, j])

# ems = ['tnt-pitz',
#        'nalkanes816-westbrook',
#        'methyldecanoate-herbinet',
#        'methylbutanoate-fisher',
#        'methanol-zabetta',
#        'methane-gri211',
#        'methane-gri12',
#        'isopentanol-tsujimura',
#        'h2-zsely',
#        'h2-sun',
#        'h2-li',
#        'h2-konnov',
#        'h2-gri',
#        'h2-davis',
#        'h2-dagaut',
#        'ethanol-curran',
#        'dme-fischer',
#        'diesel-pei',
#        'cyclohexane-silke',
#        'coh2-li',
#        'coh2-gri',
#        'coh2-davis',
#        'alkylaromatics-nakamura',
#        'prf-curran',
#        'octane-mehl',
#        'methane-kazakov24',
#        'methane-kazakov21',
#        'methane-grired',
#        'methane-gri30-nonox',
#        'methane-gri30',
#        'heptane-wang',
#        'heptane-patel',
#        'heptane-mehl',
#        'heptane-lu188',
#        'h2-oconaire',
#        'h2-cantera',
#        'gasoline-mehl-Cal-323',
#        'gasoline-mehl',
#        'ethylene-williams',
#        'ethylene-usc',
#        'ethylene-luo',
#        'ethanol-marinov',
#        'dme-zhao',
#        'dme-bhagatwala',
#        'coh2-yetter',
#        'coh2-hawkes',
#        'butanol-sarathy',
#        'butane-marinov',
#        'biodiesel-lu-lowT',
#        'biodiesel-lu-highT',
#        'biodiesel-herbinet',
#        'methane-lu30']
#
# table = f'{"mechanism":40} | {"ns":>4} | {"nr":>5} | {"mr":>2} | {"mp":>2} | {"mn":>2} | {"mt":>2} | ' \
#         f'{"elem":>5} | {"3body":>5} | {"Lind":>5} | {"Troe":>5} | {"rev":>5}\n'
# table += '-' * len(table) + '\n'
#
# for em in ems:
#     cantera_xml = f'/sierra/dev/mahanse/ChemicalMechanisms/{em}/{em}.xml'
#     group_name = em
#
#     print(f'loading {cantera_xml}')
#     ctsol = ct.Solution(cantera_xml, group_name)
#
#     data = ChemicalMechanismSpec._extract_cantera_mechanism_data(ctsol)
#
#     elem_list, ref_temperature, ref_pressure, spec_name_list, spec_dict, reac_list = data
#
#     nreactants = map(lambda r: len(r['reactants']), reac_list)
#     nproducts = map(lambda r: len(r['products']), reac_list)
#
#     def get_net_species(r):
#         n = dict()
#         for rc in r['reactants']:
#             n[rc] = r['reactants'][rc]
#         for rc in r['products']:
#             if rc in n:
#                 n[rc] -= r['products'][rc]
#             else:
#                 n[rc] = r['products'][rc]
#         to_del = []
#         for rc in n:
#             if n[rc] < 1.e-14:
#                 to_del.append(rc)
#         for td in to_del:
#             n.pop(td)
#         return n
#
#     netspecies = map(lambda r: len(get_net_species(r)), reac_list)
#     nthirdbodies = map(lambda r: 0 if 'efficiencies' not in r else len(r['efficiencies']), reac_list)
#     nelementary = sum(map(lambda r: 1 if r['type'] == 'elementary' else 0, reac_list))
#     nthreebody = sum(map(lambda r: 1 if r['type'] == 'threebody' else 0, reac_list))
#     nlindemann = sum(map(lambda r: 1 if r['type'] == 'Lindemann' else 0, reac_list))
#     ntroe = sum(map(lambda r: 1 if r['type'] == 'Troe' else 0, reac_list))
#     nrev = sum(map(lambda r: 1 if r['reversible'] else 0, reac_list))
#
#     chem_mech = '/sierra/dev/mahanse/ChemicalMechanisms/'
#     table += f'{cantera_xml.split(chem_mech)[1].split("/")[0]:40} | {len(spec_name_list):>4} | {len(reac_list):>5} | ' \
#              f'{max(nreactants):>2} | {max(nproducts):>2} | {max(netspecies):>2} | {max(nthirdbodies):>2} | ' \
#              f'{nelementary:>5} | {nthreebody:>5} | {nlindemann:>5} | {ntroe:>5} | {nrev:>5}\n'
#
# print('\n\n' + table)
