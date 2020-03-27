import unittest
import numpy as np
import cantera as ct
from os.path import join, abspath
from spitfire import ChemicalMechanismSpec as Mechanism

T_range = [300, 1200, 1800]
p_range = [101325, 1013250]


def verify_rates_mechanism(ctsol: ct.Solution):
    mech = Mechanism.from_solution(ctsol)
    tolerance = 1.e-14

    def verify_T_p(temp, pres):
        valid = True
        for mix in [np.ones(ctsol.n_species),
                    np.array([1., 1.e-8, 1.e-8, 1.e-8, 1.e-8, 1.e-8]),
                    np.array([1.e-8, 1., 1.e-8, 1.e-8, 1.e-8, 1.e-8]),
                    np.array([1.e-8, 1.e-8, 1., 1.e-8, 1.e-8, 1.e-8]),
                    np.array([1.e-8, 1.e-8, 1.e-8, 1., 1.e-8, 1.e-8]),
                    np.array([1.e-8, 1.e-8, 1.e-8, 1.e-8, 1., 1.e-8]),
                    np.array([1.e-8, 1.e-8, 1.e-8, 1.e-8, 1.e-8, 1.]),
                    np.array([1., 1.e-16, 1.e-16, 1.e-16, 1.e-16, 1.e-16]),
                    np.array([1.e-16, 1., 1.e-16, 1.e-16, 1.e-16, 1.e-16]),
                    np.array([1.e-16, 1.e-16, 1., 1.e-16, 1.e-16, 1.e-16]),
                    np.array([1.e-16, 1.e-16, 1.e-16, 1., 1.e-16, 1.e-16]),
                    np.array([1.e-16, 1.e-16, 1.e-16, 1.e-12, 1., 1.e-16]),
                    np.array([1.e-16, 1.e-16, 1.e-16, 1.e-12, 1.e-16, 1.])]:
            ns = ctsol.n_species
            ctsol.TPY = temp, pres, mix
            rho = ctsol.density_mass
            w_ct = ctsol.net_production_rates * ctsol.molecular_weights
            w_gr = np.zeros(ns)
            mech.griffon.production_rates(T, rho, ctsol.Y, w_gr)
            valid = valid and np.max(np.abs(w_gr - w_ct) / (np.abs(w_ct) + 1.e0)) < tolerance
        return valid

    pass_test = True

    for T in T_range:
        for p in p_range:
            pass_test = pass_test and verify_T_p(T, p)

    return pass_test


def verify_sensitivities_mechanism(ctsol: ct.Solution):
    mech = Mechanism.from_solution(ctsol)

    tolerance = 1.e-2

    def verify_T_p(temp, pres):
        valid = True
        for mix in [np.ones(ctsol.n_species)]:
            ns = ctsol.n_species
            ctsol.TPY = temp, pres, mix
            rho = ctsol.density_mass

            jacGR = np.zeros((ns + 1) * (ns + 1))
            mech.griffon.prod_rates_primitive_sensitivities(rho, T, ctsol.Y, 0, jacGR)
            jacGR = jacGR.reshape((ns + 1, ns + 1), order='F')[:-1, :]

            jacFD = np.zeros_like(jacGR)
            rhsGR1 = np.zeros(ns)
            rhsGR2 = np.zeros(ns)
            dT = 1.e-4
            dr = 1.e-4
            dY = 1.e-4
            mech.griffon.production_rates(T, rho + dr, ctsol.Y, rhsGR1)
            mech.griffon.production_rates(T, rho - dr, ctsol.Y, rhsGR2)
            jacFD[:, 0] = (rhsGR1 - rhsGR2) / dr * 0.5
            mech.griffon.production_rates(T + dT, rho, ctsol.Y, rhsGR1)
            mech.griffon.production_rates(T - dT, rho, ctsol.Y, rhsGR2)
            jacFD[:, 1] = (rhsGR1 - rhsGR2) / dT * 0.5

            for spec_idx in range(ns - 1):
                Yp = np.copy(ctsol.Y)
                Yp[spec_idx] += dY
                Yp[-1] -= dY
                Ym = np.copy(ctsol.Y)
                Ym[spec_idx] -= dY
                Ym[-1] += dY
                mech.griffon.production_rates(T, rho, Yp, rhsGR1)
                mech.griffon.production_rates(T, rho, Ym, rhsGR2)
                jacFD[:, 2 + spec_idx] = (rhsGR1 - rhsGR2) / dY * 0.5

            scale = np.abs(jacFD) + 1.0
            error = np.max(np.abs((jacFD - jacGR) / scale)) > tolerance
            if error:
                print(f'T = {T}, p = {p}, Y = {mix}')
                print('fd:')
                for i in range(ns):
                    for j in range(ns + 1):
                        print(f'{jacFD[i, j]:12.2e}', end=', ')
                    print('')
                print('gr:')
                for i in range(ns):
                    for j in range(ns + 1):
                        print(f'{jacGR[i, j]:12.2e}', end=', ')
                    print('')
                print('gr-fd:')
                for i in range(ns):
                    for j in range(ns + 1):
                        print(f'{jacGR[i, j] - jacFD[i, j]:12.2e}', end=', ')
                    print('')
            valid = valid and not error
        return valid

    pass_test = True

    for T in T_range:
        for p in p_range:
            pass_test = pass_test and verify_T_p(T, p)

    return pass_test


reaction_indices = dict({
    'elem_irr_11_11_noN_constant': 0,
    'nonelem_lt1_irr_11_11_noN_constant': 1,
    'nonelem_lt1gtq_irr_11_11_noN_constant': 2,
    'nonelem_gt1_irr_11_11_noN_constant': 3,
    'elem_irr_11_11_noN_linear': 4,
    'elem_irr_11_11_noN_quadratic': 5,
    'elem_irr_11_11_noN_reciprocal': 6,
    'elem_irr_11_11_noN_Arrhenius': 7,
    'elem_rev_11_11_noN_constant': 8,
    'elem_rev_11_11_noN_linear': 9,
    'elem_rev_11_11_noN_quadratic': 10,
    'elem_rev_11_11_noN_reciprocal': 11,
    'elem_rev_11_11_noN_Arrhenius': 12,
    '3body_rev_11_11_noN_Arrhenius': 13,
    'Lindemann_rev_11_11_noN_Arrhenius': 14,
    'Troe_rev_11_11_noN_Arrhenius': 15})

xml = abspath(join('spitfire_test', 'test_mechanisms', 'reaction_test_mechanism.xml'))

def create_test(reaction_key, type):
    def test(self):
        sol = ct.Solution(thermo='IdealGas',
                          kinetics='GasKinetics',
                          species=ct.Species.listFromFile(xml),
                          reactions=[ct.Reaction.listFromFile(xml)[reaction_indices[reaction_key]]])
        if type == 'rates':
            self.assertTrue(verify_rates_mechanism(sol))
        elif type == 'sensitivities':
            self.assertTrue(verify_sensitivities_mechanism(sol))

    return test


class Accuracy(unittest.TestCase):
    pass


for reaction_key in reaction_indices:
    for quantity in ['rates', 'sensitivities']:
        setattr(Accuracy, 'test_' + quantity + '_' + reaction_key, create_test(reaction_key, quantity))

if __name__ == '__main__':
    unittest.main()
