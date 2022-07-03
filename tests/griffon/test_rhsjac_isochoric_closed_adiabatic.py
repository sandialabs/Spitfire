import unittest
from numpy import max, abs, ones, zeros, copy, sum, sqrt, hstack
from cantera import Solution, one_atm, gas_constant
import numpy as np
from spitfire import ChemicalMechanismSpec
from os.path import join, abspath
from subprocess import getoutput

test_mech_directory = abspath(join('tests', 'test_mechanisms', 'old_xmls'))
mechs = [x.replace('.yaml', '') for x in getoutput('ls ' + test_mech_directory + ' | grep .yaml').split('\n')]


def validate_on_mechanism(mech, temperature, pressure, test_rhs=True, test_jac=True):
    xml = join(test_mech_directory, mech + '.yaml')

    r = ChemicalMechanismSpec(xml, 'gas').griffon

    gas = Solution(xml)
    ns = gas.n_species

    T = temperature
    p = pressure

    gas.TPX = T, p, ones(ns)
    y = gas.Y
    rho = gas.density_mass

    state = hstack((rho, T, y[:-1]))

    rhsGR = np.empty(ns + 1)
    rhsGRTemporary = np.empty(ns + 1)
    jacGR = np.empty((ns + 1) * (ns + 1))
    r.reactor_rhs_isochoric(state, 0, 0, np.ndarray(1), 0, 0, 0, 0, 0, 0, 0, False, rhsGR)
    r.reactor_jac_isochoric(state, 0, 0, np.ndarray(1), 0, 0, 0, 0, 0, 0, 0, False, 0, rhsGRTemporary, jacGR)
    jacGR = jacGR.reshape((ns + 1, ns + 1), order='F')

    def cantera_rhs(rho_arg, T_arg, Y_arg):
        gas.TDY = T_arg, rho_arg, Y_arg
        w = gas.net_production_rates * gas.molecular_weights
        e = gas.standard_int_energies_RT * gas.T * gas_constant / gas.molecular_weights
        cv = gas.cv_mass
        rhs = zeros(ns + 1)
        rhs[0] = 0.
        rhs[1] = - sum(w * e) / (rho_arg * cv)
        rhs[2:] = w[:-1] / rho
        return rhs

    rhsCN = cantera_rhs(rho, T, y)

    if test_rhs:
        pass_rhs = max(abs(rhsGR - rhsCN) / (abs(rhsCN) + 1.)) < 100. * sqrt(np.finfo(float).eps)

    if test_jac:
        jacFD = zeros((ns + 1, ns + 1))
        wm1 = zeros(ns + 1)
        wp1 = zeros(ns + 1)
        drho = 1.e-4
        dT = 1.e-2
        dY = 1.e-6

        state_m = hstack((rho - drho, T, y[:-1]))
        state_p = hstack((rho + drho, T, y[:-1]))
        r.reactor_rhs_isochoric(state_m, 0, 0, np.ndarray(1), 0, 0, 0, 0, 0, 0, 0, False, wm1)
        r.reactor_rhs_isochoric(state_p, 0, 0, np.ndarray(1), 0, 0, 0, 0, 0, 0, 0, False, wp1)
        jacFD[:, 0] = (- wm1 + wp1) / (2. * drho)

        state_m = hstack((rho, T - dT, y[:-1]))
        state_p = hstack((rho, T + dT, y[:-1]))
        r.reactor_rhs_isochoric(state_m, 0, 0, np.ndarray(1), 0, 0, 0, 0, 0, 0, 0, False, wm1)
        r.reactor_rhs_isochoric(state_p, 0, 0, np.ndarray(1), 0, 0, 0, 0, 0, 0, 0, False, wp1)
        jacFD[:, 1] = (- wm1 + wp1) / (2. * dT)

        for i in range(ns - 1):
            y_m1, y_p1 = copy(y), copy(y)
            y_m1[i] += - dY
            y_m1[-1] -= - dY
            y_p1[i] += dY
            y_p1[-1] -= dY
            state_m = hstack((rho, T, y_m1[:-1]))
            state_p = hstack((rho, T, y_p1[:-1]))
            r.reactor_rhs_isochoric(state_m, 0, 0, np.ndarray(1), 0, 0, 0, 0, 0, 0, 0, False, wm1)
            r.reactor_rhs_isochoric(state_p, 0, 0, np.ndarray(1), 0, 0, 0, 0, 0, 0, 0, False, wp1)
            jacFD[:, 2 + i] = (- wm1 + wp1) / (2. * dY)

        gas.TDY = T, rho, y
        cv = gas.cv_mass
        cvi = gas.standard_cp_R * gas_constant / gas.molecular_weights
        w = gas.net_production_rates * gas.molecular_weights
        e = gas.standard_int_energies_RT * gas.T * gas_constant / gas.molecular_weights

        gas.TDY = T + dT, rho, y
        wp = gas.net_production_rates * gas.molecular_weights
        cvp = gas.cv_mass

        gas.TDY = T - dT, rho, y
        wm = gas.net_production_rates * gas.molecular_weights
        cvm = gas.cv_mass

        wsensT = (wp - wm) / (2. * dT)
        cvsensT = (cvp - cvm) / (2. * dT)

        jacFD11 = np.copy(jacFD[1, 1])
        jacSemiFD11 = - 1. / cv * (1. / rho * (sum(wsensT * e) + sum(cvi * w)) + cvsensT * rhsGR[1])
        pass_jac = max(abs(jacGR - jacFD) / (abs(jacGR) + 1.)) < 1.e-3

        if not pass_jac:
            print('fd:')
            for i in range(ns + 1):
                for j in range(ns + 1):
                    print(f'{jacFD[i, j]:12.2e}', end=', ')
                print('')
            print('gr:')
            for i in range(ns + 1):
                for j in range(ns + 1):
                    print(f'{jacGR[i, j]:12.2e}', end=', ')
                print('')
            print('gr-fd:')
            for i in range(ns + 1):
                for j in range(ns + 1):
                    df = (jacGR[i, j] - jacFD[i, j]) / (abs(jacFD[i, j]) + 1.0)
                    if df > 1.e-3:
                        print(f'{df:12.2e}', end=', ')
                    else:
                        print(f'{"":16}', end=', ')
                print('')
            print('')

    if test_rhs:
        return pass_rhs
    if test_jac:
        return pass_jac


def create_test(m, T, p, test_rhs, test_jac):
    def test(self):
        self.assertTrue(validate_on_mechanism(m, T, p, test_rhs, test_jac))

    return test


class Accuracy(unittest.TestCase):
    pass


temperature_dict = {'600K': 600., '1200K': 1200.}
pressure_dict = {'1atm': one_atm, '2atm': 2. * one_atm}
for mech in mechs:
    for temperature in temperature_dict:
        for pressure in pressure_dict:
            rhsname = 'test_rhs_' + mech + '_' + temperature + '_' + pressure
            jacname = 'test_jac_' + mech + '_' + temperature + '_' + pressure
            jsdname = 'test_jac_sparse_vs_dense_' + mech + '_' + temperature + '_' + pressure
            setattr(Accuracy, rhsname, create_test(mech, temperature_dict[temperature], pressure_dict[pressure],
                                                   test_rhs=True, test_jac=False))
            if 'methane' not in mech:  # skip methane in the finite difference Jacobian test
                setattr(Accuracy, jacname, create_test(mech, temperature_dict[temperature], pressure_dict[pressure],
                                                       test_rhs=False, test_jac=True))

if __name__ == '__main__':
    unittest.main()
