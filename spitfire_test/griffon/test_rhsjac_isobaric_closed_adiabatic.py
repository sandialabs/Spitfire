import unittest
from numpy import hstack, max, abs, ones, zeros, sum, sqrt
from cantera import Solution, one_atm, gas_constant
import numpy as np
from spitfire import ChemicalMechanismSpec
from os.path import join, abspath
from subprocess import getoutput

test_mech_directory = abspath(join('spitfire_test', 'test_mechanisms', 'old_xmls'))
mechs = [x.replace('.xml', '') for x in getoutput('ls ' + test_mech_directory + ' | grep .xml').split('\n')]


def validate_on_mechanism(mech, temperature, pressure, test_rhs=True, test_jac=True):
    xml = join(test_mech_directory, mech + '.xml')

    T = temperature
    p = pressure

    r = ChemicalMechanismSpec(xml, 'gas').griffon

    gas = Solution(xml)
    ns = gas.n_species

    gas.TPX = T, p, ones(ns)
    y = gas.Y

    state = hstack((T, y[:-1]))

    rhsGR = np.empty(ns)
    r.reactor_rhs_isobaric(state, p, 0., np.ndarray(1), 0, 0, 0, 0, 0, 0, 0, False, rhsGR)

    if test_jac:
        Tin, yin, tau = 0, np.ndarray(1), 0
        rhsTmp = np.empty(ns)
        jacGR = np.empty(ns * ns)
        r.reactor_jac_isobaric(state, p, Tin, yin, tau, 0, 0, 0, 0, 0, 0, False, 0, 0, rhsTmp, jacGR)
        jacGR = jacGR.reshape((ns, ns), order='F')

        dT = 1.e-6
        dY = 1.e-6
        jacFD = np.empty((ns, ns))
        rhsGR1, rhsGR2 = np.empty(ns), np.empty(ns)
        state_m = hstack((T - dT, y[:-1]))
        state_p = hstack((T + dT, y[:-1]))
        r.reactor_rhs_isobaric(state_m, p, Tin, yin, tau, 0, 0, 0, 0, 0, 0, False, rhsGR1)
        r.reactor_rhs_isobaric(state_p, p, Tin, yin, tau, 0, 0, 0, 0, 0, 0, False, rhsGR2)
        jacFD[:, 0] = (- rhsGR1 + rhsGR2) / (2. * dT)

        for i in range(ns - 1):
            y_m1, y_p1 = np.copy(y), np.copy(y)
            y_m1[i] += - dY
            y_m1[-1] -= - dY
            y_p1[i] += dY
            y_p1[-1] -= dY
            state_m = hstack((T, y_m1[:-1]))
            state_p = hstack((T, y_p1[:-1]))
            r.reactor_rhs_isobaric(state_m, p, Tin, yin, tau, 0, 0, 0, 0, 0, 0, False, rhsGR1)
            r.reactor_rhs_isobaric(state_p, p, Tin, yin, tau, 0, 0, 0, 0, 0, 0, False, rhsGR2)
            jacFD[:, 1 + i] = (- rhsGR1 + rhsGR2) / (2. * dY)

        pass_jac = max(abs(jacGR - jacFD) / (abs(jacGR) + 1.)) < 1.e-2

    w = gas.net_production_rates * gas.molecular_weights
    h = gas.standard_enthalpies_RT * gas.T * gas_constant / gas.molecular_weights
    rhsCN = zeros(ns)
    rhsCN[1:] = w[:-1] / gas.density
    rhsCN[0] = - sum(w * h) / gas.density / gas.cp_mass
    pass_rhs = max(abs(rhsGR - rhsCN) / (abs(rhsCN) + 1.)) < 100. * sqrt(np.finfo(float).eps)

    if test_rhs and test_jac:
        return pass_rhs and pass_jac
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
            setattr(Accuracy, rhsname, create_test(mech, temperature_dict[temperature], pressure_dict[pressure],
                                                   test_rhs=True, test_jac=False))
            setattr(Accuracy, jacname, create_test(mech, temperature_dict[temperature], pressure_dict[pressure],
                                                   test_rhs=False, test_jac=True))

if __name__ == '__main__':
    unittest.main()
