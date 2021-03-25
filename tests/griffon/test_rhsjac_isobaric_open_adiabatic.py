import unittest
from numpy import hstack, max, abs, zeros, sum
from cantera import Solution, gas_constant
import numpy as np
from spitfire import ChemicalMechanismSpec
from os.path import join, abspath
from subprocess import getoutput

test_mech_directory = abspath(join('tests', 'test_mechanisms', 'old_xmls'))
mechs = [x.replace('.xml', '') for x in getoutput('ls ' + test_mech_directory + ' | grep .xml').split('\n')]


def rhs_cantera(p_arg, T_arg, y_arg, Tin_arg, yin_arg, tau_arg, gas):
    gas.TPY = T_arg, p_arg, y_arg
    rho = gas.density_mass
    cp = gas.cp_mass
    w = gas.net_production_rates * gas.molecular_weights
    h = gas.standard_enthalpies_RT * gas.T * gas_constant / gas.molecular_weights

    gas.TPY = Tin_arg, p_arg, yin_arg
    hin = gas.standard_enthalpies_RT * gas.T * gas_constant / gas.molecular_weights

    rhs = zeros(gas.n_species)
    rhs[0] = - sum(w * h) / rho / cp + 1. / tau_arg * 1. / cp * np.sum(yin_arg * (hin - h))
    rhs[1:] = w[:-1] / rho + 1. / tau_arg * (yin_arg[:-1] - y_arg[:-1])

    return rhs


def validate_on_mechanism(mech, temperature, pressure, tau, do_rhs, do_jac):
    xml = join(test_mech_directory, mech + '.xml')
    T = temperature
    Tin = T + 1000.
    p = pressure

    r = ChemicalMechanismSpec(xml, 'gas').griffon

    gas = Solution(xml)
    ns = gas.n_species

    y = np.ones(ns)  # equal masses in the reactor
    gas.TPY = T, p, y
    y = np.copy(gas.Y)

    xin = np.ones(ns)  # equal moles in the feed
    gas.TPX = Tin, p, xin
    yin = np.copy(gas.Y)

    state = hstack((T, y[:-1]))

    rhsCN = rhs_cantera(p, T, y, Tin, yin, tau, gas)
    rhsGR = np.empty(ns)
    r.reactor_rhs_isobaric(state, p, Tin, yin, tau, 0, 0, 0, 0, 0, 0, True, rhsGR)

    if do_rhs:
        return max(abs(rhsGR - rhsCN) / (abs(rhsCN) + 1.)) < 1.e-4

    if do_jac:
        jacGR = np.empty(ns * ns)
        r.reactor_jac_isobaric(state, p, Tin, yin, tau, 0, 0, 0, 0, 0, 0, True, 0, 0, rhsGR, jacGR)
        jacGR = jacGR.reshape((ns, ns), order='F')

        dT = 1.e-6
        dY = 1.e-6
        jacFD = np.empty((ns, ns))
        rhsGR1, rhsGR2 = np.empty(ns), np.empty(ns)
        state_m = hstack((T - dT, y[:-1]))
        state_p = hstack((T + dT, y[:-1]))
        r.reactor_rhs_isobaric(state_m, p, Tin, yin, tau, 0, 0, 0, 0, 0, 0, True, rhsGR1)
        r.reactor_rhs_isobaric(state_p, p, Tin, yin, tau, 0, 0, 0, 0, 0, 0, True, rhsGR2)
        jacFD[:, 0] = (- rhsGR1 + rhsGR2) / (2. * dT)

        for i in range(ns - 1):
            y_m1, y_p1 = np.copy(y), np.copy(y)
            y_m1[i] += - dY
            y_m1[-1] -= - dY
            y_p1[i] += dY
            y_p1[-1] -= dY
            state_m = hstack((T, y_m1[:-1]))
            state_p = hstack((T, y_p1[:-1]))
            r.reactor_rhs_isobaric(state_m, p, Tin, yin, tau, 0, 0, 0, 0, 0, 0, True, rhsGR1)
            r.reactor_rhs_isobaric(state_p, p, Tin, yin, tau, 0, 0, 0, 0, 0, 0, True, rhsGR2)
            jacFD[:, 1 + i] = (- rhsGR1 + rhsGR2) / (2. * dY)

        pass_jac = max(abs(jacGR - jacFD) / (abs(jacGR) + 1.)) < 1.e-2

        return pass_jac


def create_test(m, T, p, tau, do_rhs, do_jac):
    def test(self):
        self.assertTrue(validate_on_mechanism(m, T, p, tau, do_rhs, do_jac))

    return test


class Accuracy(unittest.TestCase):
    pass


tau_list = [1.e-8, 1.e-6, 1.e-3]
for mech in mechs:
    for tau in tau_list:
        rhsname = 'test_rhs_' + mech + '_' + 'tau=' + str(tau)
        jacname = 'test_jac_' + mech + '_' + 'tau=' + str(tau)
        setattr(Accuracy, rhsname, create_test(mech, 600., 101325, tau, True, False))
        setattr(Accuracy, jacname, create_test(mech, 600., 101325, tau, False, True))

if __name__ == '__main__':
    unittest.main()
