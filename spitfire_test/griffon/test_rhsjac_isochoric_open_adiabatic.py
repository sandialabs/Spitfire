import unittest
from numpy import hstack, max, abs, sqrt
from cantera import Solution, gas_constant
import numpy as np
from spitfire import ChemicalMechanismSpec
from os.path import join, abspath
from subprocess import getoutput

test_mech_directory = abspath(join('spitfire_test', 'test_mechanisms', 'old_xmls'))
mechs = [x.replace('.xml', '') for x in getoutput('ls ' + test_mech_directory + ' | grep .xml').split('\n')]


def rhs_cantera(p_arg, T_arg, y_arg, rhoin, Tin_arg, yin_arg, tau_arg, gas, rhs_chem_in):
    gas.TPY = T_arg, p_arg, y_arg
    rho = gas.density_mass
    cv = gas.cv_mass
    e = gas.standard_int_energies_RT * gas.T * gas_constant / gas.molecular_weights

    gas.TDY = Tin_arg, rhoin, yin_arg
    ein = gas.standard_int_energies_RT * gas.T * gas_constant / gas.molecular_weights

    rhs = np.copy(rhs_chem_in)
    rhsMass = np.zeros(gas.n_species + 1)
    rhsMass[0] += (rhoin - rho)
    rhsMass[1] += 1. / (rho * cv) * (rhoin * np.sum(yin_arg * (ein - e)))
    rhsMass[2:] += rhoin / rho * (yin_arg[:-1] - y_arg[:-1])
    rhs += rhsMass / tau_arg
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
    rho = gas.density_mass

    xin = np.ones(ns)  # equal moles in the feed
    gas.TPX = Tin, p, xin
    yin = np.copy(gas.Y)
    rhoin = gas.density_mass

    state = hstack((rho, T, y[:-1]))

    rhsGRChemOnly = np.zeros(ns + 1)
    r.reactor_rhs_isochoric(state, rhoin, Tin, yin, tau, 0, 0, 0, 0, 0, 0, False, rhsGRChemOnly)
    rhsCN = rhs_cantera(p, T, y, rhoin, Tin, yin, tau, gas, rhsGRChemOnly)
    rhsGR = np.empty(ns + 1)
    r.reactor_rhs_isochoric(state, rhoin, Tin, yin, tau, 0, 0, 0, 0, 0, 0, True, rhsGR)

    if do_rhs:
        return max(abs(rhsGR - rhsCN) / (abs(rhsCN) + 1.)) < 100. * sqrt(np.finfo(float).eps)

    if do_jac:
        jacGR = np.empty((ns + 1) * (ns + 1))
        r.reactor_jac_isochoric(state, rhoin, Tin, yin, tau, 0, 0, 0, 0, 0, 0, True, 0, rhsGR, jacGR)
        jacGR = jacGR.reshape((ns + 1, ns + 1), order='F')

        drho = 1.e-6
        dT = 1.e-6
        dY = 1.e-6
        jacFD = np.empty((ns + 1, ns + 1))
        rhsGR1, rhsGR2 = np.empty(ns + 1), np.empty(ns + 1)

        state_m = hstack((rho - drho, T, y[:-1]))
        state_p = hstack((rho + drho, T, y[:-1]))
        r.reactor_rhs_isochoric(state_m, rhoin, Tin, yin, tau, 0, 0, 0, 0, 0, 0, True, rhsGR1)
        r.reactor_rhs_isochoric(state_p, rhoin, Tin, yin, tau, 0, 0, 0, 0, 0, 0, True, rhsGR2)
        jacFD[:, 0] = (- rhsGR1 + rhsGR2) / (2. * drho)

        state_m = hstack((rho, T - dT, y[:-1]))
        state_p = hstack((rho, T + dT, y[:-1]))
        r.reactor_rhs_isochoric(state_m, rhoin, Tin, yin, tau, 0, 0, 0, 0, 0, 0, True, rhsGR1)
        r.reactor_rhs_isochoric(state_p, rhoin, Tin, yin, tau, 0, 0, 0, 0, 0, 0, True, rhsGR2)
        jacFD[:, 1] = (- rhsGR1 + rhsGR2) / (2. * dT)

        for i in range(ns - 1):
            y_m1, y_p1 = np.copy(y), np.copy(y)
            y_m1[i] += - dY
            y_m1[-1] -= - dY
            y_p1[i] += dY
            y_p1[-1] -= dY
            state_m = hstack((rho, T, y_m1[:-1]))
            state_p = hstack((rho, T, y_p1[:-1]))
            r.reactor_rhs_isochoric(state_m, rhoin, Tin, yin, tau, 0, 0, 0, 0, 0, 0, True, rhsGR1)
            r.reactor_rhs_isochoric(state_p, rhoin, Tin, yin, tau, 0, 0, 0, 0, 0, 0, True, rhsGR2)
            jacFD[:, 2 + i] = (- rhsGR1 + rhsGR2) / (2. * dY)

        return max(abs(jacGR - jacFD) / (abs(jacGR) + 1.)) < 1.e-4


def create_test(m, T, p, tau, do_rhs, do_jac):
    def test(self):
        self.assertTrue(validate_on_mechanism(m, T, p, tau, do_rhs, do_jac))

    return test


class Accuracy(unittest.TestCase):
    pass


tau_list = [1.e-6, 1.e-3]
for mech in mechs:
    for tau in tau_list:
        rhsname = 'test_rhs_' + mech + '_' + 'tau=' + str(tau)
        jacname = 'test_jac_' + mech + '_' + 'tau=' + str(tau)
        setattr(Accuracy, rhsname, create_test(mech, 600., 101325, tau, True, False))
        if 'methane' not in mech:  # skip methane in the finite difference Jacobian spitfire_test
            setattr(Accuracy, jacname, create_test(mech, 600., 101325, tau, False, True))

if __name__ == '__main__':
    unittest.main()
