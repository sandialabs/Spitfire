import unittest
from numpy import ones, hstack
from cantera import Solution, one_atm
import numpy as np
from spitfire import ChemicalMechanismSpec
from os.path import join, abspath
from subprocess import getoutput

test_mech_directory = abspath(join('tests', 'test_mechanisms', 'old_xmls'))
mechs = [x.replace('.yaml', '') for x in getoutput('ls ' + test_mech_directory + ' | grep .yaml').split('\n')]


def validate_on_mechanism(mech, temperature, pressure, full_Jacobian, isochoric):
    xml = join(test_mech_directory, mech + '.yaml')

    r = ChemicalMechanismSpec(xml, 'gas').griffon

    gas = Solution(xml)
    ns = gas.n_species

    T = temperature
    p = pressure

    gas.TPX = T, p, ones(ns)
    y = gas.Y
    rho = gas.density_mass

    if full_Jacobian and isochoric:
        state = hstack((rho, T, y[:-1]))
        rhsGRTemporary = np.empty(ns + 1)
        jac_dense = np.empty((ns + 1) * (ns + 1))
        jac_sparse = np.empty((ns + 1) * (ns + 1))
        r.reactor_jac_isochoric(state, 0, 0, np.ndarray(1), 0, 0, 0, 0, 0, 0, 0, False, 0, rhsGRTemporary, jac_dense)
        r.reactor_jac_isochoric(state, 0, 0, np.ndarray(1), 0, 0, 0, 0, 0, 0, 0, False, 2, rhsGRTemporary, jac_sparse)
        jac_dense = jac_dense.reshape((ns + 1, ns + 1), order='F')
        jac_sparse = jac_sparse.reshape((ns + 1, ns + 1), order='F')

    elif full_Jacobian and not isochoric:
        state = hstack((T, y[:-1]))
        k = np.empty(ns)
        jac_dense = np.empty(ns * ns)
        jac_sparse = np.empty(ns * ns)
        r.reactor_jac_isobaric(state, p, 0, np.ndarray(1), 0, 0, 0, 0, 0, 0, 0, False, 0, 0, k, jac_dense)
        r.reactor_jac_isobaric(state, p, 0, np.ndarray(1), 0, 0, 0, 0, 0, 0, 0, False, 2, 0, k, jac_sparse)

        jac_dense = jac_dense.reshape((ns, ns), order='F')
        jac_sparse = jac_sparse.reshape((ns, ns), order='F')

    else:
        jac_dense = np.zeros((ns + 1) * (ns + 1))
        jac_sparse = np.zeros((ns + 1) * (ns + 1))
        r.prod_rates_primitive_sensitivities(rho, T, y, 0, jac_dense)
        r.prod_rates_primitive_sensitivities(rho, T, y, 2, jac_sparse)
        jac_dense = jac_dense.reshape((ns + 1, ns + 1), order='F')[:-1, :]
        jac_sparse = jac_sparse.reshape((ns + 1, ns + 1), order='F')[:-1, :]

        jac_fd = np.zeros_like(jac_dense)
        rhsGR1 = np.zeros(ns)
        rhsGR2 = np.zeros(ns)
        dT = 1.e-4
        dr = 1.e-4
        dY = 1.e-4
        r.production_rates(T, rho + dr, y, rhsGR1)
        r.production_rates(T, rho - dr, y, rhsGR2)
        jac_fd[:, 0] = (rhsGR1 - rhsGR2) / dr * 0.5
        r.production_rates(T + dT, rho, y, rhsGR1)
        r.production_rates(T - dT, rho, y, rhsGR2)
        jac_fd[:, 1] = (rhsGR1 - rhsGR2) / dT * 0.5

        for spec_idx in range(ns - 1):
            Yp = np.copy(y)
            Yp[spec_idx] += dY
            Yp[-1] -= dY
            Ym = np.copy(y)
            Ym[spec_idx] -= dY
            Ym[-1] += dY
            r.production_rates(T, rho, Yp, rhsGR1)
            r.production_rates(T, rho, Ym, rhsGR2)
            jac_fd[:, 2 + spec_idx] = (rhsGR1 - rhsGR2) / dY * 0.5

    pass_sparse_vs_dense_jac = np.linalg.norm(jac_dense.ravel() - jac_sparse.ravel(), ord=np.Inf) < 1.e-10

    # if not pass_sparse_vs_dense_jac:
    #     print(mech)
    #     diff = abs(jac_dense - jac_sparse) / (abs(jac_dense) + 1.)
    #     nr, nc = jac_dense.shape
    #     print('dense')
    #     for ir in range(nr):
    #         for ic in range(nc):
    #             print(f'{jac_dense[ir, ic]:8.0e}', end=', ')
    #         print('')
    #     print('sparse')
    #     for ir in range(nr):
    #         for ic in range(nc):
    #             print(f'{jac_sparse[ir, ic]:8.0e}', end=', ')
    #         print('')
    #     print('finite difference')
    #     for ir in range(nr):
    #         for ic in range(nc):
    #             print(f'{jac_fd[ir, ic]:8.0e}', end=', ')
    #         print('')
    #     print('diff')
    #     for ir in range(nr):
    #         for ic in range(nc):
    #             print(f'{diff[ir, ic]:8.0e}' if diff[ir, ic] > 1.e-12 else f'{"":8}', end=', ')
    #         print('')
    return pass_sparse_vs_dense_jac


def create_test(m, T, p, full_Jacobian, isochoric=True):
    def test(self):
        self.assertTrue(validate_on_mechanism(m, T, p, full_Jacobian, isochoric))

    return test


class Accuracy(unittest.TestCase):
    pass


temperature_dict = {'600K': 600., '1200K': 1200.}
pressure_dict = {'1atm': one_atm, '2atm': 2. * one_atm}
for mech in mechs:
    for temperature in temperature_dict:
        for pressure in pressure_dict:
            jsdname = 'test_jac_sparse_vs_dense_isochoric_' + mech + '_' + temperature + '_' + pressure
            setattr(Accuracy, jsdname, create_test(mech, temperature_dict[temperature], pressure_dict[pressure],
                                                   full_Jacobian=True, isochoric=True))
            jsdname = 'test_jac_sparse_vs_dense_isobaric_' + mech + '_' + temperature + '_' + pressure
            setattr(Accuracy, jsdname, create_test(mech, temperature_dict[temperature], pressure_dict[pressure],
                                                   full_Jacobian=True, isochoric=False))
            jsdname = 'test_jac_sparse_vs_dense_sensitivites_' + mech + '_' + temperature + '_' + pressure
            setattr(Accuracy, jsdname, create_test(mech, temperature_dict[temperature], pressure_dict[pressure],
                                                   full_Jacobian=False))

if __name__ == '__main__':
    unittest.main()
