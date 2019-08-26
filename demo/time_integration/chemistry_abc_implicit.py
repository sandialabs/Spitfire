"""
Simple chemical kinetics ('ABC') demonstration

This code demonstrates how to use Spitfire to solve an ODE system governing a simple chemical kinetic system,
with the following chemical reactions.

1: A -> B
2: A + B -> 2C

This demo shows how to use a custom linear solver routine for advanced implicit methods.
"""

"""
Copyright (c) 2018-2019 Michael Alan Hansen - All Rights Reserved
You may use, distribute and modify this code under the terms of the MIT license.

You should have received a copy of the MIT license with this file.
If not, please write to mahanse@sandia.gov or mike.hansen@chemeng.utah.edu
"""

from spitfire.time.governor import Governor, FinalTime, SaveAllDataToList
from spitfire.time.methods import ESDIRK64, ExplicitRungeKutta4Classical
from spitfire.time.nonlinear import SimpleNewtonSolver
from scipy.linalg.lapack import dgetrf as lapack_lu_factor
from scipy.linalg.lapack import dgetrs as lapack_lu_solve
import scipy.sparse.linalg as spla
import numpy as np
import matplotlib.pyplot as plt


class ChemistryProblem(object):
    """
    This class defines the right-hand side, setup, and solve methods for implicit methods with custom linear solvers
    """

    def __init__(self, k_ab, k_bc):
        self._k_ab = k_ab
        self._k_bc = k_bc
        self._lhs_inverse_op = None
        self._identity_matrix = np.eye(3)
        self._gmres_iter = 0

    def rhs(self, t, c):
        c_a = c[0]
        c_b = c[1]
        q_1 = self._k_ab * c_a
        q_2 = self._k_bc * c_a * c_b
        return np.array([-q_1 - q_2,
                         q_1 - q_2,
                         2. * q_2])

    def setup_lapack_lu(self, c, prefactor):
        c_a = c[0]
        c_b = c[1]
        dq1_da = self._k_ab
        dq1_db = 0.
        dq1_dc = 0.
        dq2_da = self._k_bc * c_b
        dq2_db = self._k_bc * c_a
        dq2_dc = 0.
        J = np.array([[-dq1_da - dq2_da, -dq1_db - dq2_db, -dq1_dc - dq2_dc],
                      [dq1_da - dq2_da, dq1_db - dq2_db, dq1_dc - dq2_dc],
                      [2. * dq2_da, 2. * dq2_db, 2. * dq2_dc]])

        linear_op = prefactor * J - self._identity_matrix

        self._lhs_inverse_op = lapack_lu_factor(linear_op)[:2]  # this [:2] is just an implementation detail of scipy

    def solve_lapack_lu(self, residual):
        return lapack_lu_solve(self._lhs_inverse_op[0],
                               self._lhs_inverse_op[1],
                               residual)[0], 1, True  # the , 1, True parts are how many iterations and success/failure

    def setup_diagonal(self, c, prefactor):
        c_a = c[0]
        c_b = c[1]
        dq1_da = self._k_ab
        dq1_db = 0.
        dq1_dc = 0.
        dq2_da = self._k_bc * c_b
        dq2_db = self._k_bc * c_a
        dq2_dc = 0.
        J = np.array([[-dq1_da - dq2_da, -dq1_db - dq2_db, -dq1_dc - dq2_dc],
                      [dq1_da - dq2_da, dq1_db - dq2_db, dq1_dc - dq2_dc],
                      [2. * dq2_da, 2. * dq2_db, 2. * dq2_dc]])

        linear_op = prefactor * J - self._identity_matrix

        self._lhs_inverse_op = 1. / linear_op.diagonal()

    def solve_diagonal(self, residual):
        return self._lhs_inverse_op * residual, 1, True  # the , 1, True parts are how many iterations and success/failure

    def setup_gmres(self, c, prefactor):
        c_a = c[0]
        c_b = c[1]
        dq1_da = self._k_ab
        dq1_db = 0.
        dq1_dc = 0.
        dq2_da = self._k_bc * c_b
        dq2_db = self._k_bc * c_a
        dq2_dc = 0.
        J = np.array([[-dq1_da - dq2_da, -dq1_db - dq2_db, -dq1_dc - dq2_dc],
                      [dq1_da - dq2_da, dq1_db - dq2_db, dq1_dc - dq2_dc],
                      [2. * dq2_da, 2. * dq2_db, 2. * dq2_dc]])

        self._linear_op = prefactor * J - self._identity_matrix

        self._lhs_op = spla.LinearOperator((3, 3), lambda x: self._linear_op.dot(x))
        self._jacobi_preconditioner = spla.LinearOperator((3, 3), lambda res: 1. / self._linear_op.diagonal() * res)

    def _increment_gmres_iter(self, *args, **kwargs):
        self._gmres_iter += 1

    def solve_gmres(self, residual):
        x, i = spla.gmres(self._lhs_op,
                          residual,
                          M=self._jacobi_preconditioner,
                          tol=1.e-8,
                          callback=self._increment_gmres_iter)
        return x, self._gmres_iter, not i


c0 = np.array([1., 0., 0.])  # initial condition
k_ab = 1.  # A -> B rate constant
k_bc = 0.2  # A + B -> 2C rate constant

problem = ChemistryProblem(k_ab, k_bc)

final_time = 10.  # final time to integrate to
time_step_size = 0.1  # size of the time step used

governor = Governor()
governor.termination_criteria = FinalTime(final_time)

data = SaveAllDataToList(initial_solution=c0)
governor.custom_post_process_step = data.save_data

governor.integrate(right_hand_side=problem.rhs,
                   initial_condition=c0,
                   controller=10. * time_step_size,
                   method=ESDIRK64(SimpleNewtonSolver()),
                   projector_setup=problem.setup_lapack_lu,
                   projector_solve=problem.solve_lapack_lu)

plt.plot(data.t_list, data.solution_list[:, 0], 'b-', label='A (imp LU, 10h)')
plt.plot(data.t_list, data.solution_list[:, 1], 'r-', label='B (imp LU, 10h)')
plt.plot(data.t_list, data.solution_list[:, 2], 'g-', label='C (imp LU, 10h)')

data.reset_data(initial_solution=c0)
governor.integrate(right_hand_side=problem.rhs,
                   initial_condition=c0,
                   controller=10. * time_step_size,
                   method=ESDIRK64(SimpleNewtonSolver()),
                   projector_setup=problem.setup_diagonal,
                   projector_solve=problem.solve_diagonal)

plt.plot(data.t_list, data.solution_list[:, 0], 'b--', label='A (imp diag, 10h)')
plt.plot(data.t_list, data.solution_list[:, 1], 'r--', label='B (imp diag, 10h)')
plt.plot(data.t_list, data.solution_list[:, 2], 'g--', label='C (imp diag, 10h)')

data.reset_data(initial_solution=c0)
governor.integrate(right_hand_side=problem.rhs,
                   initial_condition=c0,
                   controller=10. * time_step_size,
                   method=ESDIRK64(SimpleNewtonSolver()),
                   projector_setup=problem.setup_gmres,
                   projector_solve=problem.solve_gmres)

plt.plot(data.t_list, data.solution_list[:, 0], 'b--', label='A (imp diag, 10h)')
plt.plot(data.t_list, data.solution_list[:, 1], 'r--', label='B (imp diag, 10h)')
plt.plot(data.t_list, data.solution_list[:, 2], 'g--', label='C (imp diag, 10h)')

data.reset_data(initial_solution=c0)
governor.integrate(right_hand_side=problem.rhs,
                   initial_condition=c0,
                   controller=time_step_size,
                   method=ExplicitRungeKutta4Classical())

plt.plot(data.t_list, data.solution_list[:, 0], 'b-.', label='A (exp, h)')
plt.plot(data.t_list, data.solution_list[:, 1], 'r-.', label='B (exp, h)')
plt.plot(data.t_list, data.solution_list[:, 2], 'g-.', label='C (exp, h)')

plt.grid()
plt.legend()
plt.xlabel('t')
plt.ylabel('concentration')
plt.show()
