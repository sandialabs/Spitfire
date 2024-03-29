import pickle


def run():
    from spitfire import (odesolve,
                          BackwardEulerS1P1Q1,
                          KennedyCarpenterS6P4Q3,
                          KvaernoS4P3Q2,
                          KennedyCarpenterS4P3Q2,
                          KennedyCarpenterS8P5Q4, )
    from spitfire.time.nonlinear import SimpleNewtonSolver
    from scipy.linalg.lapack import dgetrf as lapack_lu_factor
    from scipy.linalg.lapack import dgetrs as lapack_lu_solve
    import scipy.sparse.linalg as spla
    import numpy as np

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

        def setup_lapack_lu(self, t, c, prefactor):
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

            self._lhs_inverse_op = lapack_lu_factor(linear_op)[
                                   :2]  # this [:2] is just an implementation detail of scipy

        def solve_lapack_lu(self, residual):
            return lapack_lu_solve(self._lhs_inverse_op[0],
                                   self._lhs_inverse_op[1],
                                   residual)[
                       0], 1, True  # the , 1, True parts are how many iterations and success/failure

        def setup_diagonal(self, t, c, prefactor):
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

        def setup_gmres(self, t, c, prefactor):
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
            self._jacobi_preconditioner = spla.LinearOperator((3, 3),
                                                              lambda res: 1. / self._linear_op.diagonal() * res)

        def _increment_gmres_iter(self, *args, **kwargs):
            self._gmres_iter += 1

        def solve_gmres(self, residual):
            x, i = spla.gmres(self._lhs_op,
                              residual,
                              M=self._jacobi_preconditioner,
                              atol=1.e-8,
                              callback=self._increment_gmres_iter,
                              callback_type='legacy')
            return x, self._gmres_iter, not i

    c0 = np.array([1., 0., 0.])  # initial condition
    k_ab = 1.  # A -> B rate constant
    k_bc = 0.2  # A + B -> 2C rate constant

    problem = ChemistryProblem(k_ab, k_bc)

    final_time = 10.  # final time to integrate to

    sol_dict = dict()

    for method in [BackwardEulerS1P1Q1(SimpleNewtonSolver()),
                   KennedyCarpenterS6P4Q3(SimpleNewtonSolver()),
                   KvaernoS4P3Q2(SimpleNewtonSolver()),
                   KennedyCarpenterS4P3Q2(SimpleNewtonSolver()),
                   KennedyCarpenterS8P5Q4(SimpleNewtonSolver())]:
        t, sol = odesolve(problem.rhs, c0, stop_at_time=final_time,
                          method=method,
                          linear_setup=problem.setup_lapack_lu,
                          linear_solve=problem.solve_lapack_lu,
                          save_each_step=True)
        sol_dict['lapack-' + method.name] = (t.copy(), sol.copy())

        t, sol = odesolve(problem.rhs, c0, stop_at_time=final_time,
                          method=method,
                          linear_setup=problem.setup_diagonal,
                          linear_solve=problem.solve_diagonal,
                          save_each_step=True)
        sol_dict['diagonal-' + method.name] = (t.copy(), sol.copy())

        t, sol = odesolve(problem.rhs, c0, stop_at_time=final_time,
                          method=method,
                          linear_setup=problem.setup_gmres,
                          linear_solve=problem.solve_gmres,
                          save_each_step=True)
        sol_dict['gmres-' + method.name] = (t.copy(), sol.copy())

    return sol_dict


if __name__ == '__main__':
    output = run()
    with open('gold.pkl', 'wb') as file_output:
        pickle.dump(output, file_output)
