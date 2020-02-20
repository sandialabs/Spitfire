# Spitfire - a Python-C++ library for building tabulated chemistry models and solving differential equations
# Copyright 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
#
# You should have received a copy of the 3-clause BSD License
# along with this program.  If not, see <https://opensource.org/licenses/BSD-3-Clause>.
#
# Questions? Contact Mike Hansen (mahanse@sandia.gov)

import unittest
from spitfire.time.governor import Governor, Steady, FinalTime
from spitfire.time.methods import ESDIRK64, AdaptiveERK54CashKarp, ForwardEuler
from spitfire.time.nonlinear import SimpleNewtonSolver
from spitfire.time.stepcontrol import PIController
import numpy as np
from scipy.sparse import csc_matrix, diags, block_diag
from scipy.sparse import eye as speye
from scipy.sparse.linalg import splu as superlu_factor


class DiffusionReaction1D_FiniteDifference(object):
    _grid_types = ['uniform', 'clustered']

    @classmethod
    def _uniform_grid(cls, grid_points):
        z = np.linspace(0., 1., grid_points)
        dz = z[1:] - z[:-1]
        return z, dz

    @classmethod
    def _clustered_grid(cls, grid_points, grid_cluster_point, grid_cluster_intensity=6.):

        if grid_cluster_intensity < 1.e-16:
            raise ValueError('cluster_coeff must be strictly positive! Given value: ' + str(grid_cluster_intensity))

        if grid_cluster_point < 0. or grid_cluster_point > 1.:
            raise ValueError('z_cluster must be between 0 and 1! Given value: ' + str(grid_cluster_point))

        z = np.linspace(0., 1., grid_points)
        zo = 1.0 / (2.0 * grid_cluster_intensity) * np.log(
            (1. + (np.exp(grid_cluster_intensity) - 1.) * grid_cluster_point) / (
                1. + (np.exp(-grid_cluster_intensity) - 1.) * grid_cluster_point))
        a = np.sinh(grid_cluster_intensity * zo)
        for i in range(grid_points):
            z[i] = grid_cluster_point / a * (np.sinh(grid_cluster_intensity * (z[i] - zo)) + a)
        z[-1] = 1.
        dz = z[1:] - z[:-1]
        return z, dz

    def __init__(self,
                 initial_conditions,
                 diffusion_coefficient,
                 source_term,
                 source_term_jacobian=None,
                 left_boundary_state=None,
                 right_boundary_state=None,
                 lewis_numbers=None,
                 variable_names=None,
                 grid=None,  # provide the grid directly OR
                 grid_points=None,  # provide number of grid points, AND
                 grid_type='uniform',  # OPTIONAL provide type of grid, AND if clustered
                 grid_cluster_intensity=2.,  # OPTIONAL provide cluster intensity, AND
                 grid_cluster_point=0.5):  # OPTIONAL provide clustering point

        self._n_equations = len(initial_conditions)

        if variable_names is None:
            self._variable_names = []
            for i in range(self._n_equations):
                self._variable_names.append('variable ' + str(i))
        else:
            self._variable_names = variable_names

        if grid is not None:
            self._x = np.copy(grid)
            self._dx = self._x[1:] - self._x[:-1]

            warning_message = lambda arg: 'Flamelet specifications: Warning! Setting the grid argument ' \
                                          'nullifies the ' + arg + ' argument.'
            if grid_points is not None:
                print(warning_message('grid_points'))
            if grid_type is not None:
                print(warning_message('grid_type'))
            if grid_cluster_intensity is not None:
                print(warning_message('grid_cluster_intensity'))
            if grid_cluster_point is not None:
                print(warning_message('grid_cluster_point'))
        else:
            if grid_type == 'uniform':
                self._x, self._dx = self._uniform_grid(grid_points)
            elif grid_type == 'clustered':
                if grid_cluster_point == 'stoichiometric':
                    grid_cluster_point = self._mechanism.stoich_mixture_fraction(self._fuel_stream, self._oxy_stream)
                self._x, self._dx = self._clustered_grid(grid_points, grid_cluster_point, grid_cluster_intensity)
            else:
                error_message = 'Flamelet specifications: Bad grid_type argument detected: ' + grid_type + '\n' + \
                                '                         Acceptable values: ' + self._grid_types
                raise ValueError(error_message)

        self._nx_interior = self._x.size - 2
        self._n_dof = self._n_equations * self._nx_interior

        if callable(diffusion_coefficient):
            self._d = diffusion_coefficient(self._x[1:-1])
        else:
            self._d = np.copy(diffusion_coefficient)
        self._lewis_numbers = lewis_numbers if lewis_numbers is not None else np.ones(self._n_equations)

        self._initial_state = np.zeros(self._n_dof)
        for offset, initial_condition in enumerate(initial_conditions):
            if callable(initial_condition):
                self._initial_state[offset::self._n_equations] = initial_condition(self._x[1:-1])
            else:
                self._initial_state[offset::self._n_equations] = initial_condition

        self._left_bc = np.zeros(self._n_equations)
        self._right_bc = np.zeros(self._n_equations)
        for bc, sbc, xp in [(left_boundary_state, 'left', self._x[0]),
                            (right_boundary_state, 'right', self._x[-1])]:
            if bc is not None:
                if sbc == 'left':
                    self._left_bc = np.copy(bc)
                else:
                    self._right_bc = np.copy(bc)
            else:
                for offset, initial_condition in enumerate(initial_conditions):
                    if callable(initial_condition):
                        if sbc == 'left':
                            self._left_bc[offset] = initial_condition(xp)
                        else:
                            self._right_bc[offset] = initial_condition(xp)
                    else:
                        if sbc == 'left':
                            self._left_bc[offset] = initial_condition
                        else:
                            self._right_bc[offset] = initial_condition

        self._variable_scales = np.ones(self._n_dof)

        self._solution_times = []
        self._solution_data = dict()
        for i in range(self._n_equations):
            self._solution_data[self._variable_names[i]] = []
            self._solution_data[self._variable_names[i]].append(self._initial_state[i::self._n_equations])

        self._source_term = source_term
        self._source_term_jacobian = source_term_jacobian

        self._lhs_inverse_operator = None
        self._I = csc_matrix(speye(self._n_dof))

        dxt = self._dx[:-1] + self._dx[1:]
        self._major_coeffs = - 2. * self._d / (self._dx[:-1] * self._dx[1:])
        self._sub_coeffs = 2. * self._d / (dxt * self._dx[:-1])
        self._sup_coeffs = 2. * self._d / (dxt * self._dx[1:])

        if self._lewis_numbers is None:
            self._cmaj = np.tile(self._major_coeffs, (self._n_equations, 1)).T
            self._csub = np.tile(self._sub_coeffs, (self._n_equations, 1)).T
            self._csup = np.tile(self._sup_coeffs, (self._n_equations, 1)).T

            majdiag = np.tile(self._major_coeffs, (self._n_equations, 1)).T.ravel()
            supdiag = np.tile(self._sup_coeffs, (self._n_equations, 1)).T.ravel()
            subdiag = np.tile(np.hstack((self._sub_coeffs[1:], self._sub_coeffs[0])),
                              (self._n_equations, 1)).T.ravel()
        else:
            le = 1. / np.tile(self._lewis_numbers, (self._nx_interior, 1))
            self._cmaj = le * np.tile(self._major_coeffs, (self._n_equations, 1)).T
            self._csub = le * np.tile(self._sub_coeffs, (self._n_equations, 1)).T
            self._csup = le * np.tile(self._sup_coeffs, (self._n_equations, 1)).T

            majdiag = (le * np.tile(self._major_coeffs, (self._n_equations, 1)).T).ravel()
            supdiag = (le * np.tile(self._sup_coeffs, (self._n_equations, 1)).T).ravel()
            subdiag = (le * np.tile(np.hstack((self._sub_coeffs[1:], self._sub_coeffs[0])),
                                    (self._n_equations, 1)).T).ravel()

        self._djac = csc_matrix(diags([majdiag, supdiag, subdiag], [0, self._n_equations, -self._n_equations]))

    def _global_rhs(self, t, state_interior):
        neq = self._n_equations
        nxi = self._nx_interior
        state_with_bcs = np.vstack((self._left_bc,
                                    state_interior.reshape((nxi, neq)),
                                    self._right_bc))
        rhs = (self._csub * state_with_bcs[:-2, :] + \
               self._csup * state_with_bcs[2:, :] + \
               self._cmaj * state_with_bcs[1:-1, :]).ravel()
        for i in range(nxi):
            rhs[i * neq:(i + 1) * neq] += self._source_term(state_interior[i * neq: (i + 1) * neq])
        return rhs

    def _global_jacobian(self, state_interior):
        neq = self._n_equations
        nxi = self._nx_interior
        karray = np.ndarray((nxi, neq, neq))
        for i in range(nxi):
            karray[i, :, :] = self._source_term_jacobian(state_interior[i * neq: (i + 1) * neq])
        kjac = block_diag(karray, format='csc')
        return kjac + self._djac

    def _setup_superlu(self, state_interior, prefactor):
        jac = self._global_jacobian(state_interior)
        jac.eliminate_zeros()
        self._lhs_inverse_operator = superlu_factor(prefactor * jac - self._I)

    def _solve_superlu(self, residual):
        return self._lhs_inverse_operator.solve(residual), 1, True

    def _do_insitu_processing(self, t, state, *args, **kwargs):
        self._solution_times.append(t)
        neq = self._n_equations
        for i in range(neq):
            self._solution_data[self._variable_names[i]].append(state[i::neq])

    def trajectory_data(self, key):
        if key not in self._solution_data.keys():
            print('Available data:', self._solution_data.keys())
            raise ValueError('data identifier ' + str(key) + ' is not valid!')
        else:
            return np.array(self._solution_data[key])

    @property
    def grid(self):
        return self._x

    @property
    def diffusion_coefficient(self):
        return self._d

    @property
    def variable_names(self):
        return self._variable_names

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def final_state(self):
        return self._final_state

    @property
    def solution_times(self):
        return np.array(self._solution_times)

    def integrate(self,
                  termination,
                  time_method='implicit-esdirk64',
                  time_step='adaptive',
                  first_time_step=1.e-6,
                  max_time_step=1.e0,
                  minimum_time_step_count=40,
                  transient_tolerance=1.e-10,
                  write_log=False,
                  log_rate=100,
                  maximum_steps_per_jacobian=1,
                  linear_solve_tolerance=1.e-15,
                  nonlinear_solve_tolerance=1.e-12):

        governor = Governor()
        governor.termination_criteria = termination
        governor.minimum_time_step_count = minimum_time_step_count
        governor.projector_setup_rate = maximum_steps_per_jacobian
        governor.do_logging = write_log
        governor.log_rate = log_rate
        governor.norm_weighting = 1. / self._variable_scales
        governor.custom_post_process_step = self._do_insitu_processing

        self._linear_solve_tolerance = linear_solve_tolerance
        self._nonlinear_solve_tolerance = nonlinear_solve_tolerance

        if time_method.lower() == 'esdirk64':
            method = ESDIRK64(SimpleNewtonSolver(evaluate_jacobian_every_iter=False,
                                                 norm_weighting=1. / self._variable_scales,
                                                 tolerance=self._nonlinear_solve_tolerance),
                              norm_weighting=1. / self._variable_scales)
            projector_setup = self._setup_superlu
            projector_solve = self._solve_superlu
        elif time_method.lower() == 'erk54 cash karp':
            method = AdaptiveERK54CashKarp()
            projector_setup = None
            projector_solve = None
        elif time_method.lower() == 'forward euler':
            method = ForwardEuler()
            projector_setup = None
            projector_solve = None
            if isinstance(time_step, str):
                if time_step.lower() == 'adaptive':
                    raise ValueError('Cannot do forward euler with adaptive stepping!')

        if isinstance(time_step, str):
            if time_step.lower() == 'adaptive':
                controller = PIController(first_step=first_time_step,
                                          max_step=max_time_step,
                                          target_error=transient_tolerance)
        else:
            controller = time_step

        self._final_state = governor.integrate(right_hand_side=self._global_rhs,
                                               projector_setup=projector_setup,
                                               projector_solve=projector_solve,
                                               initial_condition=self._initial_state,
                                               controller=controller,
                                               method=method)[1]

    def integrate_to_steady(self, steady_tolerance=1.e-4, **kwargs):
        self.steady_tolerance = steady_tolerance
        self.integrate(Steady(steady_tolerance), **kwargs)

    def integrate_to_time(self, final_time, **kwargs):
        self.final_time = final_time
        self.integrate(FinalTime(final_time), **kwargs)


class Test(unittest.TestCase):
    def test(self):
        import numpy as np

        def right_hand_side(c, k_ab, k_bc):
            """
            Computes the right-hand side function for the ODE system.

            Note that time integration requires a function that takes (t, y) as arguments.
            To accomodate this, we will write a lambda after defining the rate constants,
            which passes the appropriate y value and rate constant to this function (and ignores the time).

            :param c: current concentration vector
            :param k_ab: the rate constant of the reaction A -> B
            :param k_bc: the rate constant of the reaction A + B -> 2C
            :return: right-hand side of the ODE system
            """
            c_a = c[0]
            c_b = c[1]
            c_c = c[2]
            q_1 = k_ab * c_a
            q_2 = k_bc * c_a * c_b
            return np.array([-q_1 - q_2, q_1 - q_2, 2. * q_2])

        def jacobian(c, k_ab, k_bc):
            c_a = c[0]
            c_b = c[1]
            c_c = c[2]
            q_1 = k_ab * c_a
            q_2 = k_bc * c_a * c_b
            dq1_ca = k_ab
            dq1_cb = 0.
            dq1_cc = 0.
            dq2_ca = k_bc * c_b
            dq2_cb = k_bc * c_a
            dq2_cc = 0.
            return np.array([[-dq1_ca - q_2, -dq1_cb - dq2_cb, -dq1_cc - dq2_cc],
                             [dq1_ca - q_2, dq1_cb - dq2_cb, dq1_cc - dq2_cc],
                             [2. * dq2_ca, 2. * dq2_cb, 2. * dq2_cc]])

        k_ab = 1.
        k_bc = 10.

        dr = DiffusionReaction1D_FiniteDifference(initial_conditions=[lambda x: np.exp(-100. * x ** 2),
                                                                      lambda x: 0.5 * (1. + np.sin(2. * np.pi * x)),
                                                                      0.],
                                                  left_boundary_state=[1., 1., 0.],
                                                  right_boundary_state=[0., 1., 0.],
                                                  diffusion_coefficient=lambda x: np.exp(-1.e0 * (x - 0.2) ** 2),
                                                  source_term=lambda y: right_hand_side(y, k_ab, k_bc),
                                                  source_term_jacobian=lambda y: jacobian(y, k_ab, k_bc),
                                                  variable_names=['A', 'B', 'C'],
                                                  grid_points=32,
                                                  grid_type='clustered',
                                                  grid_cluster_intensity=2.,
                                                  grid_cluster_point=0.2)

        dr.integrate_to_time(final_time=0.05,
                             first_time_step=1.e-5,
                             time_step=1.e-5,
                             time_method='forward euler',
                             log_rate=40, write_log=False,
                             transient_tolerance=1.e-8)
        dr.integrate_to_time(final_time=0.05,
                             first_time_step=1.e-4,
                             time_method='erk54 cash karp',
                             log_rate=40, write_log=False,
                             transient_tolerance=1.e-8)
        dr.integrate_to_time(final_time=0.05,
                             first_time_step=1.e-3,
                             time_method='esdirk64',
                             log_rate=10, write_log=False,
                             transient_tolerance=1.e-8)

        ca = dr.trajectory_data('A')
        cb = dr.trajectory_data('B')
        cc = dr.trajectory_data('C')
        x = dr.grid[1:-1]


if __name__ == '__main__':
    unittest.main()
