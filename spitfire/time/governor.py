"""
This module contains the time integration governor abstraction,
which combines time steppers, nonlinear solvers, and step controllers into a generic integration loop
that supports arbitrary in situ processing, logging, termination of the integration loop, and more.
For instance, evaluation of the linear operator (Jacobian/preconditioner evaluation and factorization)
in implicit methods is guided by the Governor, as it can oversee the entire integration.
In many cases the Jacobian/preconditioner can be lagged for several time steps to speed up the integration process substantially.
"""

# Spitfire - a Python-C++ library for building tabulated chemistry models and solving differential equations                    
# Copyright 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS). 
#                       
# You should have received a copy of the 3-clause BSD License                                        
# along with this program.  If not, see <https://opensource.org/licenses/BSD-3-Clause>.   
#                    
# Questions? Contact Mike Hansen (mahanse@sandia.gov)    

from numpy import copy as numpy_copy
from numpy import any, logical_or, isinf, isnan, min, max, array, Inf, diag_indices, save as numpy_save
from scipy.linalg import norm
from spitfire.time.stepcontrol import ConstantTimeStep
from spitfire.time.methods import ImplicitTimeStepper
from spitfire.time.nonlinear import finite_difference_jacobian
from scipy.linalg.lapack import dgetrf as lapack_lu_factor
from scipy.linalg.lapack import dgetrs as lapack_lu_solve
import time as timer
import datetime


class SaveAllDataToList(object):
    """A class that saves solution data and times from time integration at a particular step frequency

    **Constructor**:

    Parameters
    ----------
    initial_solution : np.ndarray
        the initial solution
    initial_time : float
        the initial time (default: 0.)
    save_frequency : int
        how many steps are taken between solution data and times being saved (default: 1)
    file_prefix : str
        file prefix where solution times and solution data will be dumped (to numpy binary) during the simulation
    file_first_and_last_only : bool
        whether or not to save all data (False, default) or only the first and last solutions to the numpy binary file
    save_first_and_last_only : bool
        whether or not to retain (in memory) all data (False, default) or only the first and last solutions
    """

    def __init__(self, initial_solution, initial_time=0., save_frequency=1, file_prefix=None,
                 file_first_and_last_only=False, save_first_and_last_only=False):
        self._t_list = [numpy_copy(initial_time)]
        self._solution_list = [numpy_copy(initial_solution)]
        self._save_count = 0
        self._save_frequency = save_frequency
        self._file_prefix = file_prefix
        self._file_first_and_last_only = file_first_and_last_only
        self._save_first_and_last_only = save_first_and_last_only

    @property
    def t_list(self):
        """Obtain a np.array of the list of solution times that were saved"""
        return array(self._t_list)

    @property
    def solution_list(self):
        """Obtain a np.array of the solutions that were saved"""
        return array(self._solution_list)

    def save_data(self, t, solution, *args, **kwargs):
        """Method to provide to the Governor() object as the custom_post_process_step"""
        self._save_count += 1
        if self._save_count == self._save_frequency:
            self._save_count = 0
            if len(self.t_list) > 1 and self._save_first_and_last_only:
                self._t_list[-1] = numpy_copy(t)
                self._solution_list[-1] = numpy_copy(solution)
            else:
                self._t_list.append(numpy_copy(t))
                self._solution_list.append(numpy_copy(solution))
            if self._file_prefix is not None:
                if self._file_first_and_last_only:
                    numpy_save(self._file_prefix + '_times.npy', self.t_list[[0, -1]])
                    numpy_save(self._file_prefix + '_solutions.npy', self.solution_list[[0, -1], :])
                else:
                    numpy_save(self._file_prefix + '_times.npy', self.t_list)
                    numpy_save(self._file_prefix + '_solutions.npy', self.solution_list)

    def reset_data(self, initial_solution, initial_time=0.):
        """Reset the data on a SaveAllDataToList object"""
        self._t_list = [numpy_copy(initial_time)]
        self._solution_list = [numpy_copy(initial_solution)]


class FinalTime(object):
    """
    Wrapper class for terminating time integration based on a specified final time.
    """

    def __init__(self, final_time):
        self.final_time = final_time


class Steady(object):
    """
    Wrapper class for terminating time integration based on reaching a steady state (small residual).
    """

    def __init__(self, steady_tolerance):
        self.steady_tolerance = steady_tolerance


class NumberOfTimeSteps(object):
    """
    Wrapper class for terminating time integration based on the number of time steps taken.
    """

    def __init__(self, number_of_steps):
        self.number_of_steps = number_of_steps


class CustomTermination(object):
    """
    Wrapper class for terminating time integration based on a custom rule, a function f(state, t, nt, residual)
    which returns True when time integration should continue.
    """

    def __init__(self, custom_rule):
        self.custom_rule = custom_rule


class Governor(object):
    """
    The class that drives time integration at a high level.

    Objects are built with an empty constructor and then attributes are set directly.

    Attributes
    ----------
    do_logging : bool
        whether or not to continually write out the integrator status and some statistics (default: True)
        (turn this off for best performance)
    log_rate : int
        how many time steps between new lines in the log (default: 1) (increase for best performance)
    lines_per_header : int
        how many log lines are written between rewriting the header (default: 10)
    extra_logger_log : callable
        a method of form f(state, current_time, number_of_time_steps, number_nonlinear_iter, number_linear_iter)
        that adds to the log line (default: None)
    extra_logger_title_line1 : str
        the first line to place above the extra_logger_log text (default: None)
    extra_logger_title_line2 : str
        the second line to place above the extra_logger_log text (default: None)
    projector_setup_rate : int
        the largest number of steps that can occur before setting up the linear projector (default: 1 (every step))
    time_step_increase_factor_to_force_jacobian : float
        how much the time step size must increase on a time step to force setup of the projector (default: 1.05)
    time_step_decrease_factor_to_force_jacobian : float
        how much the time step size must decrease on a time step to force setup of the projector (default: 0.9)
    termination_criteria : CustomTermination or FinalTime or Steady or NumberOfTimeSteps
        a time termination object (default: None)
    norm_weighting : np.ndarray or float
        how the temporal error estimate is weighted in its norm calculation (default: 1)
    norm_order : int or np.Inf
        the order of the norm used in the temporal error estimate (default: Inf)
    minimum_time_step_count : int
        minimum number of time steps that can be run (default: 0)
    maximum_time_step_count : int
        maximum number of time steps that can be run (default: Inf)
    strict_temporal_error_control : bool
        whether or not to enforce that error-controlled adaptive time stepping keeps the error estimate below the target (default: False)
    nonlinear_solve_must_converge : bool
        whether or not the nonlinear solver in each time step of implicit methods must converge (default: False)
    warn_on_failed_step : bool
        whether or not to print a warning when a step fails (default: False)
    return_on_failed_step : bool
        whether or not to return and stop integrating when a step fails (default: False)
    time_step_reduction_factor_on_failure : float
        factor used in reducing the step size after a step fails, if not returning on failure (default: 0.8)
    custom_update_checking_rule : callable
        method of the form f(state, dstate, time_error, target_error, nonlinear_solve_converged) that checks validity of a state update (default: None)
    time_step_reduction_factor_on_slow_solve : float
        factor used in reducing the step size after a step is deemed slow by the nonlinear solver (default: 0.8)
    clip_to_positive : bool
        whether or not to clip solution values to be nonnegative after each time step (default: False)
    custom_post_process_step : callable
        method of the form f(current_time, state) that is called after each time step is complete (default: None)
    show_solver_stats_in_situ : bool
        whether or not to include the number of nonlinear iterations per step, linear iterations per nonlinear iteration,
        number of time steps per Jacobian evaluation (projector setup) in the logged output (default: False)
    """

    def __init__(self):

        # termination criteria
        self._stop_stepping_at_final_time = False
        self._stop_stepping_at_steady_state = False
        self._stop_stepping_at_number_of_steps = False
        self._stop_stepping_custom_rule = None
        self._termination_criteria = None
        self.norm_weighting = 1.
        self.norm_order = Inf
        self.minimum_time_step_count = 0
        self.maximum_time_step_count = Inf

        # step acceptance/rejection
        self.strict_temporal_error_control = False
        self.nonlinear_solve_must_converge = False
        self.warn_on_failed_step = False
        self.return_on_failed_step = False
        self.time_step_reduction_factor_on_failure = 0.8
        self.custom_update_checking_rule = None
        self.time_step_reduction_factor_on_slow_solve = 0.8

        # projector setup
        self.projector_setup_rate = 1
        self._projector_setup_count = 0
        self.time_step_increase_factor_to_force_jacobian = 1.05
        self.time_step_decrease_factor_to_force_jacobian = 0.9
        self.show_solver_stats_in_situ = False

        # logger
        self.do_logging = True
        self.extra_logger_log = None
        self.extra_logger_title_line1 = None
        self.extra_logger_title_line2 = None
        self.log_rate = 1
        self.lines_per_header = 10
        self._log_count = 0
        self._log_title_count = 0

        # other
        self._debug_mode = False
        self.clip_to_positive = False
        self.custom_post_process_step = None

    @property
    def termination_criteria(self):
        if self._stop_stepping_at_final_time:
            return 'Final time: ' + str(self.final_time)
        elif self._stop_stepping_at_number_of_steps:
            return 'Number of steps: ' + str(self.number_of_steps_to_run)
        elif self._stop_stepping_at_steady_state:
            return 'Steady state: ' + str(self.steady_state_tolerance)
        elif self._stop_stepping_custom_rule is not None:
            return 'Custom rule'
        else:
            return 'Termination criteria have not been set!'

    @termination_criteria.setter
    def termination_criteria(self, tc):
        self._termination_criteria = tc

        if isinstance(tc, FinalTime):
            self._stop_stepping_at_final_time = True
            self.final_time = tc.final_time

        elif isinstance(tc, Steady):
            self._stop_stepping_at_steady_state = True
            self.steady_state_tolerance = tc.steady_tolerance

        elif isinstance(tc, NumberOfTimeSteps):
            self._stop_stepping_at_number_of_steps = True
            self.number_of_steps_to_run = tc.number_of_steps

        elif isinstance(tc, CustomTermination):
            self._stop_stepping_custom_rule = tc.custom_rule

        else:
            raise ValueError('Bad termination criteria given.')

    def _print_debug_mode(self, statement):
        if self._debug_mode:
            print('\n  --> DEBUG:', statement, '\n')

    def _check_continue_stepping(self, state, t, nt, residual):
        if nt < self.minimum_time_step_count:
            return True

        if nt >= self.maximum_time_step_count:
            return False

        if self._stop_stepping_at_final_time:
            return t < self.final_time

        elif self._stop_stepping_at_number_of_steps:
            return nt < self.number_of_steps_to_run

        elif self._stop_stepping_at_steady_state:
            return residual > self.steady_state_tolerance

        elif self._stop_stepping_custom_rule is not None:
            return self._stop_stepping_custom_rule(state, t, nt, residual)
            # keep stepping if 'custom_rule' says to keep stepping (True)
            # custom_rule is for something like terminating integration at ignition delay,
            # which may involve time derivatives of the state or complex comparisons to an initial condition
        else:
            raise ValueError('Somehow a bad termination critera got past the Governor construction.')

    def _there_are_any_naninf(self, an_array):
        return any(logical_or(isinf(an_array), isnan(an_array)))

    def _check_state_update(self, state, dstate,
                            time_error, target_error,
                            nonlinear_solve_converged):
        # check for NaN and Inf
        # check for exceeding target error and strict_temporal_error_control
        # check for nonlinear convergence
        # check a custom rule that might check physical realizability (e.g. temperature bounds), for instance
        #
        # return False if we should NOT accept the update
        if time_error > target_error and self.strict_temporal_error_control:
            self._print_debug_mode('check_state_update(): strict error exceeded')
            return False
        elif self._there_are_any_naninf(dstate):
            self._print_debug_mode('check_state_update(): NaN or Inf detected')
            return False
        elif nonlinear_solve_converged is not None and (
                    not nonlinear_solve_converged and self.nonlinear_solve_must_converge):
            self._print_debug_mode('check_state_update(): required nonlinear solve failed to converge')
            return False
        elif self.custom_update_checking_rule is not None:
            self._print_debug_mode('check_state_update(): running custom_rule...')
            return self.custom_update_checking_rule(state, dstate, time_error, nonlinear_solve_converged)
        else:
            return True

    def _process_failed_step(self, time_step_size):
        if self.return_on_failed_step:
            raise ValueError('Step failed and return_on_failed_step=True, stopping!')
        else:
            self._print_debug_mode('failed step with dt = ' + str(time_step_size) + ', reducing by a factor of ' + str(
                self.time_step_reduction_factor_on_failure))
            time_step_size *= self.time_step_reduction_factor_on_failure
            freshen_projector = True
            if self.warn_on_failed_step:
                print('Warning! Step failed! return_on_failed_step=False so continuing on... retrying step...')
            return time_step_size, freshen_projector

    def _log_header(self, method_name):
        if self.do_logging:
            now = datetime.datetime.now()
            print('\n', now.strftime('%Y-%m-%d %H:%M'), ': Spitfire running case with method:', method_name,
                  flush=True)

    def _log_footer(self, cputime):
        if self.do_logging:
            now = datetime.datetime.now()
            print('\n', now.strftime('%Y-%m-%d %H:%M'), ': Spitfire finished in {:.8e} seconds!\n'.format(cputime),
                  flush=True)

    def _write_log(self,
                   state,
                   current_time,
                   time_step_size,
                   residual,
                   number_of_time_steps,
                   number_nonlinear_iter,
                   number_linear_iter,
                   number_projector_setup,
                   cputime):
        self._log_count += 1
        sim_time_lines = [
            (f'{"number of":<10}', f'{"time steps":<10}', f' {number_of_time_steps:<9}'),
            (f'{"simulation":<10}', f'{"time (s)":<10}', f'{current_time:<10.2e}'),
            (f'{"time step":<10}', f'{"size (s)":<10}', f'{time_step_size:<10.2e}'), ]
        if number_nonlinear_iter == 'n/a':
            nni_over_nts = 'n/a'
            nli_over_nni = 'n/a'
            nts_over_nps = 'n/a'
            advanced_lines = [
                (f'{"nlin. iter":<10}', f'{"per step":<10}', f'{nni_over_nts:<10}'),
                (f'{"lin. iter":<10}', f'{"per nlin.":<10}', f'{nli_over_nni:<10}'),
                (f'{"steps":<10}', f'{"per Jac.":<10}', f'{nts_over_nps:<10}')]
        else:
            nni_over_nts = number_nonlinear_iter / number_of_time_steps
            nli_over_nni = number_linear_iter / number_nonlinear_iter
            nts_over_nps = number_of_time_steps / number_projector_setup
            advanced_lines = [
                (f'{"nlin. iter":<10}', f'{"per step":<10}', f'{nni_over_nts:<10.2f}'),
                (f'{"lin. iter":<10}', f'{"per nlin.":<10}', f'{nli_over_nni:<10.2f}'),
                (f'{"steps":<10}', f'{"per Jac.":<10}', f'{nts_over_nps:<10.2f}')]
        cput_over_nts = 1.e3 * cputime / float(number_of_time_steps)
        residual_line = [(f'{"diff. eqn.":<10}', f'{"|residual|":<10}', f'{residual:<10.2e}')]
        cpu_time_lines = [
            (f'{"total cpu":<10}', f'{"time (s)":<10}', f'{cputime:<10.2e}'),
            (f'{"cput per":<10}', f'{"step (ms)":<10}', f'{cput_over_nts:<10.2e}')]
        if self.do_logging:
            log_lines = sim_time_lines
            if self.show_solver_stats_in_situ:
                log_lines += advanced_lines
            log_lines += residual_line
            log_lines += cpu_time_lines

            extra_title_line1 = '' if self.extra_logger_title_line1 is None else self.extra_logger_title_line1
            extra_title_line2 = '' if self.extra_logger_title_line2 is None else self.extra_logger_title_line2

            title_line_1 = '|' + ' | '.join([a for (a, b, c) in log_lines])[:-1] + '|' + extra_title_line1
            title_line_2 = '|' + ' | '.join([b for (a, b, c) in log_lines])[:-1] + '|' + extra_title_line2
            log_str = '|' + ' | '.join([c for (a, b, c) in log_lines])[:-1] + '|'

            line_of_dashes = '-' * (len(title_line_2) - 1)
            log_title_str = '\n' + title_line_1 + '\n' + title_line_2 + '\n' + line_of_dashes + '|'

            if self.extra_logger_log is not None:
                log_str += self.extra_logger_log(state,
                                                 current_time,
                                                 number_of_time_steps,
                                                 number_nonlinear_iter,
                                                 number_linear_iter)

            if number_of_time_steps == 1:
                print(log_title_str)
            if self._log_count == self.log_rate:
                self._log_title_count += 1
                self._log_count = 0
                if self._log_title_count == self.lines_per_header:
                    print(line_of_dashes)
                    print(log_title_str)
                    self._log_title_count = 0
                print(log_str, flush=True)

    def _check_projector(self,
                         nlsuccess,
                         nlslowness,
                         time_step_size,
                         recent_time_step_size):
        if self._projector_setup_count == self.projector_setup_rate:
            self._print_debug_mode('check_projector(): count = rate')
            self._projector_setup_count = 0
            return True, time_step_size
        elif nlslowness:
            self._print_debug_mode('check_projector(): nonlinear solve converging slowly')
            time_step_size *= self.time_step_reduction_factor_on_slow_solve
            return True, time_step_size
        elif not nlsuccess:
            time_step_size *= self.time_step_reduction_factor_on_failure
            self._print_debug_mode('check_projector(): nonlinear solve failed to converge')
            return True, time_step_size
        elif time_step_size > recent_time_step_size * self.time_step_increase_factor_to_force_jacobian:
            self._print_debug_mode('check_projector(): time step increased by too much')
            return True, time_step_size
        elif time_step_size < recent_time_step_size * self.time_step_decrease_factor_to_force_jacobian:
            self._print_debug_mode('check_projector(): time step decreased by too much')
            return True, time_step_size
        else:
            return False, time_step_size

    def _post_process_step(self, current_time, state):
        if self.clip_to_positive:
            state[state < 0.] = 0.

        if self.custom_post_process_step is not None:
            self.custom_post_process_step(current_time, state)

        return state

    def integrate(self,
                  right_hand_side=None,
                  projector_setup=None,
                  projector_solve=None,
                  initial_condition=None,
                  initial_time=0.,
                  method=None,
                  controller=None):
        """
        Run a time integration problem

        :param right_hand_side: the right-hand side of the ODE system, in the form f(t, y)
        :param projector_setup: the linear projector setup method, in the form f(y, jacobian_scale)
        :param projector_solve: the linear projector solve method, in the form f(residual)
        :param initial_condition: the initial state vector
        :param initial_time: the initial time
        :param method: the time stepping method (a spitfire.time.methods.TimeStepper object)
        :param controller: the time step controller, either a float (for constant time step) or spitfire.time.stepcontrol class
        :return: a tuple of time integration statistics (dictionary), final state, final time, and final time step size
        """

        if self._stop_stepping_at_final_time and (initial_time > self.final_time):
            raise ValueError(f'Error in Governor.integrate - the initial time of {initial_time} exceeds '
                             f'the specified final time of {self.final_time}')

        method_is_implicit = isinstance(method, ImplicitTimeStepper)
        use_finite_difference_jacobian = method_is_implicit and projector_setup is None
        build_projector_in_governor = (method_is_implicit and method.nonlinear_solver.setup_projector_in_governor) or \
                                      use_finite_difference_jacobian
        freshen_projector = True

        if isinstance(controller, float):
            time_step_value = controller
            controller = ConstantTimeStep(time_step_value)

        self._log_header(method.name)

        state = numpy_copy(initial_condition)
        current_time = numpy_copy(initial_time)
        number_of_time_steps = 0
        smallest_step_size = 1.e305
        largest_step_size = 0.
        time_step_size = controller.first_step_size()
        smallest_step_size = min([smallest_step_size, time_step_size])
        largest_step_size = max([largest_step_size, time_step_size])

        number_nonlinear_iter = 0
        number_linear_iter = 0
        number_projector_setup = 0

        if use_finite_difference_jacobian:
            self._lapack_lhs_factor = None
            rhs_from_state = lambda state: right_hand_side(current_time, state)

            my_diag_indices = diag_indices(initial_condition.size)

            def finite_difference_jacobian_setup(state, prefactor):
                j = finite_difference_jacobian(rhs_from_state, rhs_from_state(state), state) * prefactor
                j[my_diag_indices] -= 1.
                self._lapack_lhs_factor = lapack_lu_factor(j)[:2]

            def finite_difference_jacobian_solve(residual):
                return lapack_lu_solve(self._lapack_lhs_factor[0],
                                       self._lapack_lhs_factor[1],
                                       residual)[0], 1, True

            projector_setup = finite_difference_jacobian_setup
            projector_solve = finite_difference_jacobian_solve

        continue_time_stepping = True

        cpu_time_0 = timer.perf_counter()
        while continue_time_stepping:

            if self._stop_stepping_at_final_time:
                if current_time + time_step_size > self.final_time:
                    time_step_size = self.final_time - current_time

            if method_is_implicit:
                if build_projector_in_governor and freshen_projector:
                    projector_setup(state, time_step_size * method.implicit_coefficient)
                    number_projector_setup += 1
            self._projector_setup_count += 1

            step_output = method.single_step(state,
                                             current_time,
                                             time_step_size,
                                             right_hand_side,
                                             lambda x: projector_setup(x, time_step_size * method.implicit_coefficient),
                                             projector_solve)
            dstate = step_output.solution_update
            time_error = step_output.temporal_error
            nliter = step_output.nonlinear_iter
            liter = step_output.linear_iter
            nlsuccess = step_output.nonlinear_converged
            nlisslow = step_output.slow_nonlinear_convergence
            number_projector_setup += step_output.projector_setups if step_output.projector_setups is not None else 0
            residual = norm(dstate * self.norm_weighting) / time_step_size

            recent_time_step_size = time_step_size
            if self._check_state_update(state, dstate, time_error, controller.target_error(), nlsuccess):
                state += dstate
                current_time += time_step_size
                number_of_time_steps += 1
                number_nonlinear_iter = number_nonlinear_iter + nliter if nliter is not None else 'n/a'
                number_linear_iter = number_linear_iter + liter if liter is not None else 'n/a'

                state = self._post_process_step(current_time, state)

                self._write_log(state,
                                current_time,
                                time_step_size,
                                residual,
                                number_of_time_steps,
                                number_nonlinear_iter,
                                number_linear_iter,
                                number_projector_setup,
                                timer.perf_counter() - cpu_time_0)

                time_step_size = controller(number_of_time_steps, time_step_size, step_output)

                if method_is_implicit:
                    freshen_projector, time_step_size = self._check_projector(nlsuccess, nlisslow,
                                                                              time_step_size, recent_time_step_size)

            else:
                time_step_size, freshen_projector = self._process_failed_step(time_step_size)

            smallest_step_size = min([smallest_step_size, time_step_size])
            largest_step_size = max([largest_step_size, time_step_size])

            continue_time_stepping = self._check_continue_stepping(state,
                                                                   current_time,
                                                                   number_of_time_steps,
                                                                   residual)
        total_runtime = timer.perf_counter() - cpu_time_0
        stats_dict = {'time steps': number_of_time_steps,
                      'simulation time': current_time,
                      'nonlinear iter': number_nonlinear_iter,
                      'linear iter': number_linear_iter,
                      'Jacobian setups': number_projector_setup,
                      'total cpu time (s)': total_runtime}
        if self.do_logging:
            print('\nIntegration successfully completed!')
            print('\nStatistics:')
            print('- number of time steps :', number_of_time_steps)
            print('- final simulation time:', current_time)
            print('- smallest time step   :', smallest_step_size)
            print('- average time step    :', current_time / number_of_time_steps)
            print('- largest time step    :', largest_step_size)
            print('\n  CPU time')
            print('- total    (s) : {:.6e}'.format(total_runtime))
            print('- per step (ms): {:.6e}'.format(1.e3 * total_runtime / number_of_time_steps))
            if method_is_implicit:
                print('\n  Nonlinear iterations')
                print('- total   : {:}'.format(number_nonlinear_iter))
                print('- per step: {:.1f}'.format(number_nonlinear_iter / number_of_time_steps))
                print('\n  Linear iterations')
                print('- total     : {:}'.format(number_linear_iter))
                print('- per step  : {:.1f}'.format(number_linear_iter / number_of_time_steps))
                print('- per nliter: {:.1f}'.format(number_linear_iter / number_nonlinear_iter))
                print('\n  Jacobian setups')
                print('- total     : {:}'.format(number_projector_setup))
                print('- steps per : {:.1f}'.format(number_of_time_steps / number_projector_setup))
                print('- nliter per: {:.1f}'.format(number_nonlinear_iter / number_projector_setup))
                print('- liter per : {:.1f}'.format(number_linear_iter / number_projector_setup))

        self._log_footer(total_runtime)

        self._log_count = 0
        self._log_title_count = 0
        self._projector_setup_count = 0
        return stats_dict, state, current_time, time_step_size
