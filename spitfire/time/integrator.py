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

import numpy as np
from numpy import any, logical_or, isinf, isnan, min, max, array, Inf, diag_indices, zeros
from scipy.linalg import norm
from spitfire.time.stepcontrol import ConstantTimeStep, PIController
from spitfire.time.methods import KennedyCarpenterS6P4Q3
from spitfire.time.nonlinear import finite_difference_jacobian, SimpleNewtonSolver
from scipy.linalg.lapack import dgetrf as lapack_lu_factor
from scipy.linalg.lapack import dgetrs as lapack_lu_solve
import time as timer
import datetime
import logging


def _log_header(verbose,
                method_name):
    if verbose:
        now = datetime.datetime.now()
        print('\n', now.strftime('%Y-%m-%d %H:%M'), ': Spitfire running case with method:', method_name,
              flush=True)


def _log_footer(verbose,
                cputime):
    if verbose:
        now = datetime.datetime.now()
        print('\n', now.strftime('%Y-%m-%d %H:%M'), ': Spitfire finished in {:.8e} seconds!\n'.format(cputime),
              flush=True)


def _print_debug_mode(debug_verbose, statement):
    if debug_verbose:
        print('\n  --> DEBUG:', statement, '\n')


def _check_state_update(debug_verbose, state, dstate,
                        time_error, target_error,
                        nonlinear_solve_converged,
                        strict_temporal_error_control,
                        nonlinear_solve_must_converge,
                        custom_update_check):
    # check for NaN and Inf
    # check for exceeding target error and strict_temporal_error_control
    # check for nonlinear convergence
    # check a custom rule that might check physical realizability (e.g. temperature bounds), for instance
    #
    # return False if we should NOT accept the update

    if time_error > target_error and strict_temporal_error_control:
        _print_debug_mode(debug_verbose, 'check_state_update(): strict error exceeded')
        return False
    elif any(logical_or(isinf(dstate), isnan(dstate))):
        _print_debug_mode(debug_verbose, 'check_state_update(): NaN or Inf detected')
        return False
    elif nonlinear_solve_converged is not None and (
                not nonlinear_solve_converged and nonlinear_solve_must_converge):
        _print_debug_mode(debug_verbose, 'check_state_update(): required nonlinear solve failed to converge')
        return False
    elif custom_update_check is not None:
        _print_debug_mode(debug_verbose, 'check_state_update(): running custom_rule...')
        return custom_update_check(state, dstate, time_error, nonlinear_solve_converged)
    else:
        return True


def _process_failed_step(debug_verbose,
                         time_step_size,
                         return_on_failed_step,
                         time_step_reduction_factor_on_failure,
                         warn_on_failed_step):
    if return_on_failed_step:
        raise ValueError('Step failed and return_on_failed_step=True, stopping!')
    else:
        _print_debug_mode(debug_verbose,
                          f'failed step with dt = {time_step_size}, '
                          f'reducing by a factor of {time_step_reduction_factor_on_failure}')
        time_step_size *= time_step_reduction_factor_on_failure
        eval_linear_setup = True
        if warn_on_failed_step:
            print('Warning! Step failed! return_on_failed_step=False so continuing on... retrying step...')
        return time_step_size, eval_linear_setup


def _check_linear_setup(debug_verbose,
                        nlsuccess,
                        nlslowness,
                        time_step_size,
                        recent_time_step_size,
                        linear_setup_count,
                        projector_setup_rate,
                        time_step_reduction_factor_on_slow_solve,
                        time_step_reduction_factor_on_failure,
                        time_step_increase_factor_to_force_jacobian,
                        time_step_decrease_factor_to_force_jacobian):
    if linear_setup_count == projector_setup_rate:
        _print_debug_mode(debug_verbose, 'check_projector(): count = rate')
        linear_setup_count = 0
        return True, time_step_size, linear_setup_count
    elif nlslowness:
        _print_debug_mode(debug_verbose, 'check_projector(): nonlinear solve converging slowly')
        time_step_size *= time_step_reduction_factor_on_slow_solve
        return True, time_step_size, linear_setup_count
    elif not nlsuccess:
        time_step_size *= time_step_reduction_factor_on_failure
        _print_debug_mode(debug_verbose, 'check_projector(): nonlinear solve failed to converge')
        return True, time_step_size, linear_setup_count
    elif time_step_size > recent_time_step_size * time_step_increase_factor_to_force_jacobian:
        _print_debug_mode(debug_verbose, 'check_projector(): time step increased by too much')
        return True, time_step_size, linear_setup_count
    elif time_step_size < recent_time_step_size * time_step_decrease_factor_to_force_jacobian:
        _print_debug_mode(debug_verbose, 'check_projector(): time step decreased by too much')
        return True, time_step_size, linear_setup_count
    else:
        return False, time_step_size, linear_setup_count


def _write_log(verbose,
               show_solver_stats_in_situ,
               log_count,
               log_rate,
               log_title_count,
               lines_per_header,
               extra_logger_title_line1,
               extra_logger_title_line2,
               extra_logger_log,
               state,
               current_time,
               time_step_size,
               residual,
               number_of_time_steps,
               number_nonlinear_iter,
               number_linear_iter,
               number_projector_setup,
               cputime):
    log_count += 1
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
    if verbose:
        log_lines = sim_time_lines
        if show_solver_stats_in_situ:
            log_lines += advanced_lines
        log_lines += residual_line
        log_lines += cpu_time_lines

        extra_title_line1 = '' if extra_logger_title_line1 is None else extra_logger_title_line1
        extra_title_line2 = '' if extra_logger_title_line2 is None else extra_logger_title_line2

        title_line_1 = '|' + ' | '.join([a for (a, b, c) in log_lines])[:-1] + '|' + extra_title_line1
        title_line_2 = '|' + ' | '.join([b for (a, b, c) in log_lines])[:-1] + '|' + extra_title_line2
        log_str = '|' + ' | '.join([c for (a, b, c) in log_lines])[:-1] + '|'

        line_of_dashes = '-' * (len(title_line_2) - 1)
        log_title_str = '\n' + title_line_1 + '\n' + title_line_2 + '\n' + line_of_dashes + '|'

        if extra_logger_log is not None:
            log_str += extra_logger_log(state,
                                        current_time,
                                        number_of_time_steps,
                                        number_nonlinear_iter,
                                        number_linear_iter)

        if number_of_time_steps == 1:
            print(log_title_str)
        if log_count == log_rate:
            log_title_count += 1
            log_count = 0
            if log_title_count == lines_per_header:
                print(line_of_dashes)
                print(log_title_str)
                log_title_count = 0
            print(log_str, flush=True)
    return log_count, log_title_count


def odesolve(right_hand_side,
             initial_state,
             output_times=None,
             save_each_step=False,
             initial_time=0.,
             stop_criteria=None,
             stop_at_time=None,
             stop_at_steady=None,
             minimum_time_step_count=0,
             maximum_time_step_count=Inf,
             pre_step_callback=None,
             post_step_callback=None,
             step_update_callback=None,
             method=KennedyCarpenterS6P4Q3(SimpleNewtonSolver()),
             step_size=PIController(),
             linear_setup=None,
             linear_solve=None,
             linear_setup_rate=1,
             mass_setup=None,
             mass_matvec=None,
             verbose=False,
             debug_verbose=False,
             log_rate=1,
             log_lines_per_header=10,
             extra_logger_title_line1=None,
             extra_logger_title_line2=None,
             extra_logger_log=None,
             norm_weighting=1.,
             strict_temporal_error_control=False,
             nonlinear_solve_must_converge=False,
             warn_on_failed_step=False,
             return_on_failed_step=False,
             time_step_reduction_factor_on_failure=0.8,
             time_step_reduction_factor_on_slow_solve=0.8,
             time_step_increase_factor_to_force_jacobian=1.05,
             time_step_decrease_factor_to_force_jacobian=0.9,
             show_solver_stats_in_situ=False,
             return_info=False,
             throw_on_failure=True):
    """
    Solve a time integration problem with a wide variety of solvers, termination options, etc.

    :param right_hand_side: the right-hand side of the ODE system, in the form f(t, y)
    :param initial_state: the initial state vector
    :param output_times: a collection of times at which the state will be returned
    :param save_each_step: set to True to save all data at each time step, or set to a positive integer to specify a step frequency of saving data
    :param initial_time: the initial time
    :param stop_criteria: any data with a call operator (state, t, dt, nt, residual) that returns True to stop time integration
    :param stop_at_time: force time integration to stop at exactly the provided final time
    :param stop_at_steady: force time integration to stop when a steady state is identified, provide either a boolean or a float for the tolerance
    :param minimum_time_step_count: minimum number of time steps that can be run (default: 0)
    :param maximum_time_step_count: maximum number of time steps that can be run (default: Inf)
    :param pre_step_callback: method of the form f(current_time, current_state, number_of_time_steps) called before each step (default: None)
    :param post_step_callback: method of the form f(current_time, current_state, residual, number_of_time_steps) called after each step, that can optionally return a modified state vector (default: None)
    :param step_update_callback: method of the form f(state, dstate, time_error, target_error, nonlinear_solve_converged) that checks validity of a state update (default: None)
    :param method: the time stepping method (a spitfire.time.methods.TimeStepper object), defaults to KennedyCarpenterS6P4Q3(SimpleNewtonSolver())
    :param step_size: the time step controller, either a float (for constant time step) or spitfire.time.stepcontrol class
    :param linear_setup: the linear system setup method, in the form f(t, y, scale), to set up scale * J - M
    :param linear_solve: the linear system solve method, in the form f(residual), to solve (scale * J - M)x = b
    :param linear_setup_rate: the largest number of steps that can occur before setting up the linear projector (default: 1 (every step))
    :param mass_setup: the setup method for the mass matrix/operator, in the form f(t, y), not supported yet, but hopefully soon...
    :param mass_matvec: the matrix-vector product operator for the mass matrix, in the form f(x) to produce M.dot(x), not supported yet, but hopefully soon...
    :param verbose: whether or not to continually write out the integrator status and some statistics, turn off for best performance (default: False)
    :param debug_verbose: whether or not to write A LOT of information during integration, do not use in any normal situation (default: True)
    :param log_rate: how frequently verbose output should be written, increase or turn off output for best performance (default: 1 = every step)
    :param log_lines_per_header: how many log lines are written between rewriting the header (default: 10)
    :param extra_logger_title_line1: the first line to place above the extra_logger_log text
    :param extra_logger_title_line2: the second line to place above the extra_logger_log text
    :param extra_logger_log: a method of form f(state, time, n_steps, number_nonlinear_iter, number_linear_iter) that adds to the log output
    :param return_info: whether or not to return solver statistics (default: False)
    :param norm_weighting: how the temporal error estimate is weighted in its norm calculation, can be a float or np.array (default: 1)
    :param strict_temporal_error_control: whether or not to enforce that error-controlled adaptive time stepping keeps the error estimate below the target (default: False)
    :param nonlinear_solve_must_converge: whether or not the nonlinear solver in each time step of implicit methods must converge (default: False)
    :param warn_on_failed_step: whether or not to print a warning when a step fails (default: False)
    :param return_on_failed_step: whether or not to return and stop integrating when a step fails (default: False)
    :param time_step_reduction_factor_on_failure: factor used in reducing the step size after a step fails, if not returning on failure (default: 0.8)
    :param time_step_reduction_factor_on_slow_solve: factor used in reducing the step size after a step is deemed slow by the nonlinear solver (default: 0.8)
    :param time_step_increase_factor_to_force_jacobian: how much the time step size must increase on a time step to force setup of the projector (default: 1.05)
    :param time_step_decrease_factor_to_force_jacobian: how much the time step size must decrease on a time step to force setup of the projector (default: 0.9)
    :param show_solver_stats_in_situ: whether or not to include the number of nonlinear iterations per step, linear iterations per nonlinear iteration, number of time steps per Jacobian evaluation (projector setup) in the logged output (default: False)
    :param return_info: whether or not to return a dictionary of solver statistics
    :param throw_on_failure: whether or not to throw an exception on integrator/model failure (default: True)
    :return this method returns a variety of options:
        1. output_times is provided: returns an array of output states, and the solver stats dictionary if return_info is True
        2. save_each_step is True (or a positive integer frequency): returns an array of times, and an array of output states, and the solver stats dictionary if return_info is True
        3. else: returns an the final state vector, final time, and final time step, and the solver stats dictionary if return_info is True
    """

    if not isinstance(initial_state, np.ndarray):
        raise TypeError('Error in Spitfire odesolve, the initial_state argument was not a NumPy array.')
    if len(initial_state.shape) > 1:
        raise TypeError('Error in Spitfire odesolve, the initial_state argument was not a 1-dimensional NumPy array'
                        ', you may simply need to call ravel() on it.')
    if not (isinstance(initial_time, float) or isinstance(initial_time, np.ndarray)):
        if isinstance(initial_time, int):
            initial_time = float(initial_time)
        else:
            raise TypeError(
                f'Error in Spitfire odesolve, the initial_time, {initial_time}, must be provided as a float.')
    if initial_time < 0.:
        raise ValueError(f'Error in Spitfire odesolve - the initial time of {initial_time} is negative.')
    if stop_at_time is not None:
        if not isinstance(stop_at_time, float):
            if isinstance(stop_at_time, int):
                stop_at_time = float(stop_at_time)
            else:
                raise TypeError('Error in Spitfire odesolve, the stop_at_time argument must be provided as a float, '
                                'which is the final time at which integration will cease.')
        if initial_time > stop_at_time:
            raise ValueError(f'Error in Spitfire odesolve - the initial time of {initial_time} exceeds '
                             f'the specified final time of {final_time}')
    if stop_at_steady is not None:
        if not (isinstance(stop_at_steady, bool) or isinstance(stop_at_steady, float)):
            raise TypeError('Error in Spitfire odesolve, the stop_at_time argument must be provided as '
                            'a boolean (default tolerance if True) or a float.')

    if not (isinstance(save_each_step, bool) or isinstance(save_each_step, int)):
        raise ValueError('Error in Spitfire odesolve, the save_each_step argument must be either True/False '
                         'or a positive integer (the step frequency at which data is saved).')
    else:
        if isinstance(save_each_step, int) and save_each_step < 0:
            raise ValueError('Error in Spitfire odesolve, the save_each_step argument must be either True/False '
                             'or a positive integer (the step frequency at which data is saved).')

    if output_times is not None:
        if stop_at_time is not None:
            raise ValueError('Error in Spitfire odesolve, the stop_at_time argument may not be provided if the '
                             'output_times argument is also in use.')
        if stop_at_steady is not None:
            raise ValueError('Error in Spitfire odesolve, the stop_at_steady argument may not be provided if the '
                             'output_times argument is also in use.')
        if stop_criteria is not None:
            raise ValueError('Error in Spitfire odesolve, the stop_criteria argument may not be provided if the '
                             'output_times argument is also in use.')
        if save_each_step:
            raise ValueError('Error in Spitfire odesolve, the save_each_step argument may not be provided if the '
                             'output_times argument is also in use.')
        if np.min(output_times) < initial_time:
            raise ValueError('Error in Spitfire odesolve, the provided output_times must be greater than or equal to'
                             ' the initial_time (defaults to 0.)')

        ot_list = output_times.tolist()
        if len(set(ot_list)) != len(ot_list):
            raise ValueError('Error in Spitfire odesolve, the provided output_times must be unique.')
        if ot_list != sorted(ot_list):
            raise ValueError('Error in Spitfire odesolve, the provided output_times must be increasing.')

    if output_times is None and stop_at_time is None and stop_at_steady is None and stop_criteria is None:
        raise ValueError('Error in Spitfire odesolve, you have not specified enough information to stop a simulation, '
                         'you must provide output_times, stop_at_time=tfinal, '
                         'stop_at_steady=[True or tolerance], '
                         'or stop_criteria as a function(t, state, residual, nsteps)')

    if isinstance(step_size, PIController):
        if not method.is_adaptive:
            raise TypeError('The method provided {method.name} cannot be used with a PI controller'
                            ' (the default step_size argument), you must set step_size equal to a constant value.')

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    try:
        coerce_dt_at_final_step = False
        if stop_at_time is not None:
            coerce_dt_at_final_step = True

        if output_times is not None:
            coerce_dt_at_final_step = True
            stop_at_time = output_times[-1]
            output_states = zeros((output_times.size, initial_state.size))
            output_time_idx = 0
            if output_times[0] < 1e-14:
                output_states[output_time_idx, :] = np.copy(initial_state)
                output_time_idx += 1

        stop_at_steady_state = stop_at_steady is not None
        if stop_at_steady_state:
            steady_tolerance = 1.e-4 if isinstance(stop_at_steady, bool) else stop_at_steady

        use_finite_difference_jacobian = method.is_implicit and linear_setup is None
        build_projector_in_governor = (method.is_implicit and method.nonlinear_solver.setup_projector_in_governor) or \
                                      use_finite_difference_jacobian
        eval_linear_setup = True

        log_count = 0
        log_title_count = 0
        linear_setup_count = 0

        if isinstance(step_size, float):
            time_step_value = step_size
            step_size = ConstantTimeStep(time_step_value)

        _log_header(verbose, method.name)

        current_state = np.copy(initial_state)
        current_time = np.copy(initial_time)

        if save_each_step:
            t_list = [np.copy(current_time)]
            solution_list = [np.copy(current_state)]

        number_of_time_steps = 0
        smallest_step_size = 1.e305
        largest_step_size = 0.
        time_step_size = step_size.first_step_size()
        smallest_step_size = min([smallest_step_size, time_step_size])
        largest_step_size = max([largest_step_size, time_step_size])

        number_nonlinear_iter = 0
        number_linear_iter = 0
        number_projector_setup = 0

        if use_finite_difference_jacobian:
            rhs_from_state = lambda state: right_hand_side(current_time, state)

            class DefaultLinearSolver:
                def __init__(self):
                    self.lapack_lhs_factor = None
                    self.diag_indices = diag_indices(initial_state.size)

                def setup(self, t, state, prefactor):
                    j = finite_difference_jacobian(rhs_from_state, rhs_from_state(state), state) * prefactor
                    j[self.diag_indices] -= 1.
                    self.lapack_lhs_factor = lapack_lu_factor(j)[:2]

                def solve(self, residual):
                    return lapack_lu_solve(self.lapack_lhs_factor[0],
                                           self.lapack_lhs_factor[1],
                                           residual)[0], 1, True

            dls = DefaultLinearSolver()
            linear_setup = dls.setup
            linear_solve = dls.solve

        continue_time_stepping = True

        if method.is_implicit:
            method_coeff = method.implicit_coefficient

        cpu_time_0 = timer.perf_counter()
        while continue_time_stepping:

            if output_times is not None and current_time + time_step_size > output_times[output_time_idx]:
                time_step_size = output_times[output_time_idx] - current_time

            if coerce_dt_at_final_step and current_time + time_step_size > stop_at_time:
                time_step_size = stop_at_time - current_time

            if pre_step_callback is not None:
                pre_step_callback(current_time, current_state, number_of_time_steps)

            if method.is_implicit:
                if build_projector_in_governor and eval_linear_setup:
                    linear_setup(current_time, current_state, time_step_size * method_coeff)
                    number_projector_setup += 1
            linear_setup_count += 1

            if mass_matvec is None:
                mass_matvec = lambda x, *args: x

            step_output = method.single_step(current_state,
                                             current_time,
                                             time_step_size,
                                             right_hand_side,
                                             lambda t, x: linear_setup(t, x, time_step_size * method_coeff),
                                             linear_solve,
                                             mass_setup,
                                             mass_matvec)
            dstate = step_output.solution_update
            time_error = step_output.temporal_error
            nliter = step_output.nonlinear_iter
            liter = step_output.linear_iter
            nlsuccess = step_output.nonlinear_converged
            nlisslow = step_output.slow_nonlinear_convergence
            number_projector_setup += step_output.projector_setups if step_output.projector_setups is not None else 0
            residual = norm(dstate * norm_weighting, ord=np.Inf) / time_step_size

            recent_time_step_size = time_step_size
            if _check_state_update(debug_verbose, current_state, dstate,
                                   time_error, step_size.target_error(), nlsuccess,
                                   strict_temporal_error_control,
                                   nonlinear_solve_must_converge,
                                   step_update_callback):
                current_state += dstate
                current_time += time_step_size
                number_of_time_steps += 1
                number_nonlinear_iter = number_nonlinear_iter + nliter if nliter is not None else 'n/a'
                number_linear_iter = number_linear_iter + liter if liter is not None else 'n/a'

                if post_step_callback is not None:
                    psco = post_step_callback(current_time, current_state, residual, number_of_time_steps)
                    current_state = current_state if psco is None else np.copy(psco)

                if output_times is not None and current_time >= output_times[output_time_idx]:
                    output_states[output_time_idx, :] = np.copy(current_state)
                    output_time_idx += 1

                if save_each_step:
                    if isinstance(save_each_step, bool) or \
                            (isinstance(save_each_step, int) and not (number_of_time_steps % save_each_step)):
                        t_list.append(np.copy(current_time))
                        solution_list.append(np.copy(current_state))

                log_count, log_title_count = _write_log(verbose,
                                                        show_solver_stats_in_situ,
                                                        log_count,
                                                        log_rate,
                                                        log_title_count,
                                                        log_lines_per_header,
                                                        extra_logger_title_line1,
                                                        extra_logger_title_line2,
                                                        extra_logger_log,
                                                        current_state,
                                                        current_time,
                                                        time_step_size,
                                                        residual,
                                                        number_of_time_steps,
                                                        number_nonlinear_iter,
                                                        number_linear_iter,
                                                        number_projector_setup,
                                                        timer.perf_counter() - cpu_time_0)

                time_step_size = step_size(number_of_time_steps, time_step_size, step_output)

                if method.is_implicit:
                    eval_linear_setup, time_step_size, linear_setup_count = _check_linear_setup(debug_verbose,
                                                                                                nlsuccess,
                                                                                                nlisslow,
                                                                                                time_step_size,
                                                                                                recent_time_step_size,
                                                                                                linear_setup_count,
                                                                                                linear_setup_rate,
                                                                                                time_step_reduction_factor_on_slow_solve,
                                                                                                time_step_reduction_factor_on_failure,
                                                                                                time_step_increase_factor_to_force_jacobian,
                                                                                                time_step_decrease_factor_to_force_jacobian)

            else:
                time_step_size, eval_linear_setup = _process_failed_step(debug_verbose,
                                                                         time_step_size,
                                                                         return_on_failed_step,
                                                                         time_step_reduction_factor_on_failure,
                                                                         warn_on_failed_step)

            smallest_step_size = min([smallest_step_size, time_step_size])
            largest_step_size = max([largest_step_size, time_step_size])

            if stop_criteria is not None:
                continue_time_stepping = not stop_criteria(current_time, current_state,
                                                           residual, number_of_time_steps)

            if number_of_time_steps < minimum_time_step_count:
                continue_time_stepping = True
            if number_of_time_steps > maximum_time_step_count:
                continue_time_stepping = False
            if coerce_dt_at_final_step and current_time >= stop_at_time:
                continue_time_stepping = False
            if stop_at_steady_state and residual < steady_tolerance:
                continue_time_stepping = False

        total_runtime = timer.perf_counter() - cpu_time_0

        if verbose:
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
            if method.is_implicit:
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

        _log_footer(verbose, total_runtime)
        stats_dict = {'success': True,
                      'time steps': number_of_time_steps,
                      'simulation time': current_time,
                      'total cpu time (s)': total_runtime}
        if method.is_implicit:
            stats_dict.update({'nonlinear iter': number_nonlinear_iter,
                               'linear iter': number_linear_iter,
                               'Jacobian setups': number_projector_setup})

    except Exception as error:
        stats_dict = {'success': False}
        if verbose:
            print(f'Spitfire odesolve caught the following Exception during time integration:\n')
        logger.exception(error)
        logging.disable(level=logging.DEBUG)
        if throw_on_failure:
            raise ValueError('odesolve failed to integrate the system due to an Exception being caught - see above')

    logging.disable(level=logging.DEBUG)

    if output_times is not None:
        if return_info:
            return output_states, stats_dict
        else:
            return output_states
    elif save_each_step:
        if return_info:
            return array(t_list), array(solution_list), stats_dict
        else:
            return array(t_list), array(solution_list)
    else:
        if return_info:
            return current_state, current_time, time_step_size, stats_dict
        else:
            return current_state, current_time, time_step_size
