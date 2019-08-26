"""
This module contains controllers for adaptive time stepping based on embedded temporal error estimation.
"""

"""
Copyright (c) 2018-2019 Michael Alan Hansen - All Rights Reserved
You may use, distribute and modify this code under the terms of the MIT license.

You should have received a copy of the MIT license with this file.
If not, please write to mahanse@sandia.gov or mike.hansen@chemeng.utah.edu
"""

from numpy import zeros, min, copy


class ConstantTimeStep(object):
    """A simple wrapper class for a constant time step.

    **Constructor**:

    Parameters
    ----------
    step_size : float
        the size of the time step
    """

    def __init__(self, step_size):
        self.step_size = step_size

    def __call__(self, *args, **kwargs):
        return self.step_size

    def first_step_size(self):
        """Obtain the initial step size"""
        return self.step_size

    def last_step_size(self):
        """Obtain the most recent step size"""
        return self.step_size

    def target_error(self):
        """Obtain the most target error, needed here just to avoid a base class"""
        return -1.

    def step_size_is_constant(self):
        """Whether or not this controller has a constant or variable step size"""
        return True


class PIController(object):
    """A PI controller on the embedded temporal error estimate

    **Constructor**:

    Parameters
    ----------
    kp : float
        the modal gain of the proportional control mode (default: 0.06666666667)
    ki : float
        the modal gain of the integral control mode (default: 0.1333333333)
    target_error : float
        the target error for the controller (default: 1.e-10)
    max_step : float
        the maximum allowable time step (default: 1.e-3)
    max_ramp : float
        the maximum allowable rate of increase of the time step (default: 1.1)
    first_step : float
        the initial step size (default: 1.e-6)
    """

    def __init__(self, kp=0.06666666667, ki=0.1333333333,
                 target_error=1.e-10, max_step=1.e-3,
                 max_ramp=1.1, first_step=1.e-6):
        self._kp = kp
        self._ki = ki
        self._target_error = target_error
        self._max_step = max_step
        self._max_ramp = max_ramp
        self._first_step = first_step
        self._number_of_old_values = 2
        self._err_history = zeros(self._number_of_old_values)
        self._step_history = zeros(self._number_of_old_values)

    def __call__(self, step_count, step, step_output, *args, **kwargs):
        error = step_output.temporal_error
        if error < 1.e-16:
            return min([step * self._max_ramp, self._max_step])
        if step_count < self._number_of_old_values - 1:
            self._err_history[step_count] = error
            self._step_history[step_count] = step
        else:
            self._err_history[:-1] = self._err_history[1:]
            self._step_history[:-1] = self._step_history[1:]
            self._err_history[-1] = error
            self._step_history[-1] = step
        if step_count == 0:
            mod = min([self._max_ramp, (self._target_error / error) ** self._ki])
        else:
            mod = min([self._max_ramp,
                       (self._target_error / error) ** self._ki * (self._err_history[-1] / error) ** self._kp])
        return min([step * mod, self._max_step])

    def first_step_size(self):
        """Obtain the initial step size"""
        return self._first_step

    def last_step_size(self):
        """Obtain the most recent step size"""
        return self._step_history[-1]

    def target_error(self):
        """Obtain the most target error, needed here just to avoid a base class"""
        return self._target_error

    def step_size_is_constant(self):
        """Whether or not this controller has a constant or variable step size"""
        return False


class CascadeController(object):
    """A two-level cascade control system on the embedded temporal error estimate and the ratio of two estimates.

        The stepper method must support multiple embedded error estimates.

    **Constructor**:

    Parameters
    ----------
    kp : float
        the modal gain of the proportional control mode for the error controller (default: 0.06666666667)
    ki : float
        the modal gain of the integral control mode for the error controller (default: 0.1333333333)
    ratio_kp : float
        the modal gain of the proportional control mode for the ratio controller (default: 0.1)
    ratio_ki : float
        the modal gain of the integral control mode for the ratio controller (default: 0.3)
    target_ratio : float
        the value of the target ratio for the controller (default: 1.e-2)
    initial_target_error : float
        the initial value of the target error for the controller (default: 1.e-10)
    max_step : float
        the maximum allowable time step (default: 1.e-3)
    max_ramp : float
        the maximum allowable rate of increase of the time step (default: 1.1)
    first_step : float
        the initial step size (default: 1.e-6)
    """

    def __init__(self, kp=0.06666666667, ki=0.1333333333,
                 ratio_kp=0.1, ratio_ki=0.3,
                 initial_target_error=1.e-10, target_ratio=1.e-2,
                 max_step=1.e-3, max_ramp=1.1, first_step=1.e-6):
        self._kp = kp
        self._ki = ki
        self._ratio_kp = ratio_kp
        self._ratio_ki = ratio_ki
        self._target_ratio = target_ratio
        self._initial_target_error = initial_target_error
        self._target_error = copy(initial_target_error)
        self._max_step = max_step
        self._max_ramp = max_ramp
        self._first_step = first_step
        self._number_of_old_values = 2
        self._ratio_history = zeros(self._number_of_old_values)
        self._err_history = zeros(self._number_of_old_values)
        self._step_history = zeros(self._number_of_old_values)

    def __call__(self, step_count, step, step_output, *args, **kwargs):
        error = step_output.temporal_error
        ratio = error / (1.e-12 + step_output.extra_errors[0])
        if error < 1.e-16:
            return min([step * self._max_ramp, self._max_step])
        if step_count < self._number_of_old_values - 1:
            self._err_history[step_count] = error
            self._ratio_history[step_count] = ratio
            self._step_history[step_count] = step
        else:
            self._err_history[:-1] = self._err_history[1:]
            self._ratio_history[:-1] = self._ratio_history[1:]
            self._step_history[:-1] = self._step_history[1:]
            self._err_history[-1] = error
            self._ratio_history[-1] = ratio
            self._step_history[-1] = step
        if step_count == 0:
            mod = min([self._max_ramp, (self._target_error / error) ** self._ki])
            err_mod = (self._target_ratio / ratio) ** self._ratio_ki
        else:
            mod = min([self._max_ramp,
                       (self._target_error / error) ** self._ki * (self._err_history[-1] / error) ** self._kp])
            err_mod = min([self._max_ramp, (self._target_ratio / ratio) ** self._ratio_ki * (
                self._ratio_history[-1] / ratio) ** self._ratio_kp])
        self._target_error *= err_mod
        return min([step * mod, self._max_step])

    def first_step_size(self):
        """Obtain the initial step size"""
        return self._first_step

    def last_step_size(self):
        """Obtain the most recent step size"""
        return self._step_history[-1]

    def target_error(self):
        """Obtain the most target error, needed here just to avoid a base class"""
        return self._target_error

    def step_size_is_constant(self):
        """Whether or not this controller has a constant or variable step size"""
        return False


class RatioController(object):
    """A PI controller on the ratio of two embedded temporal error estimates

        The stepper method must support multiple embedded error estimates.

    **Constructor**:

    Parameters
    ----------
    kp : float
        the modal gain of the proportional control mode (default: 0.1)
    ki : float
        the modal gain of the integral control mode (default: 0.3)
    target_ratio : float
        the target error ratio for the controller (default: 1.e-2)
    max_step : float
        the maximum allowable time step (default: 1.e-3)
    max_ramp : float
        the maximum allowable rate of increase of the time step (default: 1.1)
    first_step : float
        the initial step size (default: 1.e-6)
    """

    def __init__(self, kp=0.1, ki=0.3,
                 target_ratio=1.e-2, max_step=1.e-3,
                 max_ramp=1.1, first_step=1.e-6):
        self._kp = kp
        self._ki = ki
        self._target_ratio = target_ratio
        self._max_step = max_step
        self._max_ramp = max_ramp
        self._first_step = first_step
        self._number_of_old_values = 2
        self._ratio_history = zeros(self._number_of_old_values)
        self._step_history = zeros(self._number_of_old_values)

    def __call__(self, step_count, step, step_output, *args, **kwargs):
        error = step_output.temporal_error
        ratio = error / (1.e-12 + step_output.extra_errors[0])
        if error < 1.e-16:
            return min([step * self._max_ramp, self._max_step])
        if step_count < self._number_of_old_values - 1:
            self._ratio_history[step_count] = ratio
            self._step_history[step_count] = step
        else:
            self._ratio_history[:-1] = self._ratio_history[1:]
            self._step_history[:-1] = self._step_history[1:]
            self._ratio_history[-1] = ratio
            self._step_history[-1] = step
        if step_count == 0:
            mod = min([self._max_ramp, (self._target_ratio / ratio) ** self._ki])
        else:
            mod = min([self._max_ramp,
                       (self._target_ratio / ratio) ** self._ki * (self._ratio_history[-1] / ratio) ** self._kp])
        return min([step * mod, self._max_step])

    def first_step_size(self):
        """Obtain the initial step size"""
        return self._first_step

    def last_step_size(self):
        """Obtain the most recent step size"""
        return self._step_history[-1]

    def target_error(self):
        """Obtain the most target error, needed here just to avoid a base class"""
        return 1.e305

    def step_size_is_constant(self):
        """Whether or not this controller has a constant or variable step size"""
        return False
