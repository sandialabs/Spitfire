"""
Exponential decay demonstration

This code demonstrates how to use Spitfire to solve a very simple ordinary differential equation (ODE)
with the Forward Euler and classical explicit 4th-order Runge-Kutta methods.
"""

"""
Copyright (c) 2018-2019 Michael Alan Hansen - All Rights Reserved
You may use, distribute and modify this code under the terms of the MIT license.

You should have received a copy of the MIT license with this file.
If not, please write to mahanse@sandia.gov or mike.hansen@chemeng.utah.edu
"""

from spitfire.time.governor import Governor, FinalTime, SaveAllDataToList
from spitfire.time.methods import ForwardEuler, ExplicitRungeKutta4Classical, BackwardEuler
from spitfire.time.nonlinear import SimpleNewtonSolver
from numpy import exp, linspace, array, copy
import matplotlib.pyplot as plt


def right_hand_side(yvalue, rate_constant):
    """
    Computes the right-hand side function for the exponential decay ODE.

    Note that time integration requires a function that takes (t, y) as arguments.
    To accomodate this, we will write a lambda after defining the rate constant (k),
    which passes the appropriate y value and rate constant to this function (and ignores the time).

    :param yvalue: current solution
    :param rate_constant: the rate constant of the decay - this should be a negative number for decay to occur!
    :return: RHS of the exponential decay ODE
    """
    return rate_constant * yvalue


def exact_solution(time, initial_value, rate_constant):
    """
    Computes the exact solution to the exponential decay ODE.

    :param time: a numpy.array of times at which the exact solution will be evaluated
    :param initial_value: the initial condition of the ODE
    :param rate_constant: the rate contant of the ODE
    :return: the exact solution evaluated at the specified array of times
    """
    return initial_value * exp(rate_constant * time)


# problem specifications
y0 = array([1.])  # initial condition
k = -1.  # rate constant
final_time = 10.  # final time to integrate to
time_step_size = 0.5  # size of the time step used

# build a governor object, tell it to stop stepping when it reaches the specified final time
governor = Governor()
governor.termination_criteria = FinalTime(final_time)

# make an object for saving off the solution and time during time integration, and tell the governor to use it
data = SaveAllDataToList(initial_solution=y0)
governor.custom_post_process_step = data.save_data

# make a dictionary to save off solutions, one for Forward Euler, one for RK4
solution_dict = {}

# perform the integration, first with the Forward Euler method
# note that we supply several things to the integrate() method:
#  - right_hand_side: this is the RHS of the ODE, which we build with a lambda as noted above in the RHS docstring
#  - initial_condition: the initial value of the solution variable
#  - controller: this is how the time step is controlled, but in this case we just want a fixed step size
#  - method: the integration method used, in this case Forward Euler
governor.integrate(right_hand_side=lambda t, y: right_hand_side(y, k),
                   initial_condition=y0,
                   controller=time_step_size,
                   method=ForwardEuler())

# save the solution for Forward Euler
solution_dict['FE'] = {'t': copy(data.t_list),
                       'y': copy(data.solution_list)}

# integrate the same problem and save the solution, this time with the RK4 method instead of Forward Euler
data.reset_data(initial_solution=y0)  # we could also make a new data storage object instead
governor.integrate(right_hand_side=lambda t, y: right_hand_side(y, k),
                   initial_condition=y0,
                   controller=time_step_size,
                   method=ExplicitRungeKutta4Classical())
solution_dict['RK4'] = {'t': copy(data.t_list),
                        'y': copy(data.solution_list)}

# integrate the same problem and save the solution, this time with the Backward Euler method, an implicit solver
data.reset_data(initial_solution=y0)  # we could also make a new data storage object instead
governor.integrate(right_hand_side=lambda t, y: right_hand_side(y, k),
                   initial_condition=y0,
                   controller=time_step_size,
                   method=BackwardEuler(SimpleNewtonSolver()))
solution_dict['BE'] = {'t': copy(data.t_list),
                        'y': copy(data.solution_list)}

# plot the approximate solutions
for name, marker in [('FE', 'o'),
                     ('RK4', 'D'),
                     ('BE', 's')]:
    plt.plot(solution_dict[name]['t'], solution_dict[name]['y'], '--', label=name, marker=marker)

# plot the exact solution on a fine grid so it appears smooth
t_continuous = linspace(0., final_time, num=1000)
plt.plot(t_continuous, exact_solution(t_continuous, y0, k), '-', label='exact')

# make the plot pretty
plt.grid()
plt.legend()
plt.xlabel('t')
plt.ylabel('y')
plt.show()
