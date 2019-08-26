"""
Trajectory calculation demonstration

This code demonstrates how to use Spitfire to solve an ODE system governing ballistics with air resistance,
including adaptive time-stepping and a custom termination rule: integration is performed until the object lands.
"""

"""
Copyright (c) 2018-2019 Michael Alan Hansen - All Rights Reserved
You may use, distribute and modify this code under the terms of the MIT license.

You should have received a copy of the MIT license with this file.
If not, please write to mahanse@sandia.gov or mike.hansen@chemeng.utah.edu
"""

from spitfire.time.governor import Governor, CustomTermination, SaveAllDataToList
from spitfire.time.stepcontrol import PIController
from spitfire.time.methods import AdaptiveERK54CashKarp, ESDIRK64
from spitfire.time.nonlinear import SimpleNewtonSolver
import numpy as np
import matplotlib.pyplot as plt


def right_hand_side(q, fluid_density, drag_coeff, gravity, surface_area, mass):
    """
    Computes the right-hand side function for the ODE system.

    Note that time integration requires a function that takes (t, y) as arguments.
    To accomodate this, we will write a lambda after defining the rate constants,
    which passes the appropriate y value and rate constant to this function (and ignores the time).

    :param q: current vector of velocities and positions
    :param fluid_density: fluid density
    :param drag_coeff: drag coefficient of the flying object
    :param gravity: gravitational constant
    :param surface_area: surface area of the flying object
    :param mass: mass of the flying object
    :return: right-hand side of the ODE system
    """
    vel_x = q[0]
    vel_y = q[1]
    f = fluid_density * surface_area * drag_coeff / mass
    return np.array([-f * vel_x * vel_x,
                     -f * vel_y * vel_y - gravity,
                     vel_x,
                     vel_y])


q0 = np.array([1., 10., 0., 0.])  # initial condition

rf = 1.23  # fluid density, kg/m3
ro = 7.86e3  # object (cannonball) density, kg/m3
g = 9.8  # gravitational constant, m/s2
r = 4. * 2.54 / 100.  # object radius, m
sa = 4. * np.pi * r * r
m = ro * np.pi * r * r * r * 4. / 3.

drag_coeff_dict = {'no drag': 0.0,
                   'weak drag': 4.0,
                   'strong drag': 40.0}


def object_has_landed(state, *args, **kwargs):
    """
    Enables us to integrate until the object has landed

    :param state: the state vector for time integration
    :param args: additional positional arguments
    :param kwargs: additional keyword arguments
    :return: True if the object has not yet landed
    """
    vel_y = state[1]
    pos_y = state[3]
    return not (pos_y < 0.5 * r and vel_y < 0)


governor = Governor()
governor.termination_criteria = CustomTermination(object_has_landed)
governor.log_rate = 100

for key in drag_coeff_dict:
    cd = drag_coeff_dict[key]
    data = SaveAllDataToList(initial_solution=q0, save_frequency=10)
    governor.custom_post_process_step = data.save_data

    governor.integrate(right_hand_side=lambda t, y: right_hand_side(y, rf, cd, g, sa, m),
                       initial_condition=q0,
                       controller=PIController(),
                       method=AdaptiveERK54CashKarp())
    plt.plot(data.solution_list[:, 2], data.solution_list[:, 3], label=key + ', explicit')

    data = SaveAllDataToList(initial_solution=q0, save_frequency=10)
    governor.custom_post_process_step = data.save_data
    governor.integrate(right_hand_side=lambda t, y: right_hand_side(y, rf, cd, g, sa, m),
                       initial_condition=q0,
                       controller=PIController(),
                       method=ESDIRK64(SimpleNewtonSolver()))
    plt.plot(data.solution_list[:, 2], data.solution_list[:, 3], '--', label=key + ', implicit')

plt.grid()
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.show()
