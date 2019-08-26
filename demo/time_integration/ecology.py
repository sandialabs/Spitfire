"""
Lotka-Volterra ecology model demonstration

This code demonstrates how to use Spitfire to solve an ODE system governing the population dynamics of
a predator-prey system with two different prey species.
"""

"""
Copyright (c) 2018-2019 Michael Alan Hansen - All Rights Reserved
You may use, distribute and modify this code under the terms of the MIT license.

You should have received a copy of the MIT license with this file.
If not, please write to mahanse@sandia.gov or mike.hansen@chemeng.utah.edu
"""

from spitfire.time.governor import Governor, FinalTime, SaveAllDataToList
from spitfire.time.methods import ExplicitRungeKutta4Classical, ESDIRK64
from spitfire.time.nonlinear import SimpleNewtonSolver
import numpy as np
import matplotlib.pyplot as plt


class LotkaVolterraParameters(object):
    def __init__(self):
        self.r1, self.k1, self.b1, self.n1 = 1.0, 200., 0.01, 0.01
        self.r2, self.k2, self.b2, self.n2 = 4.0, 100., 0.1, 0.2
        self.d = 6.
        self.a12 = 4.
        self.a21 = 1. / self.a12


def right_hand_side(q, p):
    """
    Computes the right-hand side function for the Lotka-Volterra ODEs.

    Note that time integration requires a function that takes (t, q) as arguments.
    To accomodate this, we will write a lambda after defining the rate constants,
    which passes the appropriate y value and rate constant to this function (and ignores the time).

    :param q: current population vector
    :param p: a LotkaVolterraParameters instance
    :return: right-hand side of the ODE system
    """
    x1 = q[0]
    x2 = q[1]
    y = q[2]
    return np.array([p.r1 * x1 * (1. - (x1 + p.a12 * x2) / p.k1) - p.b1 * x1 * y,
                     p.r2 * x2 * (1. - (x2 + p.a21 * x1) / p.k2) - p.b2 * x2 * y,
                     ((p.n1 * x1 + p.n2 * x2) - p.d) * y])


x1 = 100.  # evasive prey initial population
x2 = 100.  # bulky prey initial population
y = 10.  # predator initial population
q0 = np.array([x1, x2, y])

p = LotkaVolterraParameters()

final_time = 10.  # final time to integrate to
time_step_size = 0.001  # size of the time step used

governor = Governor()
governor.termination_criteria = FinalTime(final_time)
governor.log_rate = 1000

data = SaveAllDataToList(initial_solution=q0, save_frequency=10)
governor.custom_post_process_step = data.save_data

governor.integrate(right_hand_side=lambda t, q: right_hand_side(q, p),
                   initial_condition=q0,
                   controller=time_step_size,
                   method=ExplicitRungeKutta4Classical())

# governor.integrate(right_hand_side=lambda t, q: right_hand_side(q, p),
#                    initial_condition=q0,
#                    controller=time_step_size,
#                    method=ESDIRK64(SimpleNewtonSolver()))

plt.plot(data.t_list, data.solution_list[:, 0], label='evasive prey')
plt.plot(data.t_list, data.solution_list[:, 1], label='bulky prey')
plt.plot(data.t_list, data.solution_list[:, 2], label='predator')

plt.grid()
plt.legend()
plt.xlabel('t')
plt.ylabel('population')
plt.show()
