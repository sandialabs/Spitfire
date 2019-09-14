from spitfire.time.governor import Governor, FinalTime, SaveAllDataToList
from spitfire.time.methods import ExplicitRungeKutta4Classical
import matplotlib.pyplot as plt
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
    q_1 = k_ab * c_a
    q_2 = k_bc * c_a * c_b
    return np.array([-q_1 - q_2, q_1 - q_2, 2. * q_2])


c0 = np.array([1., 0., 0.])  # initial condition
k_ab = 1.  # A -> B rate constant
k_bc = 0.2  # A + B -> 2C rate constant
final_time = 6.  # final time to integrate to
time_step_size = 0.01  # size of the time step used

governor = Governor()
governor.termination_criteria = FinalTime(final_time)
governor.log_rate = 100

data = SaveAllDataToList(initial_solution=c0)
governor.custom_post_process_step = data.save_data

governor.integrate(right_hand_side=lambda t, y: right_hand_side(y, k_ab, k_bc),
                   initial_condition=c0,
                   controller=time_step_size,
                   method=ExplicitRungeKutta4Classical())

plt.plot(data.t_list, data.solution_list[:, 0], label='A')
plt.plot(data.t_list, data.solution_list[:, 1], label='B')
plt.plot(data.t_list, data.solution_list[:, 2], label='C')
plt.legend(loc='best')
plt.grid()
plt.xlabel('t')
plt.ylabel('species concentration')
plt.show()
