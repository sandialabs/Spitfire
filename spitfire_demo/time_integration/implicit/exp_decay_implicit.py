from spitfire.time.governor import Governor, FinalTime, SaveAllDataToList
from spitfire.time.methods import ForwardEuler, BackwardEuler
from spitfire.time.nonlinear import SimpleNewtonSolver
import matplotlib.pyplot as plt
from numpy import array, exp

governor = Governor()
governor.termination_criteria = FinalTime(2)

dt = 0.02
k = -10.
y0 = array([1.])
rhs = lambda t, y: k * y

data = SaveAllDataToList(initial_solution=y0)
governor.custom_post_process_step = data.save_data

governor.integrate(right_hand_side=rhs,
                   initial_condition=y0,
                   controller=dt,
                   method=ForwardEuler())
plt.plot(data.t_list, data.solution_list, '--', label='Forward Euler')

data.reset_data(initial_solution=y0)
governor.integrate(right_hand_side=rhs,
                   initial_condition=y0,
                   controller=dt,
                   method=BackwardEuler(SimpleNewtonSolver()))
plt.plot(data.t_list, data.solution_list, '-', label='Backward Euler')

plt.plot(data.t_list, y0 * exp(k * data.t_list), '-.', label='exact')

plt.xlabel('t')
plt.ylabel('y')
plt.legend(loc='best')
plt.grid()
plt.show()
