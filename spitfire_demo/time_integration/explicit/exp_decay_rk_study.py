from spitfire.time.governor import Governor, FinalTime, SaveAllDataToList
import matplotlib.pyplot as plt
from numpy import array, exp
from spitfire.time.methods import GeneralAdaptiveExplicitRungeKutta as GERK

runge_kutta_methods = dict()

runge_kutta_methods['Forward Euler'] = GERK(name='Forward Euler',
                                            A=array([[0.]]),
                                            b=array([1.]),
                                            order=1)
runge_kutta_methods['Ralston'] = GERK(name='Ralston',
                                      A=array([[0., 0.],
                                               [2. / 3., 0.]]),
                                      b=array([0.25, 0.75]),
                                      order=2)

runge_kutta_methods['Fehlberg'] = GERK(name='Fehlberg',
                                       A=array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                [2. / 27., 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                [1. / 36., 1. / 12., 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                [1. / 24, 0, 1. / 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                [5. / 12., 0, -25. / 16., 25. / 16., 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                [1. / 20., 0, 0, 1. / 4., 1. / 5., 0, 0, 0, 0, 0, 0, 0, 0],
                                                [-25. / 108, 0, 0, 125. / 108., -65. / 27., 125. / 54., 0, 0, 0, 0, 0,
                                                 0, 0],
                                                [31. / 300., 0, 0, 0, 61. / 225., -2. / 9., 13. / 900., 0, 0, 0, 0, 0,
                                                 0],
                                                [2., 0, 0, -53. / 6., 704. / 45., -107. / 9., 67. / 90., 3., 0, 0, 0, 0,
                                                 0],
                                                [-91. / 108., 0, 0, 23. / 108., -976. / 135., 311. / 54., -19. / 60.,
                                                 17. / 6., -1. / 12., 0, 0, 0, 0],
                                                [2383. / 4100., 0, 0, -341. / 164., 4496. / 1025., -301. / 82.,
                                                 2133. / 4100., 45. / 82., 45. / 164., 18. / 41., 0, 0, 0],
                                                [3. / 205., 0, 0, 0, 0, -6. / 41., -3. / 205., -3. / 41., 3. / 41.,
                                                 6. / 41., 0, 0, 0],
                                                [-1777. / 4100., 0, 0, -341. / 164., 4496. / 1025., -289. / 82.,
                                                 2193. / 4100., 51. / 82., 33. / 164., 12. / 41., 0, 1., 0]]),
                                       b=array([0, 0, 0, 0, 0, 34. / 105., 9. / 35., 9. / 35., 9. / 280., 9. / 280., 0,
                                                41. / 840., 41. / 840.]),
                                       order=8)

governor = Governor()
governor.do_logging = False
governor.termination_criteria = FinalTime(2)

dt = 0.09
k = -10.
y0 = array([1.])
rhs = lambda t, y: k * y

data = SaveAllDataToList(initial_solution=y0)
governor.custom_post_process_step = data.save_data

for name, method in runge_kutta_methods.items():
    data.reset_data(initial_solution=y0)
    governor.integrate(right_hand_side=rhs,
                       initial_condition=y0,
                       controller=dt,
                       method=method)
    plt.plot(data.t_list, data.solution_list, '-', label=name)

plt.plot(data.t_list, y0 * exp(k * data.t_list), 'o', markerfacecolor='w', label='exact')

plt.xlabel('t')
plt.ylabel('y')
plt.legend(loc='best')
plt.grid()
plt.show()
