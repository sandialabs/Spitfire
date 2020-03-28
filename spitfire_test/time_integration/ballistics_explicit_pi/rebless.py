import pickle


def run():
    from spitfire.time.integrator import odesolve, SaveAllDataToList
    from spitfire.time.stepcontrol import PIController
    from spitfire.time.methods import AdaptiveERK54CashKarp
    import numpy as np

    def right_hand_side(q, fluid_density, drag_coeff, gravity, surface_area, mass):
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

    def object_has_landed(t, state, *args, **kwargs):
        vel_y = state[1]
        pos_y = state[3]
        return pos_y < 0.5 * r and vel_y < 0

    solution_data = dict()

    for key in drag_coeff_dict:
        cd = drag_coeff_dict[key]

        data = SaveAllDataToList(initial_solution=q0, save_frequency=10)

        odesolve(lambda t, y: right_hand_side(y, rf, cd, g, sa, m),
                 q0,
                 step_size=PIController(first_step=1e-6, target_error=1e-10, max_step=1e-3),
                 method=AdaptiveERK54CashKarp(),
                 post_step_callback=data.save_data,
                 stop_criteria=object_has_landed)

        solution_data[key] = (data.t_list.copy(), data.solution_list.copy())
        data.reset_data(q0, 0.)

    return solution_data


if __name__ == '__main__':
    output = run()
    with open('gold.pkl', 'wb') as file_output:
        pickle.dump(output, file_output)
