import unittest


class Test(unittest.TestCase):
    def test(self):
        from spitfire.time.governor import Governor, CustomTermination, SaveAllDataToList
        from spitfire.time.stepcontrol import PIController
        from spitfire.time.methods import AdaptiveERK54CashKarp, ESDIRK64
        from spitfire.time.nonlinear import SimpleNewtonSolver
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

        def object_has_landed(state, *args, **kwargs):
            vel_y = state[1]
            pos_y = state[3]
            return not (pos_y < 0.5 * r and vel_y < 0)

        governor = Governor()
        governor.termination_criteria = CustomTermination(object_has_landed)
        governor.log_rate = 100
        governor.do_logging = False

        for key in drag_coeff_dict:
            cd = drag_coeff_dict[key]
            data = SaveAllDataToList(initial_solution=q0, save_frequency=10)
            governor.custom_post_process_step = data.save_data

            governor.integrate(right_hand_side=lambda t, y: right_hand_side(y, rf, cd, g, sa, m),
                               initial_condition=q0,
                               controller=PIController(),
                               method=AdaptiveERK54CashKarp())


if __name__ == '__main__':
    unittest.main()
