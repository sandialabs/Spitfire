import unittest


class Test(unittest.TestCase):
    def test(self):
        from spitfire.chemistry.general_diff_rxn import DiffusionReaction1D_FiniteDifference
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
            c_c = c[2]
            q_1 = k_ab * c_a
            q_2 = k_bc * c_a * c_b
            return np.array([-q_1 - q_2, q_1 - q_2, 2. * q_2])

        def jacobian(c, k_ab, k_bc):
            c_a = c[0]
            c_b = c[1]
            c_c = c[2]
            q_1 = k_ab * c_a
            q_2 = k_bc * c_a * c_b
            dq1_ca = k_ab
            dq1_cb = 0.
            dq1_cc = 0.
            dq2_ca = k_bc * c_b
            dq2_cb = k_bc * c_a
            dq2_cc = 0.
            return np.array([[-dq1_ca - q_2, -dq1_cb - dq2_cb, -dq1_cc - dq2_cc],
                             [dq1_ca - q_2, dq1_cb - dq2_cb, dq1_cc - dq2_cc],
                             [2. * dq2_ca, 2. * dq2_cb, 2. * dq2_cc]])

        k_ab = 1.
        k_bc = 10.

        dr = DiffusionReaction1D_FiniteDifference(initial_conditions=[lambda x: np.exp(-100. * x ** 2),
                                                                      lambda x: 0.5 * (1. + np.sin(2. * np.pi * x)),
                                                                      0.],
                                                  left_boundary_state=[1., 1., 0.],
                                                  right_boundary_state=[0., 1., 0.],
                                                  diffusion_coefficient=lambda x: np.exp(-1.e0 * (x - 0.2) ** 2),
                                                  source_term=lambda y: right_hand_side(y, k_ab, k_bc),
                                                  source_term_jacobian=lambda y: jacobian(y, k_ab, k_bc),
                                                  variable_names=['A', 'B', 'C'],
                                                  grid_points=32,
                                                  grid_type='clustered',
                                                  grid_cluster_intensity=2.,
                                                  grid_cluster_point=0.2)

        dr.integrate_to_time(final_time=0.2,
                             first_time_step=1.e-5,
                             time_step=1.e-5,
                             time_method='forward euler',
                             log_rate=40, write_log=False,
                             transient_tolerance=1.e-8)
        dr.integrate_to_time(final_time=0.2,
                             first_time_step=1.e-4,
                             time_method='erk54 cash karp',
                             log_rate=40, write_log=False,
                             transient_tolerance=1.e-8)
        dr.integrate_to_time(final_time=0.2,
                             first_time_step=1.e-3,
                             time_method='esdirk64',
                             log_rate=10, write_log=False,
                             transient_tolerance=1.e-8)

        ca = dr.trajectory_data('A')
        cb = dr.trajectory_data('B')
        cc = dr.trajectory_data('C')
        x = dr.grid[1:-1]


if __name__ == '__main__':
    unittest.main()
