import pickle


def run():
    from spitfire.time.integrator import odesolve
    from spitfire.time.methods import RK4ClassicalS4P4
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

    c0 = np.array([1., 0., 0.])  # initial condition
    k_ab = 1.  # A -> B rate constant
    k_bc = 0.2  # A + B -> 2C rate constant
    final_time = 10.  # final time to integrate to
    time_step_size = 0.1  # size of the time step used

    t, sol = odesolve(lambda t, y: right_hand_side(y, k_ab, k_bc),
                      c0,
                      stop_at_time=final_time,
                      step_size=time_step_size,
                      method=RK4ClassicalS4P4(),
                      save_each_step=True)

    return dict({'t': t.copy(), 'sol': sol.copy()})


if __name__ == '__main__':
    output = run()
    with open('gold.pkl', 'wb') as file_output:
        pickle.dump(output, file_output)
