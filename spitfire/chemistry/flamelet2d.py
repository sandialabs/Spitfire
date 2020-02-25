import numpy as np
from scipy.special import erfinv
from scipy.sparse.linalg import LinearOperator, bicgstab


class Flamelet2D(object):
    _initializations = ['unreacted', 'equilibrium', 'linear-TY']
    _grid_types = ['uniform', 'clustered']
    _rates_sensitivity_option_dict = {'exact': 0, 'no-TBAF': 1}
    _sensitivity_transform_option_dict = {'exact': 0}

    @classmethod
    def _compute_dissipation_rate(cls,
                                  mixture_fraction,
                                  max_dissipation_rate,
                                  form='Peters'):
        """Compute the scalar dissipation rate across mixture fraction

            Parameters
            ----------
            mixture_fraction : array_like
                the locations of grid points in mixture fraction space
            max_dissipation_rate : float
                the maximum value of the dissipation rate
            form : str, optional
                the form of the dissipation rate's dependency on mixture fraction, defaults to 'Peters', which
                uses the form of N. Peters, Turbulent Combustion, 2000.
                Specifying anything else will yield a constant scalar dissipation rate.

            Returns
            -------
            x : array_like
                the scalar dissipation rate on the given mixture fraction grid
            """
        if form == 'Peters' or form == 'peters':
            x = max_dissipation_rate * np.exp(-2. * (erfinv(2. * mixture_fraction - 1.)) ** 2)
        else:
            x = np.empty_like(mixture_fraction)
            x[:] = max_dissipation_rate
        return x

    def __init__(self,
                 mech_spec,
                 initial_condition,
                 pressure,
                 stream_1,
                 stream_2,
                 stream_3,
                 max_dissipation_rate_1,
                 max_dissipation_rate_2,
                 dissipation_rate_1_form='Peters',
                 dissipation_rate_2_form='Peters',
                 grid_1=None,
                 grid_2=None,
                 rates_sensitivity_type='exact',
                 sensitivity_transform_type='exact'):
        self._constructor_arguments = locals()
        del self._constructor_arguments['self']

        self._gas = mech_spec.copy_stream(stream_1)
        self._stream_1 = stream_1
        self._stream_2 = stream_2
        self._stream_3 = stream_3
        self._pressure = pressure
        self._mechanism = mech_spec
        self._griffon = self._mechanism.griffon
        self._n_species = self._gas.n_species
        self._n_reactions = self._gas.n_reactions
        self._n_equations = self._n_species
        self._state_1 = np.hstack([stream_1.T, stream_1.Y[:-1]])
        self._state_2 = np.hstack([stream_2.T, stream_2.Y[:-1]])
        self._state_3 = np.hstack([stream_3.T, stream_3.Y[:-1]])

        self._rsopt = self._rates_sensitivity_option_dict[rates_sensitivity_type]
        self._stopt = self._sensitivity_transform_option_dict[sensitivity_transform_type]

        self._chi_1 = self._compute_dissipation_rate(grid_1, max_dissipation_rate_1, dissipation_rate_1_form)
        self._chi_2 = self._compute_dissipation_rate(grid_2, max_dissipation_rate_2, dissipation_rate_2_form)

        self._n_x = grid_1.size
        self._n_y = grid_2.size
        self._n_dof = self._n_equations * self._n_x * self._n_y  # - 4

        self._x_range = grid_1
        self._y_range = grid_2
        self._dx = grid_1[1:] - grid_1[:-1]
        self._dy = grid_2[1:] - grid_2[:-1]

        self._initial_state = np.zeros(self._n_dof)
        self._z_1 = np.zeros(self._n_x * self._n_y)
        self._z_2 = np.zeros(self._n_x * self._n_y)
        nx = self._n_x
        ny = self._n_y
        nq = self._n_equations
        nyq = ny * nq
        h1 = self._stream_1.enthalpy_mass
        h2 = self._stream_2.enthalpy_mass
        h3 = self._stream_3.enthalpy_mass
        Y1 = self._stream_1.Y
        Y2 = self._stream_2.Y
        Y3 = self._stream_3.Y
        if isinstance(initial_condition, np.ndarray):
            self._initial_state = np.copy(initial_condition)
        elif isinstance(initial_condition, str):
            for i in range(nx):
                x = self._x_range[i]
                for j in range(ny):
                    y = self._y_range[j]
                    ij_z = i * ny + j
                    ij = i * nyq + j * nq

                    if y > 1. - x:
                        z1 = 1. - y
                        z2 = 1. - x
                    else:
                        z1 = x
                        z2 = y

                    self._z_1[ij_z] = z1
                    self._z_2[ij_z] = z2

                    if initial_condition == 'linear-TY':
                        mix_state = z1 * self._state_3 + z2 * self._state_2 + (1. - z1 - z2) * self._state_1
                        self._initial_state[ij:ij + nq] = mix_state
                    else:
                        hmix = z1 * h3 + z2 * h2 + (1. - z1 - z2) * h1
                        Ymix = z1 * Y3 + z2 * Y2 + (1. - z1 - z2) * Y1
                        mix = mech_spec.stream('HPY', (hmix, pressure, Ymix))
                        if initial_condition == 'equilibrium':
                            mix.equilibrate('HP')
                        elif initial_condition == 'unreacted':
                            pass
                        else:
                            raise ValueError(
                                'invalid initial_condition string, only "equilibrium", "unreacted", and "linear-TY" are allowed')
                        self._initial_state[ij:ij + nq] = np.hstack([mix.T, mix.Y[:-1]])

        self._variable_scales = np.ones_like(self._initial_state)
        self._variable_scales[::nq] = 1.e3
        nxq = (self._n_x - 1) * self._n_equations
        nyq = (self._n_y - 1) * self._n_equations
        self._xcp = np.zeros(nxq)
        self._xcr = np.zeros(nxq)
        self._xcl = np.zeros(nxq)
        self._ycp = np.zeros(nyq)
        self._yct = np.zeros(nyq)
        self._ycb = np.zeros(nyq)
        bleh1 = np.zeros(self._n_x - 1)
        bleh2 = np.zeros(self._n_x - 1)
        self._griffon.flamelet_stencils(self._dx, self._n_x - 1, self._chi_1, np.ones(self._n_equations),
                                        self._xcp, self._xcl, self._xcr, bleh1, bleh2)
        self._griffon.flamelet_stencils(self._dy, self._n_y - 1, self._chi_2, np.ones(self._n_equations),
                                        self._ycp, self._ycb, self._yct, bleh1, bleh2)
        self._block_diag_jac_values = np.zeros(nx * ny * nq * nq)
        self._block_diag_jac_factors = np.zeros(nx * ny * nq * nq)
        self._block_diag_jac_pivots = np.zeros(nx * ny * nq, dtype=np.int32)
        self._jacobi_prefactor = None
        self._iteration_count = 0

    def rhs(self, t, state):
        r = np.zeros_like(state)
        self._griffon.flamelet2d_rhs(state, self._pressure, self._n_x, self._n_y, self._xcp, self._xcl, self._xcr,
                                     self._ycp, self._ycb, self._yct, r)
        return r

    def block_Jacobi_setup(self, state, prefactor):
        self._jacobi_prefactor = prefactor
        self._griffon.flamelet2d_factored_block_diag_jacobian(state, self._pressure,
                                                              self._n_x, self._n_y, self._xcp, self._ycp, prefactor,
                                                              self._block_diag_jac_values,
                                                              self._block_diag_jac_factors,
                                                              self._block_diag_jac_pivots)

    def block_Jacobi_solve(self, b):
        nx = self._n_x
        ny = self._n_y
        xcp = self._xcp
        xcl = self._xcl
        xcr = self._xcr
        ycp = self._ycp
        ycb = self._ycb
        yct = self._yct
        pf = self._jacobi_prefactor
        dj = self._block_diag_jac_values
        df = self._block_diag_jac_factors
        dp = self._block_diag_jac_pivots
        ax = np.zeros_like(b)
        rx = np.zeros_like(b)
        x = np.zeros_like(b)
        inv_dof_scales = 1. / self._variable_scales

        tol = 1.e-8
        maxiter = 16
        err = tol + 1.
        iteration = 0
        while err > tol and iteration < maxiter:
            self._griffon.flamelet2d_offdiag_matvec(x, nx, ny, xcp, xcl, xcr, ycp, ycb, yct, pf, rx)
            self._griffon.flamelet2d_block_diag_solve(nx, ny, df, dp, b - rx, x)
            self._griffon.flamelet2d_matvec(x, nx, ny, xcp, xcl, xcr, ycp, ycb, yct, pf, dj, ax)
            err = (b - ax) * inv_dof_scales
            err = np.sqrt(err.dot(err))
            iteration += 1
        return x, iteration, iteration < maxiter

    def matvec(self, v):
        nx = self._n_x
        ny = self._n_y
        xcp = self._xcp
        xcl = self._xcl
        xcr = self._xcr
        ycp = self._ycp
        ycb = self._ycb
        yct = self._yct
        pf = self._jacobi_prefactor
        dj = self._block_diag_jac_values
        av = np.zeros_like(v)
        self._griffon.flamelet2d_matvec(v, nx, ny, xcp, xcl, xcr, ycp, ycb, yct, pf, dj, av)
        return av

    def block_diag_solve(self, b):
        nx = self._n_x
        ny = self._n_y
        df = self._block_diag_jac_factors
        dp = self._block_diag_jac_pivots
        x = np.zeros_like(b)
        self._griffon.flamelet2d_block_diag_solve(nx, ny, df, dp, b, x)
        return x

    def _increment_iteration_count(self, *args, **kwargs):
        self._iteration_count += 1

    def block_Jacobi_prec_bicgstab_solve(self, b):
        a = LinearOperator((self._n_dof, self._n_dof), lambda v: self.matvec(v))
        p = LinearOperator((self._n_dof, self._n_dof), lambda rhs: self.block_diag_solve(rhs))
        self._iteration_count = 0
        x, i = bicgstab(a, b, M=p, tol=1.e-14, maxiter=8, callback=self._increment_iteration_count)
        return x, self._iteration_count, not i
