from spitfire.chemistry.flamelet2d import _Flamelet2D
from spitfire.chemistry.flamelet import Flamelet
from spitfire.chemistry.mechanism import ChemicalMechanismSpec
from spitfire.time.integrator import Governor, NumberOfTimeSteps, FinalTime, Steady, SaveAllDataToList
from spitfire.time.methods import AdaptiveERK54CashKarp, ESDIRK64, BackwardEulerWithError
from spitfire.time.nonlinear import SimpleNewtonSolver
from spitfire.time.stepcontrol import PIController
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

m = ChemicalMechanismSpec(cantera_xml='coh2-hawkes.xml', group_name='coh2-hawkes')
print(m.n_species, m.n_reactions)

pressure = 101325.

air = m.stream(stp_air=True)
air.TP = 300., pressure
synthesis_gas = m.stream('X', 'H2:1, CO:1')
exhaust_gas = m.stream('X', 'CO2:1, H2O:1, CO:0.5, H2:0.001')

fuel1 = m.copy_stream(synthesis_gas)
fuel1.TP = 1000., pressure
fuel2 = m.copy_stream(exhaust_gas)
fuel2.TP = 400., pressure

fuel1_name = 'SG'
fuel2_name = 'EG'

x_cp = m.stoich_mixture_fraction(fuel2, air)
y_cp = m.stoich_mixture_fraction(fuel1, air)
x_cp = x_cp if x_cp < 0.5 else 1. - x_cp
y_cp = y_cp if y_cp < 0.5 else 1. - y_cp

x_cc = 6.
y_cc = 6.


def make_clustered_grid(nx, ny, x_cp, y_cp, x_cc, y_cc):
    x_half1 = Flamelet._clustered_grid(nx // 2, x_cp * 2., x_cc)[0] * 0.5
    y_half1 = Flamelet._clustered_grid(nx // 2, y_cp * 2., y_cc)[0] * 0.5
    x_half2 = (0.5 - x_half1)[::-1]
    y_half2 = (0.5 - y_half1)[::-1]
    x_range = np.hstack((x_half1, 0.5 + x_half2))
    y_range = np.hstack((y_half1, 0.5 + y_half2))
    dx_mid = x_range[nx // 2 - 1] - x_range[nx // 2 - 2]
    x_range[nx // 2 - 1] -= dx_mid / 3.
    x_range[nx // 2 + 0] += dx_mid / 3.
    dy_mid = y_range[ny // 2 - 1] - y_range[ny // 2 - 2]
    y_range[ny // 2 - 1] -= dy_mid / 3.
    y_range[ny // 2 + 0] += dy_mid / 3.
    return x_range, y_range


nx = 32
ny = nx
x_range, y_range = make_clustered_grid(nx, ny, x_cp, y_cp, x_cc, y_cc)
x_grid, y_grid = np.meshgrid(x_range, y_range)

chi11_max = 1.
chi22_max = 1.

f = _Flamelet2D(m, 'unreacted', pressure, air, fuel1, fuel2, chi11_max, chi22_max, grid_1=x_range, grid_2=y_range)
nq = f._n_equations

phi0 = np.copy(f._initial_state)


def plot_contours(phi, variable, i):
    fig = plt.figure()
    iq = 0 if variable == 'T' else m.species_index(variable) + 1
    phi2d = phi[iq::nq].reshape((ny, nx), order='F')
    phi02d = phi0[iq::nq].reshape((ny, nx), order='F')
    ax = plt.subplot2grid((3, 4), (2, 1), rowspan=1, colspan=2)
    ax.cla()
    ax.plot(x_range, phi02d[0, :], 'b--', label='EQ')
    ax.plot(x_range, phi2d[0, :], 'g-', label='SLFM')
    Tmin = 200.
    Tmax = int(np.max(phi) // 100 + 1) * 100
    if variable == 'T':
        ax.set_ylim([Tmin, Tmax])
    ax.yaxis.tick_right()
    ax.set_xlabel('$Z_1$')
    ax.set_xlim([0, 1])
    ax.grid(True)
    # ax.legend(loc='best')
    ax.legend(loc='center left', bbox_to_anchor=(-0.4, 0.5), ncol=1, borderaxespad=0, frameon=False)
    ax = plt.subplot2grid((3, 4), (0, 0), rowspan=2, colspan=1)
    ax.cla()
    ax.plot(phi02d[:, 0], y_range, 'b--', label='EQ')
    ax.plot(phi2d[:, 0], y_range, 'g-', label='SLFM')
    if variable == 'T':
        ax.set_xlim([Tmin, Tmax])
    ax.set_ylabel('$Z_2$')
    ax.set_ylim([0, 1])
    ax.grid(True)
    # ax.legend(loc='best')
    ax = plt.subplot2grid((3, 4), (0, 1), rowspan=2, colspan=2)
    cax = plt.subplot2grid((3, 4), (0, 3), rowspan=2, colspan=1)
    ax.cla()
    if variable == 'T':
        contour = ax.contourf(x_grid, y_grid, phi2d, cmap=plt.get_cmap('magma'),
                              norm=Normalize(Tmin, Tmax), levels=np.linspace(Tmin, Tmax, 20))
    else:
        contour = ax.contourf(x_grid, y_grid, phi2d, cmap=plt.get_cmap('magma'),
                              levels=np.linspace(np.min(phi2d), np.max(phi2d), 20))
    plt.colorbar(contour, cax=cax)
    ax.plot([0, 0, 1, 0], [0, 1, 0, 0], 'k-', linewidth=0.5, zorder=4)
    # ax.contour(x_grid, y_grid, phi2d, cmap=plt.get_cmap('rainbow'),
    #            norm=Normalize(Tmin, Tmax), levels=np.linspace(Tmin, Tmax, 20))
    t1 = plt.Polygon(np.array([[1, 0], [1, 1], [0, 1]]), color='w', zorder=3)
    ax.add_patch(t1)
    ax.text(-0.1, 1.04, fuel1_name, fontdict={'fontweight': 'bold'},
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7), zorder=10)
    ax.text(0.95, -0.05, fuel2_name, fontdict={'fontweight': 'bold'},
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7), zorder=11)
    ax.text(-0.08, -0.05, 'air', fontdict={'fontweight': 'bold'},
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7), zorder=12)
    # ax.set_xlabel('$Z_1$')
    # ax.set_ylabel('$Z_2$', rotation=0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True)
    cax.set_title('SLFM T' if variable == 'T' else 'SLFM ' + variable)
    plt.tight_layout()
    plt.savefig(f'img_{variable}_{i}.png')


g = Governor()
g.log_rate = 10
g.clip_to_positive = True
g.norm_weighting = 1. / f._variable_scales
g.projector_setup_rate = 20
g.time_step_increase_factor_to_force_jacobian = 1.1
g.time_step_decrease_factor_to_force_jacobian = 0.8
data = SaveAllDataToList(initial_solution=phi0,
                         save_frequency=100,
                         file_prefix='ip',
                         file_first_and_last_only=True,
                         save_first_and_last_only=True)
g.custom_post_process_step = data.save_data
newton = SimpleNewtonSolver(evaluate_jacobian_every_iter=False,
                            norm_weighting=g.norm_weighting,
                            tolerance=1.e-12,
                            max_nonlinear_iter=8)
esdirk = ESDIRK64(norm_weighting=g.norm_weighting, nonlinear_solver=newton)
pi = PIController(first_step=1.e-8, target_error=1.e-10, max_step=1.e0)

viz_dt = 1.e-3
viz_nt = 100

plot_contours(phi0, 'T', 'ic')
phi = np.copy(phi0)
dt = 1.e-8
for i in range(viz_nt):
    g.termination_criteria = FinalTime((i + 1) * viz_dt)
    pi._first_step = dt
    _, phi, _, dt = g.integrate(right_hand_side=f.rhs,
                                linear_setup=f.block_Jacobi_setup,
                                linear_solve=f.block_Jacobi_solve,
                                initial_condition=phi,
                                method=esdirk,
                                controller=pi,
                                initial_time=i * viz_dt)
    plot_contours(phi, 'T', i)
