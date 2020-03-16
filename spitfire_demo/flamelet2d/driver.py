from spitfire.chemistry.flamelet2d import _Flamelet2D
from spitfire.chemistry.flamelet import Flamelet
from spitfire.chemistry.mechanism import ChemicalMechanismSpec
from spitfire.time.governor import Governor, NumberOfTimeSteps, FinalTime, Steady, SaveAllDataToList
from spitfire.time.methods import AdaptiveERK54CashKarp, ESDIRK64, BackwardEulerWithError
from spitfire.time.nonlinear import SimpleNewtonSolver
from spitfire.time.stepcontrol import PIController
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d as a3
from scipy.interpolate import RectBivariateSpline

# m = ChemicalMechanismSpec(cantera_xml='methane-lu30.xml', group_name='methane-lu30')
# m = ChemicalMechanismSpec(cantera_xml='h2-burke.xml', group_name='h2-burke')
# m = ChemicalMechanismSpec(cantera_xml='coh2-hawkes.xml', group_name='coh2-hawkes')
m = ChemicalMechanismSpec(cantera_xml='heptane-liu-hewson-chen-pitsch-highT.xml', group_name='gas')
# m = ChemicalMechanismSpec(cantera_xml='dme-bhagatwala.xml', group_name='dme-bhagatwala')
# m = ChemicalMechanismSpec(cantera_xml='ethylene-luo.xml', group_name='ethylene-luo')
pressure = 101325.

print(m.n_species, m.n_reactions)

air = m.stream(stp_air=True)
air.TP = 300., pressure
n2 = m.stream('X', 'N2:1')
c7 = m.stream('X', 'NXC7H16:1')
# dme = m.stream('X', 'CH3OCH3:1')
# ch4 = m.stream('X', 'CH4:1')
# sg = m.stream('X', 'H2:1, CO:2')
sg = m.stream('X', 'H2:1, CO:1, CH4:1, CO2:1')
# eg = m.stream('X', 'CO2:1, H2O:1, CO:0.5, H2:0.001')
# c2h4 = m.stream('X', 'C2H4:1')

# fuel1 = m.mix_fuels_for_stoich_mixture_fraction(sg, n2, 0.1, air)
fuel1 = m.copy_stream(c7)
fuel1.TP = 485., pressure
# fuel1 = m.stream(stp_air=True)
# fuel1.TP = 1000., pressure

# fuel2 = m.mix_fuels_for_stoich_mixture_fraction(c2h4, n2, 0.2, air)
fuel2 = m.copy_stream(sg)
fuel2.TP = 400., pressure
# fuel2 = m.copy_stream(eg)
# fuel2.TP = 1000., pressure


x_cp = m.stoich_mixture_fraction(fuel2, air)
y_cp = m.stoich_mixture_fraction(fuel1, air)
x_cp = x_cp if x_cp < 0.5 else 1. - x_cp
y_cp = y_cp if y_cp < 0.5 else 1. - y_cp

print(x_cp, y_cp)

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


# x_range, y_range = make_clustered_grid(nx, ny, x_cp, y_cp, x_cc, y_cc)

nx = 24
ny = nx
# x_range = Flamelet._uniform_grid(nx)[0]
# y_range = Flamelet._uniform_grid(ny)[0]
x_range, y_range = make_clustered_grid(nx, ny, x_cp, y_cp, x_cc, y_cc)
x_grid, y_grid = np.meshgrid(x_range, y_range)

f = _Flamelet2D(m, 'equilibrium', pressure, air, fuel1, fuel2, 10., 10., grid_1=x_range, grid_2=y_range)

# air_hot = m.copy_stream(air)
# air_hot.TP = 1400., pressure
# air_cool = m.copy_stream(air)
# air_cool.TP = 800., pressure
# dmech4 = m.mix_fuels_for_stoich_mixture_fraction(m.stream('X', 'CH3OCH3:1, CH4:1'), n2, 0.2, air_hot)
# dmech4.TP = 300., pressure
# f = Flamelet2D(m, 'unreacted', pressure, dmech4, air_cool, air_hot, 1., 1., grid_1=x_range, grid_2=y_range)

# ethy = m.stream('TPX', (300., pressure, 'C2H4:1'))
# coh2 = m.stream('TPX', (300., pressure, 'CO:1, H2:0.01'))
# f = Flamelet2D(m, 'equilibrium', pressure, air, ethy, coh2, 0.1, 0.1, grid_1=x_range, grid_2=y_range)

phi0 = np.copy(f._initial_state)

g = Governor()
g.log_rate = 25
g.termination_criteria = Steady(1.e-6)
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
                            tolerance=1.e-10,
                            max_nonlinear_iter=8)
esdirk = ESDIRK64(norm_weighting=g.norm_weighting, nonlinear_solver=newton)
# esdirk = BackwardEulerWithError(norm_weighting=g.norm_weighting, nonlinear_solver=newton)
pi = PIController(first_step=1.e-6, target_error=1.e-8, max_step=1.e0)
# phi = g.integrate(right_hand_side=f.rhs,
#                   projector_setup=f.block_Jacobi_setup,
#                   projector_solve=f.block_Jacobi_solve,
#                   initial_condition=phi0,
#                   method=esdirk,
#                   controller=pi)[1]
# phi = g.integrate(right_hand_side=f.rhs,
#                   projector_setup=f.block_Jacobi_setup,
#                   projector_solve=f.block_Jacobi_prec_bicgstab_solve,
#                   initial_condition=phi0,
#                   method=esdirk,
#                   controller=pi)[1]
# phi = g.integrate(right_hand_side=f.rhs,
#                   initial_condition=phi0,
#                   method=AdaptiveERK54CashKarp(),
#                   controller=PIController(first_step=1.e-8, target_error=1.e-4))[1]
# np.save('t_sg_heptane_24.npy', data.t_list)
# np.save('q_sg_heptane_24.npy', data.solution_list)
# iq = 0
# nq = f._n_equations
# phi02d = phi0[iq::nq].reshape((ny, nx), order='F')
# phi2d = phi[iq::nq].reshape((ny, nx), order='F')
# rhs2d = f.rhs(0, phi)[iq::nq].reshape((ny, nx), order='F')
# phimphi02d = phi2d - phi02d
# c1 = f._state_1
# c2 = f._state_2
# c3 = f._state_3

# # second solve begin
#
# nx_1 = np.copy(nx)
# ny_1 = np.copy(nx)
# x_range1 = np.copy(x_range)
# y_range1 = np.copy(y_range)
# x_grid1, y_grid1 = np.meshgrid(x_range1, y_range1)
#
# nx = 42
# ny = nx
# # x_range = Flamelet._uniform_grid(nx)[0]
# # y_range = Flamelet._uniform_grid(ny)[0]
# x_range, y_range = make_clustered_grid(nx, ny, x_cp, y_cp, x_cc, y_cc)
# x_grid, y_grid = np.meshgrid(x_range, y_range)
#
# t_list1 = np.load('t_sg_heptane_24.npy')
# q_list1 = np.load('q_sg_heptane_24.npy')
#
# nt1 = t_list1.size
# ndof1 = q_list1.size // nt1
# nq = ndof1 // nx_1 // ny_1
#
# phi1d_2 = np.zeros(nq * nx * ny)
#
# for iq in range(nq):
#     if nt1 == 1:
#         phi2d_1 = q_list1[iq::nq].reshape((ny_1, nx_1), order='F')
#     else:
#         phi2d_1 = q_list1[-1, iq::nq].reshape((ny_1, nx_1), order='F')
#     spl = RectBivariateSpline(x_range1, y_range1, phi2d_1, kx=1, ky=1)
#     phi1d_2[iq::nq] = spl.ev(x_grid.ravel(), y_grid.ravel())
#
# f = Flamelet2D(m, phi1d_2, pressure, air, fuel1, fuel2, 10., 10., grid_1=x_range, grid_2=y_range)
#
# # air_hot = m.copy_stream(air)
# # air_hot.TP = 1400., pressure
# # air_cool = m.copy_stream(air)
# # air_cool.TP = 800., pressure
# # dmech4 = m.mix_fuels_for_stoich_mixture_fraction(m.stream('X', 'CH3OCH3:1, CH4:1'), n2, 0.2, air_hot)
# # dmech4.TP = 300., pressure
# # f = Flamelet2D(m, 'unreacted', pressure, dmech4, air_cool, air_hot, 1., 1., grid_1=x_range, grid_2=y_range)
#
# # ethy = m.stream('TPX', (300., pressure, 'C2H4:1'))
# # coh2 = m.stream('TPX', (300., pressure, 'CO:1, H2:0.01'))
# # f = Flamelet2D(m, 'equilibrium', pressure, air, ethy, coh2, 0.1, 0.1, grid_1=x_range, grid_2=y_range)
#
# phi0 = np.copy(f._initial_state)
#
# g = Governor()
# g.log_rate = 200
# g.termination_criteria = Steady(1.e-6)
# g.clip_to_positive = True
# g.norm_weighting = 1. / f._variable_scales
# g.projector_setup_rate = 20
# g.time_step_increase_factor_to_force_jacobian = 1.1
# g.time_step_decrease_factor_to_force_jacobian = 0.8
# data = SaveAllDataToList(initial_solution=phi0,
#                          save_frequency=100,
#                          file_prefix='ip',
#                          file_first_and_last_only=True,
#                          save_first_and_last_only=True)
# g.custom_post_process_step = data.save_data
# newton = SimpleNewtonSolver(evaluate_jacobian_every_iter=False,
#                             norm_weighting=g.norm_weighting,
#                             tolerance=1.e-10,
#                             max_nonlinear_iter=8)
# esdirk = ESDIRK64(norm_weighting=g.norm_weighting, nonlinear_solver=newton)
# # esdirk = BackwardEulerWithError(norm_weighting=g.norm_weighting, nonlinear_solver=newton)
# pi = PIController(first_step=1.e-6, target_error=1.e-8, max_step=1.e0)
# # phi = g.integrate(right_hand_side=f.rhs,
# #                   projector_setup=f.block_Jacobi_setup,
# #                   projector_solve=f.block_Jacobi_solve,
# #                   initial_condition=phi0,
# #                   method=esdirk,
# #                   controller=pi)[1]
# phi = g.integrate(right_hand_side=f.rhs,
#                   projector_setup=f.block_Jacobi_setup,
#                   projector_solve=f.block_Jacobi_prec_bicgstab_solve,
#                   initial_condition=phi0,
#                   method=esdirk,
#                   controller=pi)[1]
# # phi = g.integrate(right_hand_side=f.rhs,
# #                   initial_condition=phi0,
# #                   method=AdaptiveERK54CashKarp(),
# #                   controller=PIController(first_step=1.e-8, target_error=1.e-4))[1]
# np.save('t_sg_heptane_42.npy', data.t_list)
# np.save('q_sg_heptane_42.npy', data.solution_list)
# iq = 0
# nq = f._n_equations
# phi02d = phi0[iq::nq].reshape((ny, nx), order='F')
# phi2d = phi[iq::nq].reshape((ny, nx), order='F')
# rhs2d = f.rhs(0, phi)[iq::nq].reshape((ny, nx), order='F')
# phimphi02d = phi2d - phi02d
# c1 = f._state_1
# c2 = f._state_2
# c3 = f._state_3
#
# # second solve end
#
#
# # second solve begin
#
# nx_2 = np.copy(nx)
# ny_2 = np.copy(nx)
# x_range2 = np.copy(x_range)
# y_range2 = np.copy(y_range)
# x_grid2, y_grid2 = np.meshgrid(x_range2, y_range2)
#
# nx = 54
# ny = nx
# # x_range = Flamelet._uniform_grid(nx)[0]
# # y_range = Flamelet._uniform_grid(ny)[0]
# x_range, y_range = make_clustered_grid(nx, ny, x_cp, y_cp, x_cc, y_cc)
# x_grid, y_grid = np.meshgrid(x_range, y_range)
#
# t_list2 = np.load('t_sg_eg_36.npy')
# q_list2 = np.load('q_sg_eg_36.npy')
#
# nt2 = t_list2.size
# ndof2 = q_list2.size // nt2
# nq = ndof2 // nx_2 // ny_2
#
# phi1d_3 = np.zeros(nq * nx * ny)
#
# for iq in range(nq):
#     if nt2 == 1:
#         phi2d_2 = q_list2[iq::nq].reshape((ny_2, nx_2), order='F')
#     else:
#         phi2d_2 = q_list2[-1, iq::nq].reshape((ny_2, nx_2), order='F')
#     spl = RectBivariateSpline(x_range2, y_range2, phi2d_2, kx=1, ky=1)
#     phi1d_3[iq::nq] = spl.ev(x_grid.ravel(), y_grid.ravel())
#
# f = Flamelet2D(m, phi1d_3, pressure, air, fuel1, fuel2, 10., 10., grid_1=x_range, grid_2=y_range)
#
# # air_hot = m.copy_stream(air)
# # air_hot.TP = 1400., pressure
# # air_cool = m.copy_stream(air)
# # air_cool.TP = 800., pressure
# # dmech4 = m.mix_fuels_for_stoich_mixture_fraction(m.stream('X', 'CH3OCH3:1, CH4:1'), n2, 0.2, air_hot)
# # dmech4.TP = 300., pressure
# # f = Flamelet2D(m, 'unreacted', pressure, dmech4, air_cool, air_hot, 1., 1., grid_1=x_range, grid_2=y_range)
#
# # ethy = m.stream('TPX', (300., pressure, 'C2H4:1'))
# # coh2 = m.stream('TPX', (300., pressure, 'CO:1, H2:0.01'))
# # f = Flamelet2D(m, 'equilibrium', pressure, air, ethy, coh2, 0.1, 0.1, grid_1=x_range, grid_2=y_range)
#
# phi0 = np.copy(f._initial_state)
#
# g = Governor()
# g.log_rate = 200
# g.termination_criteria = Steady(1.e-6)
# g.clip_to_positive = True
# g.norm_weighting = 1. / f._variable_scales
# g.projector_setup_rate = 20
# g.time_step_increase_factor_to_force_jacobian = 1.1
# g.time_step_decrease_factor_to_force_jacobian = 0.8
# data = SaveAllDataToList(initial_solution=phi0,
#                          save_frequency=100,
#                          file_prefix='ip',
#                          file_first_and_last_only=True,
#                          save_first_and_last_only=True)
# g.custom_post_process_step = data.save_data
# newton = SimpleNewtonSolver(evaluate_jacobian_every_iter=False,
#                             norm_weighting=g.norm_weighting,
#                             tolerance=1.e-10,
#                             max_nonlinear_iter=8)
# esdirk = ESDIRK64(norm_weighting=g.norm_weighting, nonlinear_solver=newton)
# # esdirk = BackwardEulerWithError(norm_weighting=g.norm_weighting, nonlinear_solver=newton)
# pi = PIController(first_step=1.e-6, target_error=1.e-8, max_step=1.e0)
# # phi = g.integrate(right_hand_side=f.rhs,
# #                   projector_setup=f.block_Jacobi_setup,
# #                   projector_solve=f.block_Jacobi_solve,
# #                   initial_condition=phi0,
# #                   method=esdirk,
# #                   controller=pi)[1]
# phi = g.integrate(right_hand_side=f.rhs,
#                   projector_setup=f.block_Jacobi_setup,
#                   projector_solve=f.block_Jacobi_prec_bicgstab_solve,
#                   initial_condition=phi0,
#                   method=esdirk,
#                   controller=pi)[1]
# # phi = g.integrate(right_hand_side=f.rhs,
# #                   initial_condition=phi0,
# #                   method=AdaptiveERK54CashKarp(),
# #                   controller=PIController(first_step=1.e-8, target_error=1.e-4))[1]
# np.save('t_sg_eg_54.npy', data.t_list)
# np.save('q_sg_eg_54.npy', data.solution_list)
# iq = 0
# nq = f._n_equations
# phi02d = phi0[iq::nq].reshape((ny, nx), order='F')
# phi2d = phi[iq::nq].reshape((ny, nx), order='F')
# rhs2d = f.rhs(0, phi)[iq::nq].reshape((ny, nx), order='F')
# phimphi02d = phi2d - phi02d
# c1 = f._state_1
# c2 = f._state_2
# c3 = f._state_3
#
# # second solve end


# second solve begin

nx_3 = np.copy(nx)
ny_3 = np.copy(nx)
x_range3 = np.copy(x_range)
y_range3 = np.copy(y_range)
x_grid3, y_grid3 = np.meshgrid(x_range3, y_range3)

nx = 64
ny = nx
# x_range = Flamelet._uniform_grid(nx)[0]
# y_range = Flamelet._uniform_grid(ny)[0]
x_range, y_range = make_clustered_grid(nx, ny, x_cp, y_cp, x_cc, y_cc)
x_grid, y_grid = np.meshgrid(x_range, y_range)

t_list3 = np.load('t_sg_heptane_24.npy')
q_list3 = np.load('q_sg_heptane_24.npy')

nt3 = t_list3.size
ndof3 = q_list3.size // nt3
nq = ndof3 // nx_3 // ny_3

phi1d_4 = np.zeros(nq * nx * ny)

for iq in range(nq):
    if nt3 == 1:
        phi2d_3 = q_list3[iq::nq].reshape((ny_3, nx_3), order='F')
    else:
        phi2d_3 = q_list3[-1, iq::nq].reshape((ny_3, nx_3), order='F')
    spl = RectBivariateSpline(x_range3, y_range3, phi2d_3, kx=1, ky=1)
    phi1d_4[iq::nq] = spl.ev(x_grid.ravel(), y_grid.ravel())

f = _Flamelet2D(m, phi1d_4, pressure, air, fuel1, fuel2, 10., 10., grid_1=x_range, grid_2=y_range)

# air_hot = m.copy_stream(air)
# air_hot.TP = 1400., pressure
# air_cool = m.copy_stream(air)
# air_cool.TP = 800., pressure
# dmech4 = m.mix_fuels_for_stoich_mixture_fraction(m.stream('X', 'CH3OCH3:1, CH4:1'), n2, 0.2, air_hot)
# dmech4.TP = 300., pressure
# f = Flamelet2D(m, 'unreacted', pressure, dmech4, air_cool, air_hot, 1., 1., grid_1=x_range, grid_2=y_range)

# ethy = m.stream('TPX', (300., pressure, 'C2H4:1'))
# coh2 = m.stream('TPX', (300., pressure, 'CO:1, H2:0.01'))
# f = Flamelet2D(m, 'equilibrium', pressure, air, ethy, coh2, 0.1, 0.1, grid_1=x_range, grid_2=y_range)

phi0 = np.copy(f._initial_state)

g = Governor()
g.log_rate = 25
g.termination_criteria = Steady(1.e-6)
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
                            tolerance=1.e-10,
                            max_nonlinear_iter=8)
esdirk = ESDIRK64(norm_weighting=g.norm_weighting, nonlinear_solver=newton)
# esdirk = BackwardEulerWithError(norm_weighting=g.norm_weighting, nonlinear_solver=newton)
pi = PIController(first_step=1.e-6, target_error=1.e-8, max_step=1.e0)
# phi = g.integrate(right_hand_side=f.rhs,
#                   projector_setup=f.block_Jacobi_setup,
#                   projector_solve=f.block_Jacobi_solve,
#                   initial_condition=phi0,
#                   method=esdirk,
#                   controller=pi)[1]
phi = g.integrate(right_hand_side=f.rhs,
                  projector_setup=f.block_Jacobi_setup,
                  projector_solve=f.block_Jacobi_prec_bicgstab_solve,
                  initial_condition=phi0,
                  method=esdirk,
                  controller=pi)[1]
# phi = g.integrate(right_hand_side=f.rhs,
#                   initial_condition=phi0,
#                   method=AdaptiveERK54CashKarp(),
#                   controller=PIController(first_step=1.e-8, target_error=1.e-4))[1]
np.save('t_sg_heptane_64.npy', data.t_list)
np.save('q_sg_heptane_64.npy', data.solution_list)
iq = 0
nq = f._n_equations
phi02d = phi0[iq::nq].reshape((ny, nx), order='F')
phi2d = phi[iq::nq].reshape((ny, nx), order='F')
rhs2d = f.rhs(0, phi)[iq::nq].reshape((ny, nx), order='F')
phimphi02d = phi2d - phi02d
c1 = f._state_1
c2 = f._state_2
c3 = f._state_3

# second solve end

ax = plt.subplot2grid((2, 3), (0, 2))
ax.plot(x_range, phi02d[0, :])
ax.plot(x_range, phi2d[0, :])
ax.yaxis.tick_right()
ax.set_xlabel('$Z_1$')
ax.set_xlim([0, 1])
ax.grid(True)
ax = plt.subplot2grid((2, 3), (1, 2))
ax.plot(y_range, phi02d[:, 0])
ax.plot(y_range, phi2d[:, 0])
ax.yaxis.tick_right()
ax.set_xlabel('$Z_2$')
ax.set_xlim([0, 1])
ax.grid(True)
ax = plt.subplot2grid((2, 3), (0, 0), rowspan=2, colspan=2)
ax.contourf(x_grid, y_grid, phi2d, cmap=plt.get_cmap('rainbow'))
t1 = plt.Polygon(np.array([[1, 0], [1, 1], [0, 1]]), color='w')
ax.add_patch(t1)
ax.text(0.03, 1.00, 'C2H4', fontdict={'fontweight': 'bold'},
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
ax.text(0.98, 0.04, 'COH2', fontdict={'fontweight': 'bold'}, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
ax.text(-0.08, 0.05, 'air', fontdict={'fontweight': 'bold'},
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
ax.set_xlabel('$Z_1$')
ax.set_ylabel('$Z_2$', rotation=0)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.grid(True)
plt.tight_layout()
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x_grid, y_grid, phi2d, cmap=plt.get_cmap('rainbow'))
# ax.plot_wireframe(x_grid, y_grid, phimphi02d, linewidth=1)
ax.set_xlabel('z1')
ax.set_ylabel('z2')
ax.set_zlabel('f')
# ax.plot([0], [0], [c1[iq]], 'ko')
# ax.plot([0], [1], [c2[iq]], 'ko')
# ax.plot([1], [0], [c3[iq]], 'ko')
# ax.plot([1], [1], [c1[iq]], 'ko')
plt.show()

# ax = plt.subplot2grid((2, 3), (0, 2))
# ax.plot(x_range, rhs2d[0, :])
# ax.yaxis.tick_right()
# ax.set_xlabel('$Z_1$')
# ax.set_xlim([0, 1])
# ax.grid(True)
# ax = plt.subplot2grid((2, 3), (1, 2))
# ax.plot(y_range, rhs2d[:, 0])
# ax.yaxis.tick_right()
# ax.set_xlabel('$Z_2$')
# ax.set_xlim([0, 1])
# ax.grid(True)
# ax = plt.subplot2grid((2, 3), (0, 0), rowspan=2, colspan=2)
# ax.contourf(x_grid, y_grid, rhs2d, cmap=plt.get_cmap('rainbow'))
# t1 = plt.Polygon(np.array([[1, 0], [1, 1], [0, 1]]), color='w')
# ax.add_patch(t1)
# ax.text(0.03, 1.00, 'sg', fontdict={'fontweight': 'bold'}, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
# ax.text(0.98, 0.04, 'h2', fontdict={'fontweight': 'bold'}, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
# ax.text(-0.08, 0.05, 'air', fontdict={'fontweight': 'bold'}, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
# ax.set_xlabel('$Z_1$')
# ax.set_ylabel('$Z_2$', rotation=0)
# ax.set_xlim([0, 1])
# ax.set_ylim([0, 1])
# ax.grid(True)
# plt.tight_layout()
# plt.show()
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(x_grid, y_grid, rhs2d, cmap=plt.get_cmap('rainbow'))
# # ax.plot_wireframe(x_grid, y_grid, rhs2d, linewidth=1)
# ax.set_xlabel('z1')
# ax.set_ylabel('z2')
# ax.set_zlabel('f')
# # ax.plot([0], [0], [c1[iq]], 'ko')
# # ax.plot([0], [1], [c2[iq]], 'ko')
# # ax.plot([1], [0], [c3[iq]], 'ko')
# # ax.plot([1], [1], [c1[iq]], 'ko')
# plt.show()
