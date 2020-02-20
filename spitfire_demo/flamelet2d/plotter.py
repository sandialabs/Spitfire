import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
import matplotlib.animation as manimation
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d as a3
from spitfire.chemistry.mechanism import ChemicalMechanismSpec
from spitfire.chemistry.flamelet2d import Flamelet2D

FFMpegWriter = manimation.writers['ffmpeg']
metadata = dict(title='Movie Test', artist='Matplotlib',
                comment='Movie support!')
writer = FFMpegWriter(fps=15, metadata=metadata)

nx = 24
ny = nx
x_range = np.linspace(0., 1., nx)
y_range = np.linspace(0., 1., ny)
x_grid, y_grid = np.meshgrid(x_range, y_range)

t_list = np.load('ip_times.npy')
solution_list = np.load('ip_solutions.npy')
# t_list = np.load('t_sg_eg_54.npy')
# solution_list = np.load('q_sg_eg_54.npy')

Tmin = 200.
Tmax = int(np.max(solution_list) // 100 + 1) * 100

nt = t_list.size

ndof = solution_list.size // nt
nq = ndof // nx // ny

# m = ChemicalMechanismSpec(cantera_xml='ethylene-luo.xml', group_name='ethylene-luo')
# m = ChemicalMechanismSpec(cantera_xml='coh2-hawkes.xml', group_name='coh2-hawkes')
m = ChemicalMechanismSpec(cantera_xml='heptane-liu-hewson-chen-pitsch-highT.xml', group_name='gas')

variable = 'T'

pressure = 101325.
air = m.stream(stp_air=True)
air.TP = 300., pressure
# sg = m.stream('X', 'H2:1, CO:2')
sg = m.stream('X', 'H2:1, CO:1, CH4:1, CO2:1')
fuel2 = m.copy_stream(sg)
# fuel2.TP = 300., pressure
fuel2.TP = 400., pressure
# c2h4 = m.stream('X', 'C2H4:1')
eg = m.stream('X', 'CO2:1, H2O:1, CO:0.5, H2:0.001')
c7 = m.stream('X', 'NXC7H16:1')
# fuel1 = m.copy_stream(eg)
# fuel1.TP = 300., pressure
fuel1 = m.copy_stream(c7)
fuel1.TP = 485., pressure

fuel1_name = 'C7'
fuel2_name = 'SG'

f = Flamelet2D(m, 'equilibrium', pressure, air, fuel1, fuel2, 10., 10., grid_1=x_range, grid_2=y_range)

iq = 0 if variable == 'T' else m.species_index(variable) + 1
# if nt == 1:
#     phi02d = solution_list[iq::nq].reshape((ny, nx), order='F')
# else:
#     phi02d = solution_list[0, iq::nq].reshape((ny, nx), order='F')

phi0 = np.copy(f._initial_state)[iq::nq]
phi02d = phi0.reshape((ny, nx), order='F')


def plot_contours(it):
    fig = plt.figure()
    # plt.set_current_figure(fig)
    if nt == 1:
        phi2d = solution_list[iq::nq].reshape((ny, nx), order='F')
    else:
        phi2d = solution_list[it, iq::nq].reshape((ny, nx), order='F')
    ax = plt.subplot2grid((3, 4), (2, 1), rowspan=1, colspan=2)
    ax.cla()
    ax.plot(x_range, phi02d[0, :], 'b--', label='EQ')
    ax.plot(x_range, phi2d[0, :], 'g-', label='SLFM')
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

    fig5 = plt.figure()
    # plt.set_current_figure(fig)
    ax = plt.subplot2grid((3, 4), (2, 1), rowspan=1, colspan=2)
    ax.cla()
    ax.plot(x_range, phi02d[0, :])
    ax.plot(x_range, phi2d[0, :])
    if variable == 'T':
        ax.set_ylim([Tmin, Tmax])
    ax.yaxis.tick_right()
    ax.set_xlabel('$Z_1$')
    ax.set_xlim([0, 1])
    ax.grid(True)
    ax = plt.subplot2grid((3, 4), (0, 0), rowspan=2, colspan=1)
    ax.cla()
    ax.plot(phi02d[:, 0], y_range)
    ax.plot(phi2d[:, 0], y_range)
    if variable == 'T':
        ax.set_xlim([Tmin, Tmax])
    ax.set_ylabel('$Z_2$')
    ax.set_ylim([0, 1])
    ax.grid(True)
    ax = plt.subplot2grid((3, 4), (0, 1), rowspan=2, colspan=2)
    cax = plt.subplot2grid((3, 4), (0, 3), rowspan=2, colspan=1)
    ax.cla()
    if variable == 'T':
        contour = ax.contourf(x_grid, y_grid, phi02d, cmap=plt.get_cmap('magma'),
                              norm=Normalize(Tmin, Tmax), levels=np.linspace(Tmin, Tmax, 20))
    else:
        contour = ax.contourf(x_grid, y_grid, phi02d, cmap=plt.get_cmap('magma'),
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
    cax.set_title('EQ T' if variable == 'T' else 'EQ ' + variable)
    plt.tight_layout()

    fig4 = plt.figure()
    # plt.set_current_figure(fig4)
    if nt == 1:
        phi2d = solution_list[iq::nq].reshape((ny, nx), order='F')
    else:
        phi2d = solution_list[it, iq::nq].reshape((ny, nx), order='F')
    ax = plt.subplot2grid((3, 4), (2, 1), rowspan=1, colspan=2)
    ax.cla()
    ax.plot(x_range, phi2d[0, :] - phi02d[0, :])
    ax.yaxis.tick_right()
    ax.set_xlabel('$Z_1$')
    ax.set_xlim([0, 1])
    ax.grid(True)
    ax = plt.subplot2grid((3, 4), (0, 0), rowspan=2, colspan=1)
    ax.cla()
    ax.plot(phi2d[:, 0] - phi02d[:, 0], y_range)
    ax.set_ylabel('$Z_2$')
    ax.set_ylim([0, 1])
    ax.grid(True)
    ax = plt.subplot2grid((3, 4), (0, 1), rowspan=2, colspan=2)
    cax = plt.subplot2grid((3, 4), (0, 3), rowspan=2, colspan=1)
    ax.cla()
    relerr = (phi02d - phi2d) / (np.max([np.max(phi02d), np.max(phi2d)]) + 1.e-8)
    contour = ax.contourf(x_grid, y_grid, relerr,
                          cmap=plt.get_cmap('bwr'),
                          levels=np.linspace(-np.max(np.abs(relerr)), np.max(np.abs(relerr)), 40))
    plt.colorbar(contour, cax=cax)
    ax.contour(x_grid, y_grid, relerr, levels=[0.], colors=['k'], linewidths=[0.5])
    ax.plot([0, 0, 1, 0], [0, 1, 0, 0], 'k-', linewidth=0.5, zorder=4)
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
    cax.set_title('relative error\nEQ-SLFM T' if variable == 'T' else 'relative error\nEQ-SLFM ' + variable)
    plt.tight_layout()


def plot_3d_scatter():
    fig1 = plt.figure()
    ax = fig1.add_subplot(111, projection='3d')
    xf, yf = x_grid.ravel(order='F'), y_grid.ravel(order='F')
    phi = solution_list[-1, iq::nq]
    zztri = xf + yf <= 1.
    ax.scatter(xf[zztri], yf[zztri], phi[zztri], c=phi[zztri], s=8, cmap=plt.get_cmap('rainbow'))
    ax.set_xlabel('$Z_1$')
    ax.set_ylabel('$Z_2$')
    ax.set_zlabel('T' if variable == 'T' else variable)
    ax.set_title('current SLFM')

    fig2 = plt.figure()
    ax = fig2.add_subplot(111, projection='3d')
    xf, yf = x_grid.ravel(order='F'), y_grid.ravel(order='F')
    phi = phi0
    zztri = xf + yf <= 1.
    ax.scatter(xf[zztri], yf[zztri], phi[zztri], c=phi[zztri], s=8, cmap=plt.get_cmap('rainbow'))
    ax.set_xlabel('$Z_1$')
    ax.set_ylabel('$Z_2$')
    ax.set_zlabel('T' if variable == 'T' else variable)
    ax.set_title('equilibrium')

    fig3 = plt.figure()
    ax = fig3.add_subplot(111, projection='3d')
    xf, yf = x_grid.ravel(order='F'), y_grid.ravel(order='F')
    phi = solution_list[-1, iq::nq] - phi0
    zztri = xf + yf <= 1.
    ax.scatter(xf[zztri], yf[zztri], phi[zztri], c=phi[zztri], s=8, cmap=plt.get_cmap('rainbow'))
    ax.set_xlabel('$Z_1$')
    ax.set_ylabel('$Z_2$')
    ax.set_zlabel('T' if variable == 'T' else variable)
    ax.set_title('SLFM - EQ')


plot_contours(-1)
plt.show()

plot_3d_scatter()
plt.show()

# for i in range(0, nt - 1, 1):
#     newplot(i)
#     plt.title(str(i + 1) + ' / ' + str(nt))
#     plt.show()

# with writer.saving(fig, 'movie.mp4', nt):
#     for i in range(0, nt, 7):
#         print(i + 1, '/', nt)
#         newplot(i)
#         writer.grab_frame()
