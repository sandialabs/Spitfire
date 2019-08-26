from spitfire.chemistry.tabulation import Library
import matplotlib.pyplot as plt
import numpy as np

library = Library.load_from_file('coal_library.pkl')

z_dim = library.dim('mixture_fraction')
x_dim = library.dim('dissipation_rate_stoich')
h_dim = library.dim('heat_transfer_coefficient')
a_dim = library.dim('alpha')

for ia, a in enumerate(a_dim.values):
    plt.plot(z_dim.values, library['temperature'][ia, 0, 0, :], label='$\\' + a_dim.name + '$' + f'={a:.2f}')
plt.legend(loc='best')
plt.xlabel(z_dim.name)
plt.ylabel('temperature')
plt.show()

T_max = np.max(library['temperature'])
C2H2_max = np.max(library['mass fraction C2H2'])
h_min = np.min(library['enthalpy'])
h_max = np.max(library['enthalpy'])

key1, key2, key3 = 'temperature', 'mass fraction C2H2', 'enthalpy'
alpha_indices = list(range(a_dim.npts))

ih = 0

fig, axarray = plt.subplots(len(alpha_indices), 3, sharey=True, sharex=True)
axarray[0, 0].set_title(key1)
axarray[0, 1].set_title(key2)
axarray[0, 2].set_title(key3)
for i, ig in enumerate(alpha_indices):
    axarray[i, 2].text(1.1, 0.5, '$\\alpha$' + f'={a_dim.values[ig]:.2f}',
                       horizontalalignment='left',
                       verticalalignment='center',
                       transform=axarray[i, 2].transAxes)
    axarray[i, 0].contourf(z_dim.grid[alpha_indices[i], ih, :, :],
                           x_dim.grid[alpha_indices[i], ih, :, :],
                           library[key1][alpha_indices[i], ih, :, :],
                           levels=np.linspace(300., T_max, 20))
    axarray[i, 1].contourf(z_dim.grid[alpha_indices[i], ih, :, :],
                           x_dim.grid[alpha_indices[i], ih, :, :],
                           library[key2][alpha_indices[i], ih, :, :],
                           levels=np.linspace(0, C2H2_max, 20))
    axarray[i, 2].contourf(z_dim.grid[alpha_indices[i], ih, :, :],
                           x_dim.grid[alpha_indices[i], ih, :, :],
                           library[key3][alpha_indices[i], ih, :, :],
                           levels=np.linspace(h_min, h_max, 20))
    axarray[i, 2].set_yscale('log')
    if i < len(alpha_indices) - 1:
        for ax in axarray[i, :]:
            ax.set_xticks([])
    else:
        for ax in axarray[i, :]:
            ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
            ax.set_xlabel('$\\mathcal{Z}$')
    for ax in axarray[i, :]:
        ax.grid(True)
    axarray[i, 0].set_ylabel('$\\chi_{st}$')
plt.show()
