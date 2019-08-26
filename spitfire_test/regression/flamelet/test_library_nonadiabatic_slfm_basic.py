import unittest


class Test(unittest.TestCase):
    def test(self):
        from spitfire.chemistry.tabulation import build_nonadiabatic_defect_slfm_library
        from spitfire.chemistry.mechanism import ChemicalMechanismSpec
        import matplotlib.pyplot as plt
        import numpy as np
        from os.path import abspath, join

        xml = abspath(join('spitfire_test', 'test_mechanisms', 'h2-burke.xml'))
        mech = ChemicalMechanismSpec(cantera_xml=xml, group_name='h2-burke')
        pressure = 101325.
        fuel = mech.stream('TPX', (300., pressure, 'H2:1, N2:1'))
        air = mech.stream(stp_air=True)
        air.TP = 1400., pressure

        flamelet_specs = {'mech_spec': mech,
                          'pressure': pressure,
                          'oxy_stream': air,
                          'fuel_stream': fuel,
                          'grid_points': 66,
                          'grid_cluster_intensity': 6.}

        quantities = ['temperature', 'mass fraction OH', 'enthalpy']

        lib = build_nonadiabatic_defect_slfm_library(flamelet_specs, quantities,
                                                     diss_rate_values=np.logspace(-3, 2, 8),
                                                     diss_rate_ref='stoichiometric',
                                                     verbose=False, n_defect_st=32, num_procs=4)

        z_dim = lib.dim('mixture_fraction')
        x_dim = lib.dim('dissipation_rate_stoich')
        g_dim = lib.dim('enthalpy_defect_stoich')

        T_max = np.max(lib['temperature'])
        OH_max = np.max(lib['mass fraction OH'])
        h_min = np.min(lib['enthalpy'])
        h_max = np.max(lib['enthalpy'])

        key1, key2, key3 = 'temperature', 'mass fraction OH', 'enthalpy'
        gamma_indices = [0, 7, 15, 23, 31]

        fig, axarray = plt.subplots(len(gamma_indices), 3, sharey=True, sharex=True)
        axarray[0, 0].set_title(key1)
        axarray[0, 1].set_title(key2)
        axarray[0, 2].set_title(key3)
        for i, ig in enumerate(gamma_indices):
            axarray[i, 2].text(1.1, 0.5, '$\\gamma_{st}$' + f'={g_dim.values[ig] / 1.e6:.1f}\nMJ/kg',
                               horizontalalignment='left',
                               verticalalignment='center',
                               transform=axarray[i, 2].transAxes)
            axarray[i, 0].contourf(z_dim.grid[:, :, gamma_indices[i]],
                                   x_dim.grid[:, :, gamma_indices[i]],
                                   lib[key1][:, :, gamma_indices[i]],
                                   levels=np.linspace(300., T_max, 20))
            axarray[i, 1].contourf(z_dim.grid[:, :, gamma_indices[i]],
                                   x_dim.grid[:, :, gamma_indices[i]],
                                   lib[key2][:, :, gamma_indices[i]],
                                   levels=np.linspace(0, OH_max, 20))
            axarray[i, 2].contourf(z_dim.grid[:, :, gamma_indices[i]],
                                   x_dim.grid[:, :, gamma_indices[i]],
                                   lib[key3][:, :, gamma_indices[i]],
                                   levels=np.linspace(h_min, h_max, 20))
            axarray[i, 2].set_yscale('log')
            if i < len(gamma_indices) - 1:
                for ax in axarray[i, :]:
                    ax.set_xticks([])
            else:
                for ax in axarray[i, :]:
                    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
                    ax.set_xlabel('$\\mathcal{Z}$')
            for ax in axarray[i, :]:
                ax.grid(True)
            axarray[i, 0].set_ylabel('$\\chi_{st}$')


if __name__ == '__main__':
    unittest.main()
