try:
    import unittest
    from copy import copy

    from numpy.testing import assert_allclose
    import numpy as np

    from spitfire.chemistry.mechanism import ChemicalMechanismSpec
    from spitfire.chemistry.library import Library, Dimension
    from spitfire.chemistry.flamelet import FlameletSpec
    from spitfire.chemistry.tabulation import build_adiabatic_eq_library, apply_presumed_PDF_model

    import cantera
    import cantera as ct

    import pytabprops

    if int(cantera.__version__.replace('.', '')) >= 250:
        class Test(unittest.TestCase):
            def test(self):
                gas = ct.Solution('h2o2.yaml', transport_model='Multi')
                mech = ChemicalMechanismSpec.from_solution(gas)

                fs = FlameletSpec(mech_spec=mech,
                                  initial_condition='equilibrium',
                                  oxy_stream=mech.stream('TPX', (300, 1.e5, 'O2:1, N2:3.76')),
                                  fuel_stream=mech.stream('TPY', (300, 1.e5, 'H2:1')),
                                  grid_points=16)
                eq_lib1 = build_adiabatic_eq_library(fs, verbose=False)

                z_dim = Dimension(eq_lib1.mixture_fraction_name, eq_lib1.mixture_fraction_values)
                fuel_T_dim = Dimension('fuel_temperature', np.linspace(0.0, 1.0, 4))
                air_T_dim = Dimension('air_temperature', np.linspace(0.0, 1.0, 3))

                eq_lib2 = Library(z_dim, fuel_T_dim)
                eq_lib2T = Library(fuel_T_dim, z_dim)
                eq_lib3 = Library(z_dim, fuel_T_dim, air_T_dim)
                eq_lib3T1 = Library(fuel_T_dim, z_dim, air_T_dim)
                eq_lib3T2 = Library(fuel_T_dim, air_T_dim, z_dim)

                for p in eq_lib1.props:
                    eq_lib2[p] = eq_lib2.get_empty_dataset()
                    eq_lib2T[p] = eq_lib2T.get_empty_dataset()
                    eq_lib3[p] = eq_lib3.get_empty_dataset()
                    eq_lib3T1[p] = eq_lib3T1.get_empty_dataset()
                    eq_lib3T2[p] = eq_lib3T2.get_empty_dataset()

                for i, fuel_T_offset in enumerate(fuel_T_dim.values):
                    fuel_T = 300 + fuel_T_offset * 500.
                    fs2 = copy(fs)
                    fs2.fuel_stream.TP = fuel_T, 1.e5

                    eq_tmp = build_adiabatic_eq_library(fs2, verbose=False)
                    for p in eq_lib1.props:
                        eq_lib2[p][:, i] = eq_tmp[p]
                        eq_lib2T[p][i, :] = eq_tmp[p]

                    for j, air_T_offset in enumerate(air_T_dim.values):
                        air_T = 300 + air_T_offset * 500.
                        fs3 = copy(fs2)
                        fs3.oxy_stream.TP = air_T, 1.e5

                        eq_tmp = build_adiabatic_eq_library(fs3, verbose=False)
                        for p in eq_lib1.props:
                            eq_lib3[p][:, i, j] = eq_tmp[p]
                            eq_lib3T1[p][i, :, j] = eq_tmp[p]
                            eq_lib3T2[p][i, j, :] = eq_tmp[p]

                nonT_props = list(eq_lib1.props)
                nonT_props.remove('temperature')
                eq_lib1.remove(*nonT_props)
                eq_lib2.remove(*nonT_props)
                eq_lib2T.remove(*nonT_props)
                eq_lib3.remove(*nonT_props)
                eq_lib3T1.remove(*nonT_props)
                eq_lib3T2.remove(*nonT_props)

                z_svv = np.linspace(0., 1., 6)
                Tf_svv = np.linspace(0., 1., 5)

                eq_lib1_t = apply_presumed_PDF_model(eq_lib1, 'ClipGauss', z_svv, verbose=False)
                eq_lib2_t = apply_presumed_PDF_model(eq_lib2, 'ClipGauss', z_svv, verbose=False)
                eq_lib3_t = apply_presumed_PDF_model(eq_lib3, 'ClipGauss', z_svv, num_procs=1, verbose=False)
                eq_lib2T_t = apply_presumed_PDF_model(eq_lib2T, 'ClipGauss', z_svv, verbose=False)
                eq_lib3T1_t = apply_presumed_PDF_model(eq_lib3T1, 'ClipGauss', z_svv, num_procs=1, verbose=False)
                eq_lib3T2_t = apply_presumed_PDF_model(eq_lib3T2, 'ClipGauss', z_svv, num_procs=1, verbose=False)
                eq_lib2_tt = apply_presumed_PDF_model(eq_lib2_t, 'Beta', Tf_svv,
                                                      'fuel_temperature_mean', '', 'Tfvar', num_procs=1, verbose=False)
                eq_lib3_tt = apply_presumed_PDF_model(eq_lib3_t, 'Beta', Tf_svv,
                                                      'fuel_temperature_mean', '', 'Tfvar', num_procs=1, verbose=False)

                def get_dim_names(lib):
                    return [d.name for d in lib.dims]

                self.assertEqual(['mixture_fraction'], get_dim_names(eq_lib1))
                self.assertEqual(['mixture_fraction_mean', 'scaled_scalar_variance_mean'], get_dim_names(eq_lib1_t))
                self.assertEqual(['mixture_fraction', 'fuel_temperature'], get_dim_names(eq_lib2))
                self.assertEqual(['mixture_fraction_mean', 'fuel_temperature_mean', 'scaled_scalar_variance_mean'],
                                 get_dim_names(eq_lib2_t))
                self.assertEqual(
                    ['mixture_fraction_mean', 'fuel_temperature_mean', 'scaled_scalar_variance_mean', 'Tfvar'],
                    get_dim_names(eq_lib2_tt))
                self.assertEqual(['mixture_fraction', 'fuel_temperature', 'air_temperature'],
                                 get_dim_names(eq_lib3))
                self.assertEqual(['mixture_fraction_mean', 'fuel_temperature_mean', 'air_temperature_mean',
                                  'scaled_scalar_variance_mean'],
                                 get_dim_names(eq_lib3_t))
                self.assertEqual(['mixture_fraction_mean', 'fuel_temperature_mean', 'air_temperature_mean',
                                  'scaled_scalar_variance_mean', 'Tfvar'],
                                 get_dim_names(eq_lib3_tt))

                self.assertEqual(['fuel_temperature', 'mixture_fraction'], get_dim_names(eq_lib2T))
                self.assertEqual(['fuel_temperature_mean', 'mixture_fraction_mean', 'scaled_scalar_variance_mean'],
                                 get_dim_names(eq_lib2T_t), eq_lib2T_t)

                self.assertEqual(['fuel_temperature', 'mixture_fraction', 'air_temperature'],
                                 get_dim_names(eq_lib3T1))
                self.assertEqual(['fuel_temperature', 'air_temperature', 'mixture_fraction'],
                                 get_dim_names(eq_lib3T2))
                self.assertEqual(['fuel_temperature_mean', 'mixture_fraction_mean', 'air_temperature_mean',
                                  'scaled_scalar_variance_mean'],
                                 get_dim_names(eq_lib3T1_t))
                self.assertEqual(['fuel_temperature_mean', 'air_temperature_mean', 'mixture_fraction_mean',
                                  'scaled_scalar_variance_mean'],
                                 get_dim_names(eq_lib3T2_t))

                self.assertFalse(np.any(np.isnan(eq_lib1['temperature'])))
                self.assertFalse(np.any(np.isnan(eq_lib1_t['temperature'])))
                self.assertFalse(np.any(np.isnan(eq_lib2['temperature'])))
                self.assertFalse(np.any(np.isnan(eq_lib2T['temperature'])))
                self.assertFalse(np.any(np.isnan(eq_lib2_t['temperature'])))
                self.assertFalse(np.any(np.isnan(eq_lib2T_t['temperature'])))
                self.assertFalse(np.any(np.isnan(eq_lib3['temperature'])))
                self.assertFalse(np.any(np.isnan(eq_lib3T1['temperature'])))
                self.assertFalse(np.any(np.isnan(eq_lib3T2['temperature'])))
                self.assertFalse(np.any(np.isnan(eq_lib3_t['temperature'])))
                self.assertFalse(np.any(np.isnan(eq_lib3_tt['temperature'])))
                self.assertFalse(np.any(np.isnan(eq_lib3T1_t['temperature'])))
                self.assertFalse(np.any(np.isnan(eq_lib3T2_t['temperature'])))

                self.assertIsNone(assert_allclose(eq_lib2T['temperature'].T, eq_lib2['temperature']))
                self.assertIsNone(assert_allclose(np.swapaxes(eq_lib3T1['temperature'], 0, 1),
                                                  eq_lib3['temperature']))
                self.assertIsNone(assert_allclose(np.swapaxes(np.swapaxes(eq_lib3T2['temperature'], 1, 2), 0, 1),
                                                  eq_lib3['temperature']))

                self.assertIsNone(assert_allclose(np.squeeze(eq_lib1_t['temperature'][:, 0]),
                                                  eq_lib1['temperature']))
                self.assertIsNone(assert_allclose(np.squeeze(eq_lib2_t['temperature'][:, :, 0]),
                                                  eq_lib2['temperature']))
                self.assertIsNone(assert_allclose(np.squeeze(eq_lib3_t['temperature'][:, :, :, 0]),
                                                  eq_lib3['temperature']))
                self.assertIsNone(assert_allclose(np.squeeze(eq_lib3_tt['temperature'][:, :, :, 0, 0]),
                                                  eq_lib3['temperature']))


        if __name__ == '__main__':
            unittest.main()

except ImportError:
    pass
