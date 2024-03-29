try:
    import unittest
    from copy import copy

    from numpy.testing import assert_allclose
    import numpy as np

    from spitfire.chemistry.mechanism import ChemicalMechanismSpec
    from spitfire.chemistry.library import Library, Dimension
    from spitfire.chemistry.flamelet import FlameletSpec
    from spitfire.chemistry.tabulation import build_adiabatic_eq_library, apply_mixing_model, PDFSpec, build_adiabatic_slfm_library, build_nonadiabatic_defect_transient_slfm_library

    import cantera
    import cantera as ct
    from spitfire.chemistry.ctversion import check as cantera_version_check


    if cantera_version_check('atleast', 2, 5, None):
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

                eq_lib1_t = apply_mixing_model(eq_lib1, {'mixture_fraction': PDFSpec('ClipGauss', z_svv)}, verbose=False)
                eq_lib2_t = apply_mixing_model(eq_lib2, {'mixture_fraction': PDFSpec('ClipGauss', z_svv)}, verbose=False)
                eq_lib3_t = apply_mixing_model(eq_lib3, {'mixture_fraction': PDFSpec('ClipGauss', z_svv)}, num_procs=1, verbose=False)
                eq_lib2T_t = apply_mixing_model(eq_lib2T, {'mixture_fraction': PDFSpec('ClipGauss', z_svv)}, verbose=False)
                eq_lib3T1_t = apply_mixing_model(eq_lib3T1, {'mixture_fraction': PDFSpec('ClipGauss', z_svv)}, num_procs=1, verbose=False)
                eq_lib3T2_t = apply_mixing_model(eq_lib3T2, {'mixture_fraction': PDFSpec('ClipGauss', z_svv)}, num_procs=1, verbose=False)
                eq_lib2_tt = apply_mixing_model(eq_lib2_t, {'fuel_temperature_mean': PDFSpec('Beta', Tf_svv, variance_name='Tfvar')}, added_suffix='', num_procs=1, verbose=False)
                eq_lib3_tt = apply_mixing_model(eq_lib3_t, {'fuel_temperature_mean': PDFSpec('Beta', Tf_svv, variance_name='Tfvar')}, added_suffix='', num_procs=1, verbose=False)

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

                # parallel options
                slfm = build_adiabatic_slfm_library(fs, diss_rate_values=np.logspace(-1,1,3), verbose=False)
                slfm_serial = apply_mixing_model(slfm, {'mixture_fraction': PDFSpec('DoubleDelta', z_svv)}, num_procs=1, verbose=False)
                slfm_prop   = apply_mixing_model(slfm, {'mixture_fraction': PDFSpec('DoubleDelta', z_svv, parallel_type='property')}, num_procs=2, verbose=False)
                slfm_mean   = apply_mixing_model(slfm, {'mixture_fraction': PDFSpec('DoubleDelta', z_svv, parallel_type='property-mean')}, num_procs=2, verbose=False)
                slfm_var    = apply_mixing_model(slfm, {'mixture_fraction': PDFSpec('DoubleDelta', z_svv, parallel_type='property-variance')}, num_procs=2, verbose=False)
                slfm_full   = apply_mixing_model(slfm, {'mixture_fraction': PDFSpec('DoubleDelta', z_svv, parallel_type='full')}, num_procs=2, verbose=False)
                slfm_def    = apply_mixing_model(slfm, {'mixture_fraction': PDFSpec('DoubleDelta', z_svv, parallel_type='default')}, num_procs=2, verbose=False)

                try:
                    apply_mixing_model(slfm, {'mixture_fraction': PDFSpec('DoubleDelta', z_svv, parallel_type='mean')}, num_procs=2, verbose=False)
                    self.assertTrue(False)
                except:
                    self.assertTrue(True)

                for prop in slfm.props:
                    if np.max(np.abs(slfm_serial[prop] - slfm_prop[prop])) > 1.e-16:
                        self.assertTrue(False)
                    if np.max(np.abs(slfm_serial[prop] - slfm_mean[prop])) > 1.e-16:
                        self.assertTrue(False)
                    if np.max(np.abs(slfm_serial[prop] - slfm_var[prop])) > 1.e-16:
                        self.assertTrue(False)
                    if np.max(np.abs(slfm_serial[prop] - slfm_full[prop])) > 1.e-16:
                        self.assertTrue(False)
                    if np.max(np.abs(slfm_serial[prop] - slfm_def[prop])) > 1.e-16:
                        self.assertTrue(False)

                # too few points in air_temperature dimension for default spline order 3
                try:
                    apply_mixing_model(eq_lib3, {'mixture_fraction': PDFSpec('ClipGauss', z_svv),
                                                 'air_temperature':PDFSpec('delta')}, num_procs=1, verbose=False)
                    self.assertTrue(False)
                except:
                    self.assertTrue(True)
                # now it's okay with specified spline order 2
                testlib = apply_mixing_model(eq_lib3, {'mixture_fraction': PDFSpec('ClipGauss', z_svv),
                                                       'air_temperature':PDFSpec('delta', convolution_spline_order=2)}, num_procs=1, verbose=False)
                self.assertFalse(np.any(np.isnan(testlib['temperature'])))

                # DeltaPDF w/ and w/o scaled means
                downsize = apply_mixing_model(eq_lib3,
                                              {'mixture_fraction': PDFSpec('delta', mean_values=eq_lib3.mixture_fraction_values[::2]),
                                               'fuel_temperature': PDFSpec('delta', mean_values=eq_lib3.fuel_temperature_values[::2])},
                                              num_procs=1, verbose=False)
                scaled_z = (eq_lib3.mixture_fraction_values[::2] - eq_lib3.mixture_fraction_values.min())/(eq_lib3.mixture_fraction_values.max() - eq_lib3.mixture_fraction_values.min())
                scaled_f = (eq_lib3.fuel_temperature_values[::2] - eq_lib3.fuel_temperature_values.min())/(eq_lib3.fuel_temperature_values.max() - eq_lib3.fuel_temperature_values.min())
                downsize_sc = apply_mixing_model(eq_lib3,
                                              {'mixture_fraction': PDFSpec('delta', scaled_mean_values=scaled_z),
                                               'fuel_temperature': PDFSpec('delta', scaled_mean_values=scaled_f)},
                                              num_procs=1, verbose=False)
                self.assertIsNone(assert_allclose(downsize['temperature'], eq_lib3['temperature'][::2,::2,:]))
                self.assertIsNone(assert_allclose(downsize_sc['temperature'], eq_lib3['temperature'][::2,::2,:]))

                # DeltaPDF with decreasing indepvar (gamma)
                nonad = build_nonadiabatic_defect_transient_slfm_library(fs, diss_rate_values=np.logspace(-1,1,3), n_defect_st=9, verbose=False)
                downsize = apply_mixing_model(nonad,
                                              {'mixture_fraction': PDFSpec('delta', mean_values=nonad.mixture_fraction_values[::2]),
                                               'enthalpy_defect_stoich': PDFSpec('delta', mean_values=nonad.enthalpy_defect_stoich_values[::2])},
                                              num_procs=1, verbose=False)
                self.assertIsNone(assert_allclose(downsize['temperature'], nonad['temperature'][::2,:,::2]))


        if __name__ == '__main__':
            unittest.main()

except ImportError:
    pass
