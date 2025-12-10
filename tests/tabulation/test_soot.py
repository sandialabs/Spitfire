import unittest
from os.path import abspath, join
from spitfire import ChemicalMechanismSpec, Library, apply_mixing_model, PDFSpec, build_adiabatic_bs_library, build_adiabatic_slfm_library
from spitfire import tabulate_gas_radiation_model, tabulate_soot_source_coefficients, tabulate_soot_radiation_model
from spitfire import get_soot_source_term_evaluator, soot_src_aksit_moss, soot_src_aksit_moss_benzene, soot_src_pah_alpha
from spitfire import get_soot_source_term_jacobian_evaluator, jac_soot_src_aksit_moss, jac_soot_src_aksit_moss_benzene, jac_soot_src_pah_alpha
import numpy as np


class TestSoot(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestSoot, self).__init__(*args, **kwargs)
        test_xml = abspath(join('tests', 'test_mechanisms', 'heptane-liu-hewson-chen-pitsch-highT.yaml'))
        self.mech = ChemicalMechanismSpec(cantera_input=test_xml, group_name='gas')
        pressure = 101325.
        air = self.mech.stream(stp_air=True)
        air.TP = 1200., pressure
        fuel = self.mech.stream('TPY', (300., pressure, 'NXC7H16:1'))

        flamelet_specs = {'mech_spec': self.mech, 'oxy_stream': air, 'fuel_stream': fuel, 'grid_points': 34}

        self.bs = build_adiabatic_bs_library(flamelet_specs, verbose=False)
        self.slfm = build_adiabatic_slfm_library(flamelet_specs, verbose=False, diss_rate_values=np.logspace(0, 1, 4), diss_rate_log_scaled=True)

        flamelet_specs['dissipation_rate_form'] = 'constant'
        self.slfm_const = build_adiabatic_slfm_library(flamelet_specs, verbose=False, diss_rate_values=np.logspace(0, 1, 4), diss_rate_log_scaled=True)


    def test_gas_radiation(self):
        accepted_models = ['Barlow_GreyGas']
        for mod in accepted_models:
            try:
                tabulate_gas_radiation_model(self.mech, self.bs, modelname=mod)
                self.assertTrue(True)
            except ValueError:
                self.assertTrue(False)

        try:
            tabulate_gas_radiation_model(self.mech, self.bs, modelname='bleh')
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)

        self.bs = tabulate_gas_radiation_model(self.mech, self.bs)
        self.assertTrue('gas_abs_coeff' in self.bs.props)
        self.assertTrue('gas_rad_source' in self.bs.props)
        self.assertTrue(np.abs(self.bs['gas_abs_coeff']).max()>0.)
        self.assertTrue(np.abs(self.bs['gas_rad_source']).max()>0.)

        lib_no_species = Library(*self.bs.dims)
        for prop in self.bs.props:
            lib_no_species[prop] = self.bs[prop]
        for sp in ['CH4', 'H2O', 'CO', 'CO2']:
            lib_no_species[f'mass fraction {sp}'] = np.zeros(lib_no_species.shape)
        lib_no_species = tabulate_gas_radiation_model(self.mech, lib_no_species)
        self.assertTrue(np.abs(lib_no_species['gas_abs_coeff']).max()==0.)
        self.assertTrue(np.abs(lib_no_species['gas_rad_source']).max()==0.)


    def test_soot_radiation(self):
        accepted_models = ['linear_absorption_coefficient']
        for mod in accepted_models:
            try:
                tabulate_soot_radiation_model(self.mech, self.bs, modelname=mod)
                self.assertTrue(True)
            except ValueError:
                self.assertTrue(False)

        try:
            tabulate_soot_radiation_model(self.mech, self.bs, modelname='bleh')
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)

        self.bs = tabulate_soot_radiation_model(self.mech, self.bs)
        self.assertTrue('soot_abs_coeff' in self.bs.props)
        self.assertTrue('soot_rad_source' in self.bs.props)
        self.assertTrue(np.abs(self.bs['soot_abs_coeff']).max()>0.)
        self.assertTrue(np.abs(self.bs['soot_rad_source']).max()>0.)

        self.bs = tabulate_soot_radiation_model(self.mech, self.bs, input_modelargs={'slope':0., 'intercept':0.})
        self.assertTrue(np.abs(self.bs['soot_abs_coeff']).max()==0.)
        self.assertTrue(np.abs(self.bs['soot_rad_source']).max()==0.)


    def test_soot_kinetics(self):
        test_xml = abspath(join('src', 'spitfire', 'data', 'caltech.yaml'))
        cmech = ChemicalMechanismSpec(cantera_input=test_xml, group_name='gas')
        pressure = 101325.
        air = cmech.stream(stp_air=True)
        air.TP = 300., pressure
        fuel = cmech.stream('TPY', (500., pressure, 'NXC7H16:1.'))
        flamelet_specs = {'mech_spec': cmech, 'oxy_stream': air, 'fuel_stream': fuel, 'grid_points': 34}
        lib_with_A1 = build_adiabatic_bs_library(flamelet_specs, verbose=False)

        # made up compositions to ensure nonzero coeffs
        lib_with_A1['mass fraction A1XC6H6'] = 0.2*lib_with_A1['mass fraction NXC7H16']
        lib_with_A1['mass fraction C2H2'] = 0.1*lib_with_A1['mass fraction NXC7H16']
        lib_with_A1['mass fraction OH'] = 0.08*lib_with_A1['mass fraction NXC7H16']
        lib_with_A1['mass fraction O2'] = 0.09*lib_with_A1['mass fraction NXC7H16']
        lib_with_A1['mass fraction O'] = 0.07*lib_with_A1['mass fraction NXC7H16']
        lib_with_A1['mass fraction A2XC10H8'] = 0.02*lib_with_A1['mass fraction NXC7H16']

        accepted_models = ['Aksit_Moss', 'Aksit_Moss_Benzene', 'PAH_alpha']
        for mod in accepted_models:
            try:
                tabulate_soot_source_coefficients(cmech, lib_with_A1, modelname=mod, verbose=False)
                self.assertTrue(True)
            except ValueError:
                self.assertTrue(False)

        try:
            tabulate_soot_source_coefficients(cmech, lib_with_A1, modelname='bleh', verbose=False)
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)

        expected_src_keys = ['S_Ns', 'S_Ys', 'S_Ns_nucl', 'S_Ns_coag', 'S_Ys_nucl', 'S_Ys_cond',
                             'S_Ys_c2h2', 'S_Ys_o2', 'S_Ys_oh', 'S_Ys_o', 'S_Ys_a1']
        expected_jac_keys = ['dSNs_dNs', 'dSNs_dYs', 'dSYs_dNs', 'dSYs_dYs']

        ns_test = 1.
        ys_test = 1.
        area_fac = np.power(ns_test, 1./3.) * np.power(ys_test, 2./3.)
        idx_test = self.bs.mixture_fraction_npts//2
        tol = 1.e-16

        area_fac_Ns = (1./3.)*np.power(ns_test, -2./3.) * np.power(ys_test, 2./3.)
        area_fac_Ys = (2./3.)*np.power(ns_test, 1./3.) * np.power(ys_test, -1./3.)
        coag_fac_Ns = (11./6.)*np.power(ns_test, 5./6.) * np.power(ys_test, 1./6.)
        coag_fac_Ys = (1./6.)*np.power(ns_test, 11./6.) * np.power(ys_test, -5./6.)

        soot_terms = ['coag', 'nucl_C2H2', 'surf_C2H2', 'surf_O2', 'surf_OH']
        lib_no_A1 = tabulate_soot_source_coefficients(cmech, lib_with_A1, 'Aksit_Moss', 1., verbose=False)
        for prop in soot_terms:
            self.assertTrue(prop in lib_no_A1.props)
            self.assertTrue(np.abs(lib_no_A1[prop][idx_test])>0.)
        try:
            src_eval = get_soot_source_term_evaluator(lib_no_A1, verbose=False)
            src_dict = src_eval(ns_test, ys_test, lib_no_A1['nucl_C2H2'], lib_no_A1['coag'], lib_no_A1['surf_C2H2'], lib_no_A1['surf_O2'], lib_no_A1['surf_OH'])
            self.assertTrue(True)
        except ValueError:
            self.assertTrue(False)
        for key in expected_src_keys:
            self.assertTrue(key in src_dict)
        for key in ['S_Ys_cond', 'S_Ys_o', 'S_Ys_a1']:
            self.assertTrue(np.abs(src_dict[key]).max()==0.)

        try:
            jac_eval = get_soot_source_term_jacobian_evaluator(lib_no_A1, verbose=False)
            jac_dict = jac_eval(ns_test, ys_test, lib_no_A1['nucl_C2H2'], lib_no_A1['coag'], lib_no_A1['surf_C2H2'], lib_no_A1['surf_O2'], lib_no_A1['surf_OH'])
            self.assertTrue(True)
        except ValueError:
            self.assertTrue(False)
        for key in expected_jac_keys:
            self.assertTrue(key in jac_dict)

        nucl_C2H2 = src_dict['S_Ns_nucl'][idx_test]
        coag = -src_dict['S_Ns_coag'][idx_test] / np.power(ns_test, 11./6.) / np.power(ys_test, 1./6.)
        surf_C2H2 = src_dict['S_Ys_c2h2'][idx_test] / area_fac
        surf_O2 = -src_dict['S_Ys_o2'][idx_test] / area_fac
        surf_OH = -src_dict['S_Ys_oh'][idx_test] / area_fac
        test_dict = soot_src_aksit_moss(ns_test, ys_test, nucl_C2H2, coag, surf_C2H2, surf_O2, surf_OH)
        self.assertTrue(np.abs(test_dict['S_Ns']  - (nucl_C2H2 - coag * np.power(ns_test, 11./6.) * np.power(ys_test, 1./6.))) <= tol)
        self.assertTrue(np.abs(test_dict['S_Ys']  - (144.*nucl_C2H2 + area_fac*(surf_C2H2 - surf_O2 - surf_OH))) <= tol)

        test_jac_dict = jac_soot_src_aksit_moss(ns_test, ys_test, nucl_C2H2, coag, surf_C2H2, surf_O2, surf_OH)
        self.assertTrue(np.abs(test_jac_dict['dSNs_dNs'] - (-coag*coag_fac_Ns)) <= tol)
        self.assertTrue(np.abs(test_jac_dict['dSNs_dYs'] - (-coag*coag_fac_Ys)) <= tol)
        self.assertTrue(np.abs(test_jac_dict['dSYs_dNs'] - ((surf_C2H2-surf_O2-surf_OH)*area_fac_Ns)) <= tol)
        self.assertTrue(np.abs(test_jac_dict['dSYs_dYs'] - ((surf_C2H2-surf_O2-surf_OH)*area_fac_Ys)) <= tol)

        test_dict = soot_src_aksit_moss(ns_test, ys_test, nucl_C2H2, coag, surf_C2H2, surf_O2, surf_OH, {'coag':2.})
        self.assertTrue(np.abs(test_dict['S_Ns'] - (nucl_C2H2 - 2. * coag * np.power(ns_test, 11./6.) * np.power(ys_test, 1./6.))) <= tol)
        test_jac_dict = jac_soot_src_aksit_moss(ns_test, ys_test, nucl_C2H2, coag, surf_C2H2, surf_O2, surf_OH, {'coag':2.})
        self.assertTrue(np.abs(test_jac_dict['dSNs_dNs'] - (-2.*coag*coag_fac_Ns)) <= tol)
        self.assertTrue(np.abs(test_jac_dict['dSNs_dYs'] - (-2.*coag*coag_fac_Ys)) <= tol)

        soot_terms += ['surf_A1']
        lib_with_A1 = tabulate_soot_source_coefficients(cmech, lib_with_A1, 'Aksit_Moss_Benzene', 1., verbose=False)
        
        for prop in soot_terms:
            self.assertTrue(prop in lib_with_A1.props)
            self.assertTrue(np.abs(lib_with_A1[prop][idx_test])>0.)
        try:
            src_eval = get_soot_source_term_evaluator(lib_with_A1, verbose=False)
            src_dict = src_eval(ns_test, ys_test, lib_with_A1['nucl_C2H2'], lib_with_A1['coag'], lib_with_A1['surf_C2H2'], lib_with_A1['surf_O2'], lib_with_A1['surf_OH'], lib_with_A1['surf_A1'])
            self.assertTrue(True)
        except ValueError:
            self.assertTrue(False)
        for key in expected_src_keys:
            self.assertTrue(key in src_dict)
        for key in ['S_Ys_cond', 'S_Ys_o']:
            self.assertTrue(np.abs(src_dict[key]).max()==0.)

        try:
            jac_eval = get_soot_source_term_jacobian_evaluator(lib_with_A1, verbose=False)
            jac_dict = jac_eval(ns_test, ys_test, lib_with_A1['nucl_C2H2'], lib_with_A1['coag'], lib_with_A1['surf_C2H2'], lib_with_A1['surf_O2'], lib_with_A1['surf_OH'], lib_with_A1['surf_A1'])
            self.assertTrue(True)
        except ValueError:
            self.assertTrue(False)
        for key in expected_jac_keys:
            self.assertTrue(key in jac_dict)

        nucl_C2H2 = src_dict['S_Ns_nucl'][idx_test]
        coag = -src_dict['S_Ns_coag'][idx_test] / np.power(ns_test, 11./6.) / np.power(ys_test, 1./6.)
        surf_C2H2 = src_dict['S_Ys_c2h2'][idx_test] / area_fac
        surf_O2 = -src_dict['S_Ys_o2'][idx_test] / area_fac
        surf_OH = -src_dict['S_Ys_oh'][idx_test] / area_fac
        surf_A1 = -src_dict['S_Ys_a1'][idx_test] / area_fac
        test_dict = soot_src_aksit_moss_benzene(ns_test, ys_test, nucl_C2H2, coag, surf_C2H2, surf_O2, surf_OH, surf_A1)
        self.assertTrue(np.abs(test_dict['S_Ns'] - (nucl_C2H2 - coag * np.power(ns_test, 11./6.) * np.power(ys_test, 1./6.))) <= tol)
        self.assertTrue(np.abs(test_dict['S_Ys'] - (144.*nucl_C2H2 + area_fac*(surf_C2H2 + surf_A1 - surf_O2 - surf_OH))) <= tol)

        test_jac_dict = jac_soot_src_aksit_moss_benzene(ns_test, ys_test, nucl_C2H2, coag, surf_C2H2, surf_O2, surf_OH, surf_A1)
        self.assertTrue(np.abs(test_jac_dict['dSNs_dNs'] - (-coag*coag_fac_Ns)) <= tol)
        self.assertTrue(np.abs(test_jac_dict['dSNs_dYs'] - (-coag*coag_fac_Ys)) <= tol)
        self.assertTrue(np.abs(test_jac_dict['dSYs_dNs'] - ((surf_C2H2+surf_A1-surf_O2-surf_OH)*area_fac_Ns)) <= tol)
        self.assertTrue(np.abs(test_jac_dict['dSYs_dYs'] - ((surf_C2H2+surf_A1-surf_O2-surf_OH)*area_fac_Ys)) <= tol)

        for prop in soot_terms:
            lib_with_A1.remove(prop)
        soot_terms = ['coag', 'ox_O', 'ox_OH', 'ox_O2', 'sg', 'beta_n', 'beta_c_pre', 'dimer_src']
        lib_with_A1 = tabulate_soot_source_coefficients(cmech, lib_with_A1, 'PAH_alpha', 1., verbose=False)
        for prop in soot_terms:
            self.assertTrue(prop in lib_with_A1.props)
            self.assertTrue(np.abs(lib_with_A1[prop][idx_test])>0.)
        try:
            src_eval = get_soot_source_term_evaluator(lib_with_A1, verbose=False)
            src_dict = src_eval(ns_test, ys_test, lib_with_A1['dimer_src'], lib_with_A1['coag'], lib_with_A1['sg'], lib_with_A1['ox_O2'], lib_with_A1['ox_OH'], lib_with_A1['ox_O'])
            self.assertTrue(True)
        except ValueError:
            self.assertTrue(False)
        for key in expected_src_keys:
            self.assertTrue(key in src_dict)
        for key in ['S_Ys_cond', 'S_Ys_a1']:
            self.assertTrue(np.abs(src_dict[key]).max()==0.)

        try:
            jac_eval = get_soot_source_term_jacobian_evaluator(lib_with_A1, verbose=False)
            jac_dict = jac_eval(ns_test, ys_test, lib_with_A1['dimer_src'], lib_with_A1['coag'], lib_with_A1['sg'], lib_with_A1['ox_O2'], lib_with_A1['ox_OH'], lib_with_A1['ox_O'])
            self.assertTrue(True)
        except ValueError:
            self.assertTrue(False)
        for key in expected_jac_keys:
            self.assertTrue(key in jac_dict)

        dimer_src = src_dict['S_Ys_nucl'][idx_test] / 1.e-6
        coag = -src_dict['S_Ns_coag'][idx_test] / np.power(ns_test, 11./6.) / np.power(ys_test, 1./6.)
        sg = src_dict['S_Ys_c2h2'][idx_test] / area_fac
        ox_O2 = -src_dict['S_Ys_o2'][idx_test] / area_fac
        ox_OH = -src_dict['S_Ys_oh'][idx_test] / area_fac
        ox_O = -src_dict['S_Ys_o'][idx_test] / area_fac
        test_dict = soot_src_pah_alpha(ns_test, ys_test, dimer_src, coag, sg, ox_O2, ox_OH, ox_O)
        self.assertTrue(np.abs(test_dict['S_Ns']  - (dimer_src * 1.e-6 * 0.5 / 361.4 - coag * np.power(ns_test, 11./6.) * np.power(ys_test, 1./6.))) <= tol)
        self.assertTrue(np.abs(test_dict['S_Ys']  - (dimer_src * 1.e-6 + area_fac * (sg - ox_O2 - ox_OH - ox_O))) <= tol)

        test_jac_dict = jac_soot_src_pah_alpha(ns_test, ys_test, dimer_src, coag, sg, ox_O2, ox_OH, ox_O)
        self.assertTrue(np.abs(test_jac_dict['dSNs_dNs'] - (-coag*coag_fac_Ns)) <= tol)
        self.assertTrue(np.abs(test_jac_dict['dSNs_dYs'] - (-coag*coag_fac_Ys)) <= tol)
        self.assertTrue(np.abs(test_jac_dict['dSYs_dNs'] - ((sg-ox_O2-ox_OH-ox_O)*area_fac_Ns)) <= tol)
        self.assertTrue(np.abs(test_jac_dict['dSYs_dYs'] - ((sg-ox_O2-ox_OH-ox_O)*area_fac_Ys)) <= tol)

        percent_test = 0.01
        try:
            tabulate_soot_source_coefficients(self.mech, self.bs, 'Aksit_Moss', percent_test) # no chi in lib
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)

        try:
            tabulate_soot_source_coefficients(self.mech, self.slfm, 'Aksit_Moss', percent_test)
            tabulate_soot_source_coefficients(self.mech, self.slfm_const, 'Aksit_Moss', percent_test)
            self.assertTrue(True)
        except ValueError:
            self.assertTrue(False)
        self.slfm = tabulate_soot_source_coefficients(self.mech, self.slfm, 'Aksit_Moss', percent_test)
        self.assertTrue('soot_subfilter' in self.slfm.props)
        self.assertTrue(np.abs(self.slfm['soot_subfilter'].min() - percent_test) < 1.e-16)

        turblib = apply_mixing_model(Library.squeeze(self.slfm[:,0]),
                                     {'mixture_fraction':PDFSpec('clipgauss',scaled_variance_values=np.array([0.5]),mean_values=np.array([0.,0.01,1.]))})
        positive_thresh = 1.e-16
        for prop in ['nucl_C2H2', 'surf_C2H2', 'surf_O2', 'surf_OH']:
            self.assertTrue(turblib[prop].min() >= positive_thresh)
        self.assertTrue(turblib['soot_subfilter'].min() >= percent_test)
        self.assertTrue(turblib['soot_subfilter'].max() <= 1.)


if __name__ == '__main__':
    unittest.main()
