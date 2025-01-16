import unittest
from os.path import abspath, join
from spitfire import ChemicalMechanismSpec, Library, apply_mixing_model, PDFSpec, build_adiabatic_bs_library, build_adiabatic_slfm_library
from spitfire import tabulate_gas_radiation_model, tabulate_soot_source_coefficients, tabulate_soot_radiation_model, compute_soot_source_terms
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

        tabulate_gas_radiation_model(self.mech, self.bs)
        self.assertTrue('gas_abs_coeff' in self.bs.props)
        self.assertTrue('gas_rad_source' in self.bs.props)
        self.assertTrue(np.abs(self.bs['gas_abs_coeff']).max()>0.)
        self.assertTrue(np.abs(self.bs['gas_rad_source']).max()>0.)

        lib_no_species = Library(*self.bs.dims)
        for prop in self.bs.props:
            lib_no_species[prop] = self.bs[prop]
        for sp in ['CH4', 'H2O', 'CO', 'CO2']:
            lib_no_species[f'mass fraction {sp}'] = np.zeros(lib_no_species.shape)
        tabulate_gas_radiation_model(self.mech, lib_no_species)
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

        tabulate_soot_radiation_model(self.mech, self.bs)
        self.assertTrue('soot_abs_coeff' in self.bs.props)
        self.assertTrue('soot_rad_source' in self.bs.props)
        self.assertTrue(np.abs(self.bs['soot_abs_coeff']).max()==0.)
        self.assertTrue(np.abs(self.bs['soot_rad_source']).max()==0.)

        tabulate_soot_radiation_model(self.mech, self.bs, input_modelargs={'slope':1735., 'intercept':-3.75e5})
        self.assertTrue(np.abs(self.bs['soot_abs_coeff']).max()>0.)
        self.assertTrue(np.abs(self.bs['soot_rad_source']).max()>0.)


    def test_soot_kinetics(self):
        test_xml = abspath(join('src', 'spitfire', 'data', 'caltech.yaml'))
        cmech = ChemicalMechanismSpec(cantera_input=test_xml, group_name='gas')
        pressure = 101325.
        air = cmech.stream(stp_air=True)
        air.TP = 1200., pressure
        fuel = cmech.stream('TPY', (300., pressure, 'NXC7H16:1'))
        flamelet_specs = {'mech_spec': cmech, 'oxy_stream': air, 'fuel_stream': fuel, 'grid_points': 34}
        lib_with_A1 = build_adiabatic_bs_library(flamelet_specs, verbose=False)

        accepted_models = ['Aksit_Moss', 'Aksit_Moss_Benzene', 'PAH_alpha']
        for mod in accepted_models:
            try:
                tabulate_soot_source_coefficients(cmech, lib_with_A1, modelname=mod, verbose=False)
                self.assertTrue(True)
            except ValueError:
                self.assertTrue(False)

        try:
            tabulate_soot_source_coefficients(self.mech, self.bs, modelname='bleh', verbose=False)
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)

        soot_terms = ['coag', 'nucl_C2H2', 'surf_C2H2', 'surf_O2', 'surf_OH']
        tabulate_soot_source_coefficients(self.mech, self.bs, 'Aksit_Moss', 0., verbose=False)
        for prop in soot_terms:
            self.assertTrue(prop in self.bs.props)
        try:
            compute_soot_source_terms(self.bs, 1., 1., verbose=False)
            self.assertTrue(True)
        except ValueError:
            self.assertTrue(False)
        try:
            compute_soot_source_terms(self.bs, np.array([1.]), np.array([1.]), verbose=False)
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)

        soot_terms += ['surf_A1']
        tabulate_soot_source_coefficients(cmech, lib_with_A1, 'Aksit_Moss_Benzene', 0., verbose=False)
        for prop in soot_terms:
            self.assertTrue(prop in lib_with_A1.props)
        try:
            compute_soot_source_terms(lib_with_A1, 1., 1., verbose=False)
            self.assertTrue(True)
        except ValueError:
            self.assertTrue(False)

        soot_terms = ['coag', 'ox_O', 'ox_OH', 'ox_O2', 'sg', 'beta_n', 'beta_c_pre', 'dimer_src']
        tabulate_soot_source_coefficients(cmech, lib_with_A1, 'PAH_alpha', 0., verbose=False)
        for prop in soot_terms:
            self.assertTrue(prop in lib_with_A1.props)
        try:
            compute_soot_source_terms(lib_with_A1, 1., 1., verbose=False)
            self.assertTrue(True)
        except ValueError:
            self.assertTrue(False)

        percent_test = 0.01
        try:
            tabulate_soot_source_coefficients(self.mech, self.bs, 'Aksit_Moss', percent_test)
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)

        try:
            tabulate_soot_source_coefficients(self.mech, self.slfm, 'Aksit_Moss', percent_test)
            self.assertTrue(True)
        except ValueError:
            self.assertTrue(False)
        self.assertTrue('soot_correction' in self.slfm.props)
        self.assertTrue(self.slfm['soot_correction'].min() - percent_test < 1.e-16)

        turblib = apply_mixing_model(Library.squeeze(self.slfm[:,0]),
                                     {'mixture_fraction':PDFSpec('clipgauss',scaled_variance_values=np.array([0.5]),mean_values=np.array([0.,0.01,1.]))})
        positive_thresh = 1.e-16
        for prop in ['nucl_C2H2', 'surf_C2H2', 'surf_O2', 'surf_OH']:
            self.assertTrue(turblib[prop].min() >= positive_thresh)
        self.assertTrue(turblib['soot_correction'].min() >= percent_test)
        self.assertTrue(turblib['soot_correction'].max() <= 1.)


if __name__ == '__main__':
    unittest.main()
