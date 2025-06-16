import unittest
from tests.tabulation.nonadiabatic_defect_with_superadiabatic.get_libs import run_slfm, run_eq, get_abs_relerr, get_params, run_inputs_check
from spitfire.chemistry.ctversion import check as cantera_version_check
import numpy as np


if cantera_version_check('atleast', 2, 5, None):

    class Test(unittest.TestCase):
        def test_transient_slfm_superad(self):
            relthresh = 1.e-5
            n_superad = 3
            n_subad = 5
            lib, lib_sup, lib_sub, lib_both, lib_shift = run_slfm(n_superad, n_subad, transient=True, num_procs=1)

            self.assertTrue(lib.enthalpy_defect_stoich_npts==lib_sub.enthalpy_defect_stoich_npts)
            self.assertTrue(lib.enthalpy_defect_stoich_npts+n_superad==lib_sup.enthalpy_defect_stoich_npts)
            self.assertTrue(lib.enthalpy_defect_stoich_npts+n_superad==lib_both.enthalpy_defect_stoich_npts)
            self.assertTrue(lib_both.enthalpy_defect_stoich_npts==lib_shift.enthalpy_defect_stoich_npts)

            self.assertTrue(get_abs_relerr(lib_sup.enthalpy_defect_stoich_values[n_superad:],lib.enthalpy_defect_stoich_values).max()<=relthresh)
            suberr = get_abs_relerr(lib_sub.enthalpy_defect_stoich_values,lib.enthalpy_defect_stoich_values)
            self.assertTrue(suberr[0]<=relthresh)
            self.assertTrue(suberr.max()>relthresh)
            self.assertTrue(get_abs_relerr(lib_sup.enthalpy_defect_stoich_values[:n_superad+1],lib_both.enthalpy_defect_stoich_values[:n_superad+1]).max()<=relthresh)
            self.assertTrue(get_abs_relerr(lib_sub.enthalpy_defect_stoich_values,lib_both.enthalpy_defect_stoich_values[n_superad:]).max()<=relthresh)
            shifterr = get_abs_relerr(lib_shift.enthalpy_defect_stoich_values,lib_both.enthalpy_defect_stoich_values)
            self.assertTrue(shifterr[n_superad]<=relthresh)
            self.assertTrue(shifterr.max()>relthresh)

            for prop in lib.props:
                self.assertTrue(get_abs_relerr(lib_sup[prop][:,:,n_superad:],lib[prop]).max()<=relthresh)
                self.assertTrue(get_abs_relerr(lib_sub[prop][:,:,0],lib[prop][:,:,0]).max()<=relthresh)
                self.assertTrue(get_abs_relerr(lib_sup[prop][:,:,:n_superad+1],lib_both[prop][:,:,:n_superad+1]).max()<=relthresh)
                self.assertTrue(get_abs_relerr(lib_sub[prop],lib_both[prop][:,:,n_superad:]).max()<=relthresh)

            prop = 'temperature'
            tol = 1.e-12
            _, _, fuel_shift, _ = get_params()
            self.assertTrue(np.abs(lib_shift[prop][-1,:,0] - (lib_both[prop][-1,:,0]+fuel_shift))<=tol)
            self.assertTrue(np.abs(lib_shift[prop][-1,:,n_superad] - lib_both[prop][-1,:,n_superad]).max()<=tol)
            self.assertTrue(np.abs(lib_shift[prop][-1,:,-1] - (lib_both[prop][-1,:,-1]-fuel_shift))<=tol)
            self.assertTrue(np.abs(lib_shift[prop][0,:,0] - lib_both[prop][0,:,0]).max()<=tol)
            self.assertTrue(np.abs(lib_shift[prop][0,:,n_superad] - lib_both[prop][0,:,n_superad]).max()<=tol)
            self.assertTrue(np.abs(lib_shift[prop][0,:,-1] - lib_both[prop][0,:,-1]).max()<=tol)

        def test_steady_slfm_superad(self):
            relthresh = 1.e-5
            n_superad = 3
            n_subad = 5
            lib, lib_sup, lib_sub, lib_both, lib_shift = run_slfm(n_superad, n_subad, transient=False, num_procs=1)

            self.assertTrue(lib.enthalpy_defect_stoich_npts==lib_sub.enthalpy_defect_stoich_npts)
            self.assertTrue(lib.enthalpy_defect_stoich_npts+n_superad==lib_sup.enthalpy_defect_stoich_npts)
            self.assertTrue(lib.enthalpy_defect_stoich_npts+n_superad==lib_both.enthalpy_defect_stoich_npts)
            self.assertTrue(lib_both.enthalpy_defect_stoich_npts==lib_shift.enthalpy_defect_stoich_npts)

            self.assertTrue(get_abs_relerr(lib_sup.enthalpy_defect_stoich_values[n_superad:],lib.enthalpy_defect_stoich_values).max()<=relthresh)
            suberr = get_abs_relerr(lib_sub.enthalpy_defect_stoich_values,lib.enthalpy_defect_stoich_values)
            self.assertTrue(suberr[0]<=relthresh)
            self.assertTrue(suberr.max()>relthresh)
            self.assertTrue(get_abs_relerr(lib_sup.enthalpy_defect_stoich_values[:n_superad+1],lib_both.enthalpy_defect_stoich_values[:n_superad+1]).max()<=relthresh)
            self.assertTrue(get_abs_relerr(lib_sub.enthalpy_defect_stoich_values,lib_both.enthalpy_defect_stoich_values[n_superad:]).max()<=relthresh)
            shifterr = get_abs_relerr(lib_shift.enthalpy_defect_stoich_values,lib_both.enthalpy_defect_stoich_values)
            self.assertTrue(shifterr[n_superad]<=relthresh)
            self.assertTrue(shifterr.max()>relthresh)

            for prop in lib.props:
                self.assertTrue(get_abs_relerr(lib_sup[prop][:,:,n_superad:],lib[prop]).max()<=relthresh)
                self.assertTrue(get_abs_relerr(lib_sub[prop][:,:,0],lib[prop][:,:,0]).max()<=relthresh)
                self.assertTrue(get_abs_relerr(lib_sup[prop][:,:,:n_superad+1],lib_both[prop][:,:,:n_superad+1]).max()<=relthresh)
                self.assertTrue(get_abs_relerr(lib_sub[prop],lib_both[prop][:,:,n_superad:]).max()<=relthresh)

            prop = 'temperature'
            tol = 1.e-12
            _, _, fuel_shift, _ = get_params()
            self.assertTrue(np.abs(lib_shift[prop][-1,:,0] - (lib_both[prop][-1,:,0]+fuel_shift))<=tol)
            self.assertTrue(np.abs(lib_shift[prop][-1,:,n_superad] - lib_both[prop][-1,:,n_superad]).max()<=tol)
            self.assertTrue(np.abs(lib_shift[prop][-1,:,-1] - (lib_both[prop][-1,:,-1]-fuel_shift))<=tol)
            self.assertTrue(np.abs(lib_shift[prop][0,:,0] - lib_both[prop][0,:,0]).max()<=tol)
            self.assertTrue(np.abs(lib_shift[prop][0,:,n_superad] - lib_both[prop][0,:,n_superad]).max()<=tol)
            self.assertTrue(np.abs(lib_shift[prop][0,:,-1] - lib_both[prop][0,:,-1]).max()<=tol)

        def test_eq_superad(self):
            relthresh = 1.e-5
            n_superad = 3
            n_subad = 5
            lib, lib_sup, lib_sub, lib_both, lib_shift = run_eq(n_superad, n_subad)

            self.assertTrue(lib.enthalpy_defect_stoich_npts==lib_sub.enthalpy_defect_stoich_npts)
            self.assertTrue(lib.enthalpy_defect_stoich_npts+n_superad==lib_sup.enthalpy_defect_stoich_npts)
            self.assertTrue(lib.enthalpy_defect_stoich_npts+n_superad==lib_both.enthalpy_defect_stoich_npts)
            self.assertTrue(lib_both.enthalpy_defect_stoich_npts==lib_shift.enthalpy_defect_stoich_npts)

            self.assertTrue(get_abs_relerr(lib_sup.enthalpy_defect_stoich_values[n_superad:],lib.enthalpy_defect_stoich_values).max()<=relthresh)
            suberr = get_abs_relerr(lib_sub.enthalpy_defect_stoich_values,lib.enthalpy_defect_stoich_values)
            self.assertTrue(suberr[0]<=relthresh)
            self.assertTrue(suberr.max()>relthresh)
            self.assertTrue(get_abs_relerr(lib_sup.enthalpy_defect_stoich_values[:n_superad+1],lib_both.enthalpy_defect_stoich_values[:n_superad+1]).max()<=relthresh)
            self.assertTrue(get_abs_relerr(lib_sub.enthalpy_defect_stoich_values,lib_both.enthalpy_defect_stoich_values[n_superad:]).max()<=relthresh)
            shifterr = get_abs_relerr(lib_shift.enthalpy_defect_stoich_values,lib_both.enthalpy_defect_stoich_values)
            self.assertTrue(shifterr[n_superad]<=relthresh)
            self.assertTrue(shifterr.max()>relthresh)

            for prop in lib.props:
                self.assertTrue(get_abs_relerr(lib_sup[prop][:,n_superad:],lib[prop]).max()<=relthresh)
                self.assertTrue(get_abs_relerr(lib_sub[prop][:,0],lib[prop][:,0]).max()<=relthresh)
                self.assertTrue(get_abs_relerr(lib_sup[prop][:,:n_superad+1],lib_both[prop][:,:n_superad+1]).max()<=relthresh)
                self.assertTrue(get_abs_relerr(lib_sub[prop],lib_both[prop][:,n_superad:]).max()<=relthresh)

            prop = 'temperature'
            tol = 1.e-12
            _, _, fuel_shift, _ = get_params()
            self.assertTrue(np.abs(lib_shift[prop][-1,0] - (lib_both[prop][-1,0]+fuel_shift))<=tol)
            self.assertTrue(np.abs(lib_shift[prop][-1,-1] - (lib_both[prop][-1,-1]-fuel_shift))<=tol)
            self.assertTrue(np.abs(lib_shift[prop][0,:] - lib_both[prop][0,:]).max()<=tol)

        def test_superad_check_inputs(self):
            self.assertTrue(run_inputs_check('eq'))
            self.assertTrue(run_inputs_check('transient_slfm'))
            self.assertTrue(run_inputs_check('steady_slfm'))


    if __name__ == '__main__':
        unittest.main()
