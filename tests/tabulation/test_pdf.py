try:
    import unittest
    from spitfire import BetaPDF, DoubleDeltaPDF, apply_mixing_model, PDFSpec, Library, Dimension, compute_pdf_max_integration_errors
    import numpy as np
    from pytabprops import LagrangeInterpolant1D


    class TestPDF(unittest.TestCase):

        def __init__(self, *args, **kwargs):
            super(TestPDF, self).__init__(*args, **kwargs)
            self._svar = np.array([0., 1.e-5, 6.e-4, 1.e-3, 0.1, 0.5, 0.8, 0.86, 0.9, 0.95, 1.])
            self._mean = np.hstack((0,np.logspace(-5,0,100)))
            self._bp = BetaPDF()
            self._ddelta = DoubleDeltaPDF()
            self._maxerrsdd = compute_pdf_max_integration_errors(self._ddelta, self._mean, self._svar)


        def _pdf_integral(self, pdf_class, tolerance):
            errors = np.zeros((self._svar.size * self._mean.size))
            i = 0
            for svar in self._svar:
                for mean in self._mean:
                    pdf_class.set_mean(mean)
                    pdf_class.set_scaled_variance(svar)
                    interp = LagrangeInterpolant1D(3, self._mean, np.ones_like(self._mean), False)
                    integral = pdf_class.integrate(interp)
                    errors[i] = integral - 1.
                    i += 1
            maxerr = np.max(np.abs(errors))
            if maxerr > tolerance:
                self.assertTrue(False)
            else:
                self.assertTrue(True)
            return maxerr

        def test_beta_satifies_pdf_integral(self):
            err = self._pdf_integral(self._bp, 6.e-4)

        def test_doubledelta_satifies_pdf_integral(self):
            err = self._pdf_integral(self._ddelta, 1.e-10)
            if np.abs(err - self._maxerrsdd[0]) > 1.e-16:
                self.assertTrue(False)


        def _mean_integral(self, pdf_class, tolerance):
            errors = np.zeros((self._svar.size * self._mean.size))
            i = 0
            for svar in self._svar:
                for mean in self._mean:
                    pdf_class.set_mean(mean)
                    pdf_class.set_scaled_variance(svar)
                    interp = LagrangeInterpolant1D(3, self._mean, self._mean, False)
                    integral = pdf_class.integrate(interp)
                    errors[i] = (integral - mean)/(mean+1.e-6)
                    i += 1
            maxerr = np.max(np.abs(errors))
            if maxerr > tolerance:
                self.assertTrue(False)
            else:
                self.assertTrue(True)
            return maxerr

        def test_beta_satifies_mean_integral(self):
            err = self._mean_integral(self._bp, 3.e-6)

        def test_doubledelta_satifies_mean_integral(self):
            err = self._mean_integral(self._ddelta, 1.e-10)
            if np.abs(err - self._maxerrsdd[1]) > 1.e-16:
                self.assertTrue(False)


        def _variance_integral(self, pdf_class, tolerance):
            errors = np.zeros((self._svar.size * self._mean.size))
            i = 0
            for svar in self._svar:
                for mean in self._mean:
                    pdf_class.set_mean(mean)
                    pdf_class.set_scaled_variance(svar)
                    maxvar = mean * (1.-mean)
                    interp = LagrangeInterpolant1D(3, self._mean, (self._mean - mean)*(self._mean - mean), False)
                    integral = pdf_class.integrate(interp)
                    errors[i] = (integral - svar*maxvar)/(svar*maxvar+1.e-6)
                    i += 1
            maxerr = np.max(np.abs(errors))
            if maxerr > tolerance:
                self.assertTrue(False)
            else:
                self.assertTrue(True)
            return maxerr

        def test_beta_satifies_variance_integral(self):
            err = self._variance_integral(self._bp, 1.e-2)

        def test_doubledelta_satifies_variance_integral(self):
            err = self._variance_integral(self._ddelta, 1.e-2)
            if np.abs(err - self._maxerrsdd[2]) > 1.e-16:
                self.assertTrue(False)


        def test_supported_pdfs(self):
            lib = Library(Dimension('mixture_fraction', np.linspace(0,1,100)))
            lib['prop'] = lib.mixture_fraction_values**2+0.1
            for pdfname in ['ClipGauss', 'Beta', 'DoubleDelta']:
                try:
                    lib1 = apply_mixing_model(lib, {'mixture_fraction': PDFSpec(pdf=pdfname, scaled_variance_values=np.array([1.e-3]))}, verbose=False)
                    self.assertTrue(True)
                    if pdfname=='Beta':
                        lib2 = apply_mixing_model(lib, {'mixture_fraction': PDFSpec(pdf='beta', scaled_variance_values=np.array([1.e-3]))}, verbose=False)
                        self.assertTrue(True)
                        for prop in lib1.props:
                            maxerr=np.max(np.abs(lib1[prop]-lib2[prop]))
                            if maxerr > 1.e-16:
                                self.assertTrue(False)
                            else:
                                self.assertTrue(True)
                except:
                    self.assertTrue(False)
            for pdfname in ['UNKNOWN']:
                try:
                    apply_mixing_model(lib, {'mixture_fraction': PDFSpec(pdf=pdfname, scaled_variance_values=np.array([1.e-3]))}, verbose=False)
                    self.assertTrue(False)
                except:
                    self.assertTrue(True)


    if __name__ == '__main__':
        unittest.main()

except ImportError:
    pass
