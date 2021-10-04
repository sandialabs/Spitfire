try:
    import unittest
    from copy import copy
    from os.path import abspath, join

    from numpy.testing import assert_allclose
    import numpy as np
    try:
        from scipy.integrate import simpson
    except ImportError:
        from scipy.integrate import simps as simpson

    from spitfire.chemistry.mechanism import ChemicalMechanismSpec
    from spitfire.chemistry.library import Library, Dimension
    from spitfire.chemistry.flamelet import FlameletSpec
    from spitfire.chemistry.tabulation import build_adiabatic_slfm_library, apply_mixing_model, PDFSpec

    import cantera
    import cantera as ct

    import pytabprops
    if int(cantera.__version__.replace('.', '')) >= 250:
        class Test(unittest.TestCase):
            def test(self):
                test_xml = abspath(join('tests', 'test_mechanisms', 'h2-burke.xml'))
                m = ChemicalMechanismSpec(cantera_xml=test_xml, group_name='h2-burke')
                pressure = 101325.
                air = m.stream(stp_air=True)
                air.TP = 300., pressure
                fuel = m.stream('TPY', (300., pressure, 'H2:1'))

                flamelet_specs = {'mech_spec': m, 'oxy_stream': air, 'fuel_stream': fuel, 'grid_points': 34}

                slfm = build_adiabatic_slfm_library(flamelet_specs,
                                                    diss_rate_values=np.logspace(-1, 4, 40),
                                                    diss_rate_ref='stoichiometric',
                                                    include_extinguished=True,
                                                    verbose=False)

                class LogMean1ParamPDF:
                    def __init__(self, sigma):
                        self._sigma = sigma
                        self._mu = 0.
                        self._s2pi = np.sqrt(2. * np.pi)
                        self._xt = np.logspace(-6, 6, 1000)
                        self._pdft = np.zeros_like(self._xt)

                    def get_pdf(self, x):
                        s = self._sigma
                        m = self._mu
                        return 1. / (x * s * self._s2pi) * np.exp(-(np.log(x) - m) * (np.log(x) - m) / (2. * s * s))

                    def set_mean(self, mean):
                        self._mu = np.log(mean) - 0.5 * self._sigma * self._sigma
                        self._pdft = self.get_pdf(self._xt)

                    def set_variance(self, variance):
                        pass

                    def set_scaled_variance(self, variance):
                        raise ValueError(
                            'cannot use set_scaled_variance on LogMean1ParamPDF, use direct variance values')

                    def integrate(self, interpolant):
                        ig = interpolant(self._xt) * self._pdft
                        return simpson(ig, x=self._xt)

                lm_pdf = LogMean1ParamPDF(1.0)

                mass_fracs = slfm.props
                mass_fracs.remove('temperature')
                slfm.remove(*mass_fracs)

                slfm_l = apply_mixing_model(
                    slfm,
                    mixing_spec={}
                )

                slfm_t = apply_mixing_model(
                    slfm,
                    mixing_spec={'dissipation_rate_stoich': PDFSpec(pdf=lm_pdf, variance_values=np.array([1.]))}
                )

                slfm_t2 = apply_mixing_model(
                    slfm,
                    mixing_spec={'dissipation_rate_stoich': PDFSpec(pdf=lm_pdf, variance_values=np.array([1.])),
                                 'mixture_fraction': 'delta'}
                )

                slfm_t3 = apply_mixing_model(
                    slfm,
                    mixing_spec={'dissipation_rate_stoich': PDFSpec(pdf=lm_pdf, variance_values=np.array([1.])),
                                 'mixture_fraction': PDFSpec(pdf='delta')}
                )

                slfm_t4 = apply_mixing_model(
                    slfm,
                    mixing_spec={'dissipation_rate_stoich': 'delta',
                                 'mixture_fraction': PDFSpec(pdf='delta')},
                    added_suffix='_avg'
                )

                slfm_tt1 = apply_mixing_model(
                    slfm_t,
                    mixing_spec={
                        'mixture_fraction_mean': PDFSpec(pdf='ClipGauss', scaled_variance_values=np.linspace(0, 1, 8))},
                    added_suffix=''
                )

                slfm_tt2 = apply_mixing_model(
                    slfm,
                    mixing_spec={'dissipation_rate_stoich': PDFSpec(pdf=lm_pdf, variance_values=np.array([1.])),
                                 'mixture_fraction': PDFSpec(pdf='ClipGauss',
                                                             scaled_variance_values=np.linspace(0, 1, 8))}
                )

                # todo: this test could use some checks, right now it just makes sure that all of the variations
                # above and the custom PDF can actually run


        if __name__ == '__main__':
            unittest.main()

except ImportError:
    pass
