try:
    import pytabprops
    import unittest
    from os.path import join, abspath

    from numpy.testing import assert_allclose

    from tests.tabulation.turb_adiabatic_equilibrium.rebless import run
    from spitfire.chemistry.library import Library

    import cantera

    if int(cantera.__version__.replace('.', '')) >= 250:
        class Test(unittest.TestCase):
            def test(self):
                output_library = run()

                gold_file = abspath(join('tests',
                                         'tabulation',
                                         'turb_adiabatic_equilibrium',
                                         'gold.pkl'))
                gold_library = Library.load_from_file(gold_file)

                for prop in gold_library.props:
                    self.assertIsNone(assert_allclose(gold_library[prop], output_library[prop], atol=1.e-14))

                self.assertEqual(output_library.dims[0].name, 'mixture_fraction_mean')
                self.assertEqual(output_library.dims[1].name, 'scaled_scalar_variance_mean')


        if __name__ == '__main__':
            unittest.main()

except ImportError:
    pass
