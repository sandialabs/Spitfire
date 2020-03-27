import unittest
from numpy import max, abs, ones, zeros, hstack
from cantera import gas_constant
from spitfire import ChemicalMechanismSpec as Mechanism
import cantera as ct

species_decl = list([ct.Species('A', 'H:2'),
                     ct.Species('B', 'H:1'),
                     ct.Species('C', 'O:1'),
                     ct.Species('D', 'O:2'),
                     ct.Species('E', 'H:1, O:2'),
                     ct.Species('N', 'H:1, O:1')])
species_dict = dict({'const': list(), 'nasa7': list()})
for i, s in enumerate(species_decl):
    species_dict['const'].append(ct.Species(s.name, s.composition))
    species_dict['const'][-1].thermo = ct.ConstantCp(300., 3000., 101325., (300., 0., 0., float(i+1) * 1.e4))
    species_dict['nasa7'].append(ct.Species(s.name, s.composition))
    coeffs = [float(i+1) * v for v in [1.e0, 1.e-2, 1.e-4, 1.e-6, 1.e-8, 1.e-10, 1.e-12]]
    species_dict['nasa7'][-1].thermo = ct.NasaPoly2(300., 3000., 101325., hstack([1200.] + coeffs + coeffs))

mechs = [('const', Mechanism.from_solution(ct.Solution(thermo='IdealGas',
                                                       kinetics='GasKinetics',
                                                       species=species_dict['const'],
                                                       reactions=[]))),
         ('nasa7', Mechanism.from_solution(ct.Solution(thermo='IdealGas',
                                                       kinetics='GasKinetics',
                                                       species=species_dict['nasa7'],
                                                       reactions=[])))]

tolerance = 1.e-14


def do_mmw(griffon, gas, T, p, y):
    gas.TPY = T, p, y
    ct = gas.mean_molecular_weight
    gr = griffon.mixture_molecular_weight(y)
    return abs(gr - ct) / abs(gr) < tolerance


def do_density(griffon, gas, T, p, y):
    gas.TPY = T, p, y
    ct = gas.density_mass
    gr = griffon.ideal_gas_density(p, T, y)
    return abs(gr - ct) / abs(gr) < tolerance


def do_pressure(griffon, gas, T, p, y):
    gas.TPY = T, p, y
    rho = gas.density_mass
    gr = griffon.ideal_gas_pressure(rho, T, y)
    return abs(gr - p) / abs(gr) < tolerance


def do_cpmix(griffon, gas, T, p, y):
    gas.TPY = T, p, y
    ct = gas.cp_mass
    gr = griffon.cp_mix(T, y)
    return abs(gr - ct) / abs(gr) < tolerance


def do_cvmix(griffon, gas, T, p, y):
    gas.TPY = T, p, y
    ct = gas.cv_mass
    gr = griffon.cv_mix(T, y)
    return abs(gr - ct) / abs(gr) < tolerance


def do_cpspec(griffon, gas, T, p, y):
    gas.TPY = T, p, y
    ct = gas.standard_cp_R * gas_constant / gas.molecular_weights
    gr = zeros(gas.n_species)
    griffon.species_cp(T, gr)
    return max(abs(gr - ct) / abs(gr)) < tolerance


def do_cvspec(griffon, gas, T, p, y):
    gas.TPY = T, p, y
    ct = gas.standard_cp_R * gas_constant / gas.molecular_weights - gas_constant / gas.molecular_weights
    gr = zeros(gas.n_species)
    griffon.species_cv(T, gr)
    return max(abs(gr - ct) / abs(gr)) < tolerance


def do_hmix(griffon, gas, T, p, y):
    gas.TPY = T, p, y
    ct = gas.enthalpy_mass
    gr = griffon.enthalpy_mix(T, y)
    return abs(abs(gr - ct) / (abs(gr) + 1.e0)) < tolerance


def do_emix(griffon, gas, T, p, y):
    gas.TPY = T, p, y
    ct = gas.int_energy_mass
    gr = griffon.energy_mix(T, y)
    return abs(abs(gr - ct) / (abs(gr) + 1.e0)) < tolerance


def do_hspec(griffon, gas, T, p, y):
    gas.TPY = T, p, y
    ct = gas.standard_enthalpies_RT * gas.T * gas_constant / gas.molecular_weights
    gr = zeros(gas.n_species)
    griffon.species_enthalpies(T, gr)
    return max(abs(gr - ct) / (abs(gr) + 1.e0)) < tolerance


def do_espec(griffon, gas, T, p, y):
    gas.TPY = T, p, y
    ct = gas.standard_int_energies_RT * gas.T * gas_constant / gas.molecular_weights
    gr = zeros(gas.n_species)
    griffon.species_energies(T, gr)
    return max(abs(gr - ct) / (abs(gr) + 1.e0)) < tolerance


quantity_test_dict = {'mixture molecular weight': do_mmw,
                      'density': do_density,
                      'pressure': do_pressure,
                      'cp mix': do_cpmix,
                      'cv mix': do_cvmix,
                      'cp species': do_cpspec,
                      'cv species': do_cvspec,
                      'enthalpy mix': do_hmix,
                      'energy mix': do_emix,
                      'enthalpy species': do_hspec,
                      'energy species': do_espec}
temperature_dict = {'low': 300., 'high': 1200.}


def validate_on_mechanism(mech, temperature, quantity):
    r = mech.griffon
    gas = mech.gas
    p = 101325.
    gas.TPY = temperature, p, ones(gas.n_species)
    y = gas.Y
    return quantity_test_dict[quantity](r, gas, temperature, p, y)


def create_test(m, T, quantity):
    def test(self):
        self.assertTrue(validate_on_mechanism(m, T, quantity))

    return test


class Accuracy(unittest.TestCase):
    pass


for mech in mechs:
    for quantity in quantity_test_dict.keys():
        for temperature in temperature_dict:
            testname = 'test_' + quantity.replace(' ', '-') + '_' + mech[0] + '_' + temperature + 'T'
            setattr(Accuracy, testname, create_test(mech[1], temperature_dict[temperature], quantity))

if __name__ == '__main__':
    unittest.main()
