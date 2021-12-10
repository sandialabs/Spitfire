import unittest
from numpy import max, abs, ones, zeros, hstack, sqrt, finfo
from cantera import gas_constant
from spitfire import ChemicalMechanismSpec as Mechanism
import pickle
import cantera as ct

ct.suppress_thermo_warnings()  # because we are faking heat capacity polynomials

species_decl = list([ct.Species('A', 'H:2'),
                     ct.Species('B', 'H:1'),
                     ct.Species('C', 'O:1'),
                     ct.Species('D', 'O:2'),
                     ct.Species('E', 'H:1, O:2'),
                     ct.Species('N', 'H:1, O:1')])
species_dict = dict({'const': list(), 'nasa7': list(), 'nasa9-1': list(), 'nasa9-2': list(), 'nasa9-3': list()})
for i, s in enumerate(species_decl):
    species_dict['const'].append(ct.Species(s.name, s.composition))
    species_dict['const'][-1].thermo = ct.ConstantCp(300., 3000., 101325., (300., 0., 0., float(i + 1) * 1.e4))

    species_dict['nasa7'].append(ct.Species(s.name, s.composition))
    coeffs1 = [float(i + 1) * v for v in [1.e0, 1.e-2, 1.e-4, 1.e-6, 1.e-8, 1.e-10, 1.e-12]]
    coeffs2= [float(i + 2) * v for v in [1.e0, 1.e-2, 1.e-4, 1.e-6, 1.e-8, 1.e-10, 1.e-12]]
    species_dict['nasa7'][-1].thermo = ct.NasaPoly2(300., 3000., 101325., hstack([1200.] + coeffs1 + coeffs2))

    species_dict['nasa9-1'].append(ct.Species(s.name, s.composition))
    coeffs = [float(i + 1) * v for v in [1e0, 1e0, 1.e0, 1.e-2, 1.e-4, 1.e-6, 1.e-10, 1.e2, 1.e1]]
    species_dict['nasa9-1'][-1].thermo = ct.Nasa9PolyMultiTempRegion(300., 3000., 101325., hstack([1] + [300, 3000.] + coeffs))

    species_dict['nasa9-2'].append(ct.Species(s.name, s.composition))
    coeffs1 = [float(i + 1) * v for v in [1e0, 1e0, 1.e0, 1.e-2, 1.e-4, 1.e-6, 1.e-10, 1.e2, 1.e1]]
    coeffs2 = [float(i + 2) * v for v in [1e0, 1e0, 1.e0, 1.e-2, 1.e-4, 1.e-6, 1.e-10, 1.e2, 1.e1]]
    species_dict['nasa9-2'][-1].thermo = ct.Nasa9PolyMultiTempRegion(300., 3000., 101325., hstack([2] + [300, 1200.] + coeffs1 + [1200., 3000.] + coeffs2))

    species_dict['nasa9-3'].append(ct.Species(s.name, s.composition))
    coeffs1 = [float(i + 1) * v for v in [1e0, 1e0, 1.e0, 1.e-2, 1.e-4, 1.e-6, 1.e-10, 1.e2, 1.e1]]
    coeffs2 = [float(i + 2) * v for v in [1e0, 1e0, 1.e0, 1.e-2, 1.e-4, 1.e-6, 1.e-10, 1.e2, 1.e1]]
    coeffs3 = [float(i + 3) * v for v in [1e0, 1e0, 1.e0, 1.e-2, 1.e-4, 1.e-6, 1.e-10, 1.e2, 1.e1]]
    species_dict['nasa9-3'][-1].thermo = ct.Nasa9PolyMultiTempRegion(300., 3000., 101325., hstack([3] + [300, 1200.] + coeffs1 + [1200., 2000.] + coeffs2 + [2000., 3000.] + coeffs3))

mechs = [(s, Mechanism.from_solution(ct.Solution(thermo='IdealGas', kinetics='GasKinetics', species=species_dict[s], reactions=[]))) for s in species_dict]


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

def do_dcpdT(griffon, gas, T, p, y):
    dcpspecdT_gr = zeros(gas.n_species)
    griffon.dcpdT_species(T, y, dcpspecdT_gr)
    gas.TPY = T, p, y
    cpspec1_ct = gas.standard_cp_R * gas_constant / gas.molecular_weights
    dT = 1.e-2
    gas.TPY = T + dT, p, y
    cpspec2_ct = gas.standard_cp_R * gas_constant / gas.molecular_weights
    dcpspecdT_ct = (cpspec2_ct - cpspec1_ct) / dT
    this_fd_tol = 1.e-4
    return max(abs(dcpspecdT_ct - dcpspecdT_gr) / cpspec1_ct.dot(y)) < this_fd_tol

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
                      'energy species': do_espec,
                      'dcpdT': do_dcpdT}
temperature_dict = {'low': 300., 'high': 1400.}


def validate_on_mechanism(mech, temperature, quantity, serialize_mech):
    if serialize_mech:
        serialized = pickle.dumps(mech)
        mech = pickle.loads(serialized)
    r = mech.griffon
    gas = mech.gas
    p = 101325.
    gas.TPY = temperature, p, ones(gas.n_species)
    y = gas.Y
    return quantity_test_dict[quantity](r, gas, temperature, p, y)


def create_test(m, T, quantity, serialize_mech):
    def test(self):
        self.assertTrue(validate_on_mechanism(m, T, quantity, serialize_mech))

    return test


class Accuracy(unittest.TestCase):
    pass


for mech in mechs:
    for quantity in quantity_test_dict.keys():
        for temperature in temperature_dict:
            for serialize_mech in [False, True]:
                testname = 'test_' + quantity.replace(' ', '-') + '_' + mech[0] + '_' + temperature + 'T' + \
                           f'{"_serialized_mech" if serialize_mech else ""}'
                setattr(Accuracy, testname, create_test(mech[1],
                                                        temperature_dict[temperature],
                                                        quantity,
                                                        serialize_mech))

if __name__ == '__main__':
    unittest.main()
