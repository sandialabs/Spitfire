"""
This module facilitates loading chemical reaction mechanisms in Cantera format and mixing streams in useful ways.
"""

# Spitfire - a Python-C++ library for building tabulated chemistry models and solving differential equations                    
# Copyright 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS). 
#                       
# You should have received a copy of the 3-clause BSD License                                        
# along with this program.  If not, see <https://opensource.org/licenses/BSD-3-Clause>.   
#                    
# Questions? Contact Mike Hansen (mahanse@sandia.gov)    

import cantera as ct
from numpy import sum
from spitfire.griffon.griffon import PyCombustionKernels


class CanteraWrapper(object):
    def __init__(self, mech_xml_path, group_name, solution=None):
        self._mech_xml_path = mech_xml_path if mech_xml_path is not None else 'cantera-XML-not-given'
        self._group_name = group_name if group_name is not None else 'cantera-group-not-given'
        if solution is None:
            try:
                self._solution = ct.Solution(self._mech_xml_path, self._group_name)
            except Exception as error:
                raise ValueError(
                    'Cantera failed to build a Solution object for the given XML and group name. Error: ' + str(error))
        else:
            self._solution = solution

    @property
    def mech_xml_path(self):
        return self._mech_xml_path

    @property
    def group_name(self):
        return self._group_name

    @property
    def solution(self):
        return self._solution


class ChemicalMechanismSpec(object):
    """A class that loads chemical mechanisms and mixes streams.

    This class facilitates some simple way of specifying the fuel and oxidizer streams for flamelets
    and of blending these streams to make mixtures for zero-dimensional simulations.

    **Constructor**: specify a chemical mechanism in cantera XML format

    Parameters
    ----------
    cantera_xml : str
        a cantera XML file describing the thermochemistry and (optionally) transport properties
    group_name : str
        the phase to use (e.g. a phase with transport properties vs without, if such a split exists in the XML file)
    """

    def __init__(self, cantera_xml=None, group_name=None, cantera_solution=None):
        if cantera_solution is None:
            self._cantera_wrapper = CanteraWrapper(cantera_xml, group_name)
        else:
            self._cantera_wrapper = CanteraWrapper(None, None, cantera_solution)

        self._griffon = PyCombustionKernels()
        self._populate_griffon_mechanism_data(*self._extract_cantera_mechanism_data(self._cantera_wrapper.solution))

    @classmethod
    def from_solution(cls, solution: ct.Solution):
        return ChemicalMechanismSpec(cantera_solution=solution)

    def _populate_griffon_mechanism_data(self,
                                         elem_list,
                                         ref_temperature,
                                         ref_pressure,
                                         spec_name_list,
                                         spec_dict,
                                         reac_list):
        self._griffon.mechanism_set_ref_pressure(ref_pressure)
        self._griffon.mechanism_set_ref_temperature(ref_temperature)

        for e in elem_list:
            self._griffon.mechanism_add_element(e)

        for s in spec_name_list:
            self._griffon.mechanism_add_species(s, spec_dict[s]['atoms'])

        self._griffon.mechanism_resize_heat_capacity_data()

        for s in spec_dict:
            cp = spec_dict[s]['heat-capacity']
            if cp['type'] == 'constant':
                self._griffon.mechanism_add_const_cp(s, cp['Tmin'], cp['Tmax'], cp['T0'], cp['h0'], cp['s0'], cp['cp'])
            elif cp['type'] == 'NASA7':
                self._griffon.mechanism_add_nasa7_cp(s, cp['Tmin'], cp['Tmid'], cp['Tmax'], cp['low-coeffs'].tolist(),
                                                     cp['high-coeffs'].tolist())

        add_smpl = self._griffon.mechanism_add_reaction_simple
        add_3bdy = self._griffon.mechanism_add_reaction_three_body
        add_Lind = self._griffon.mechanism_add_reaction_Lindemann
        add_Troe = self._griffon.mechanism_add_reaction_Troe
        add_smpl_ord = self._griffon.mechanism_add_reaction_simple_with_special_orders
        add_3bdy_ord = self._griffon.mechanism_add_reaction_three_body_with_special_orders
        add_Lind_ord = self._griffon.mechanism_add_reaction_Lindemann_with_special_orders
        add_Troe_ord = self._griffon.mechanism_add_reaction_Troe_with_special_orders

        for i, rx in enumerate(reac_list):
            if rx['type'] == 'simple':
                if 'orders' in rx:
                    add_smpl_ord(rx['reactants'], rx['products'], rx['reversible'],
                                 rx['A'], rx['b'], rx['Ea'] / ct.gas_constant, rx['orders'])
                else:
                    add_smpl(rx['reactants'], rx['products'], rx['reversible'],
                             rx['A'], rx['b'], rx['Ea'] / ct.gas_constant)
            elif rx['type'] == 'three-body':
                if 'orders' in rx:
                    add_3bdy_ord(rx['reactants'], rx['products'], rx['reversible'],
                                 rx['A'], rx['b'], rx['Ea'] / ct.gas_constant,
                                 rx['efficiencies'], rx['default-eff'], rx['orders'])
                else:
                    add_3bdy(rx['reactants'], rx['products'], rx['reversible'],
                             rx['A'], rx['b'], rx['Ea'] / ct.gas_constant,
                             rx['efficiencies'], rx['default-eff'])
            elif rx['type'] == 'Lindemann':
                if 'orders' in rx:
                    add_Lind_ord(rx['reactants'], rx['products'], rx['reversible'],
                                 rx['fwd-A'], rx['fwd-b'], rx['fwd-Ea'] / ct.gas_constant,
                                 rx['efficiencies'], rx['default-eff'],
                                 rx['flf-A'], rx['flf-b'], rx['flf-Ea'] / ct.gas_constant,
                                 rx['orders'])
                else:
                    add_Lind(rx['reactants'], rx['products'], rx['reversible'],
                             rx['fwd-A'], rx['fwd-b'], rx['fwd-Ea'] / ct.gas_constant,
                             rx['efficiencies'], rx['default-eff'],
                             rx['flf-A'], rx['flf-b'], rx['flf-Ea'] / ct.gas_constant)
            elif rx['type'] == 'Troe':
                if 'orders' in rx:
                    add_Troe_ord(rx['reactants'], rx['products'], rx['reversible'],
                                 rx['fwd-A'], rx['fwd-b'], rx['fwd-Ea'] / ct.gas_constant,
                                 rx['efficiencies'], rx['default-eff'],
                                 rx['flf-A'], rx['flf-b'], rx['flf-Ea'] / ct.gas_constant,
                                 rx['Troe-params'].tolist(),
                                 rx['orders'])
                else:
                    add_Troe(rx['reactants'], rx['products'], rx['reversible'],
                             rx['fwd-A'], rx['fwd-b'], rx['fwd-Ea'] / ct.gas_constant,
                             rx['efficiencies'], rx['default-eff'],
                             rx['flf-A'], rx['flf-b'], rx['flf-Ea'] / ct.gas_constant,
                             rx['Troe-params'].tolist())

    @classmethod
    def _extract_cantera_mechanism_data(cls, ctsol: ct.Solution):
        elem_list = ctsol.element_names
        ref_temperature = 298.15
        ref_pressure = ctsol.reference_pressure
        spec_name_list = list()
        spec_dict = dict()
        reac_temporary_list = list()
        # todo: add error checking
        for i in range(ctsol.n_species):
            sp = ctsol.species(i)

            spec_name_list.append(sp.name)

            if isinstance(sp.thermo, ct.ConstantCp):
                spec_dict[sp.name] = dict({'atoms': sp.composition,
                                           'heat-capacity': dict({
                                               'type': 'constant',
                                               'Tmin': sp.thermo.min_temp,
                                               'Tmax': sp.thermo.max_temp,
                                               'T0': sp.thermo.coeffs[0],
                                               'h0': sp.thermo.coeffs[1],
                                               's0': sp.thermo.coeffs[2],
                                               'cp': sp.thermo.coeffs[3]})})
            elif isinstance(sp.thermo, ct.NasaPoly2):
                spec_dict[sp.name] = dict({'atoms': sp.composition,
                                           'heat-capacity': dict({
                                               'type': 'NASA7',
                                               'Tmin': sp.thermo.min_temp,
                                               'Tmid': sp.thermo.coeffs[0],
                                               'Tmax': sp.thermo.max_temp,
                                               'low-coeffs': sp.thermo.coeffs[8:],
                                               'high-coeffs': sp.thermo.coeffs[1:8]})})

        for i in range(ctsol.n_reactions):
            rx = ctsol.reaction(i)
            if isinstance(rx, ct.FalloffReaction):
                f = rx.falloff
                if isinstance(f, ct.TroeFalloff):
                    reac_temporary_list.append((3, dict({'type': 'Troe',
                                                         'reversible': rx.reversible,
                                                         'reactants': rx.reactants,
                                                         'products': rx.products,
                                                         'default-eff': rx.default_efficiency,
                                                         'efficiencies': rx.efficiencies,
                                                         'fwd-A': rx.high_rate.pre_exponential_factor,
                                                         'fwd-b': rx.high_rate.temperature_exponent,
                                                         'fwd-Ea': rx.high_rate.activation_energy,
                                                         'flf-A': rx.low_rate.pre_exponential_factor,
                                                         'flf-b': rx.low_rate.temperature_exponent,
                                                         'flf-Ea': rx.low_rate.activation_energy,
                                                         'Troe-params': rx.falloff.parameters})))
                    if rx.orders:
                        reac_temporary_list[-1][1]['orders'] = rx.orders
                else:
                    reac_temporary_list.append((2, dict({'type': 'Lindemann',
                                                         'reversible': rx.reversible,
                                                         'reactants': rx.reactants,
                                                         'products': rx.products,
                                                         'default-eff': rx.default_efficiency,
                                                         'efficiencies': rx.efficiencies,
                                                         'fwd-A': rx.high_rate.pre_exponential_factor,
                                                         'fwd-b': rx.high_rate.temperature_exponent,
                                                         'fwd-Ea': rx.high_rate.activation_energy,
                                                         'flf-A': rx.low_rate.pre_exponential_factor,
                                                         'flf-b': rx.low_rate.temperature_exponent,
                                                         'flf-Ea': rx.low_rate.activation_energy})))
                    if rx.orders:
                        reac_temporary_list[-1][1]['orders'] = rx.orders
            elif isinstance(rx, ct.ThreeBodyReaction):
                reac_temporary_list.append((1, dict({'type': 'three-body',
                                                     'reversible': rx.reversible,
                                                     'reactants': rx.reactants,
                                                     'products': rx.products,
                                                     'default-eff': rx.default_efficiency,
                                                     'efficiencies': rx.efficiencies,
                                                     'A': rx.rate.pre_exponential_factor,
                                                     'b': rx.rate.temperature_exponent,
                                                     'Ea': rx.rate.activation_energy})))
                if rx.orders:
                    reac_temporary_list[-1][1]['orders'] = rx.orders
            elif isinstance(rx, ct.ElementaryReaction):
                reac_temporary_list.append((0, dict({'type': 'simple',
                                                     'reversible': rx.reversible,
                                                     'reactants': rx.reactants,
                                                     'products': rx.products,
                                                     'A': rx.rate.pre_exponential_factor,
                                                     'b': rx.rate.temperature_exponent,
                                                     'Ea': rx.rate.activation_energy})))
                if rx.orders:
                    reac_temporary_list[-1][1]['orders'] = rx.orders
        reac_list = [y[1] for y in sorted(reac_temporary_list, key=lambda x: x[0])]
        return elem_list, ref_temperature, ref_pressure, spec_name_list, spec_dict, reac_list

    @property
    def gas(self):
        """Obtain this mechanism's cantera Solution object"""
        return self._cantera_wrapper.solution

    @property
    def griffon(self):
        """Obtain this mechanism's griffon PyCombustionKernels object"""
        return self._griffon

    @property
    def mech_xml_path(self):
        """Obtain the path of the identified mechanism's XML specification"""
        return self._cantera_wrapper.mech_xml_path

    @property
    def group_name(self):
        """Obtain the group name of the identified mechanism's XML specification"""
        return self._cantera_wrapper.group_name

    @property
    def n_species(self):
        """Obtain the number of species in this chemical mechanism"""
        return self._cantera_wrapper.solution.n_species

    @property
    def n_reactions(self):
        """Obtain the number of reactions in this chemical mechanism"""
        return self._cantera_wrapper.solution.n_reactions

    @property
    def species_names(self):
        """Obtain the names of speciesin this chemical mechanism"""
        return self._cantera_wrapper.solution.species_names

    def species_index(self, name):
        """Obtain the index of a particular species"""
        return self._cantera_wrapper.solution.species_index(name)

    def molecular_weight(self, id):
        if isinstance(id, str):
            return self._cantera_wrapper.solution.molecular_weights[self._cantera_wrapper.solution.species_index(id)]
        elif isinstance(id, int):
            return self._cantera_wrapper.solution.molecular_weights[id]
        else:
            raise TypeError('ChemicalMechanismSpec.molecular_weight(id) takes a string or integer, given ' + str(id))

    def stream(self, properties=None, values=None, stp_air=False):
        """
        Build a mixture of species with certain properties

        :param properties: a string of keys used in building a cantera Quantity (e.g., 'TPX' or 'TP' or 'X', etc.)
        :param values: the values of the properties
        :param stp_air: special option to make a stream of air at standard temperature and pressure
            This produces a stream of 3.74 mol N2 per mole O2 at 300 K and one atmosphere
        :return: a cantera Quantity object with the specified properties
        """
        q = ct.Quantity(self._cantera_wrapper.solution)
        if stp_air:
            if properties is not None or values is not None:
                print('Warning in building a stream of air at standard conditions!'
                      'The properties and values arguments will be ignored because stp_air=True was set.')
            q.TPX = 300., 101325., 'o2:1 n2:3.74'
        else:
            if properties is None:
                raise ValueError('ChemicalMechanismSpec.stream() was called improperly.\n'
                                 'There are two ways to build streams:\n'
                                 ' 1)  stream(stp_air=True)\n'
                                 ' 2)  stream(properties, values), e.g. stream(\'X\', \'O2:1, N2:1\')\n'
                                 '                                   '
                                 'or stream(\'TPY\', (300., 101325., \'O2:1, N2:1\'))\n')
            else:
                if properties is not None and values is None:
                    raise ValueError('ChemicalMechanismSpec.stream() expects two arguments '
                                     'if properties are set in the construction')
                setattr(q, properties, values)
        return q

    def copy_stream(self, stream):
        """Make a duplicate of a stream - use this to avoid inadvertently modifying a stream by reference."""
        q = ct.Quantity(self._cantera_wrapper.solution)
        q.TPX = stream.TPX
        return q

    def copy_streams(self, streams):
        """Make a copy of a list of streams, returning a list with copies of each stream."""
        qlist = []
        for stream in streams:
            qlist.append(self.copy_stream(stream))
        return qlist

    @staticmethod
    def mix_streams(streams, basis, constant='HP'):
        """
        Mix a number of streams by mass/mole and at constant HP, TP, UV, etc. (as supported by Cantera)

        :param streams: a list of tuples as [(stream, amount)] where amount is the mass/moles (depending on the basis)
        :param basis: whether amounts are masses or moles
        :param constant: property pair held constant, such as HP, TP, UV - any combination supported by Cantera
        :return: the requested mixture
        """
        q_list = []
        for stream, amount in streams:
            if basis == 'mass':
                stream.mass = amount
            elif basis == 'mole':
                stream.moles = amount
            stream.constant = constant
            q_list.append(stream)
        mix = sum(q_list)
        return mix

    def mix_for_equivalence_ratio(self, phi, fuel, oxy):
        """Mix a stream of fuel and oxidizer such that the mixture has a specified equivalence ratio."""
        self._cantera_wrapper.solution.set_equivalence_ratio(phi, fuel.X, oxy.X)
        return ct.Quantity(self._cantera_wrapper.solution)

    def mix_for_normalized_equivalence_ratio(self, normalized_phi, fuel, oxy):
        """Mix a stream of fuel and oxidizer such that the mixture has a specified normalized equivalence ratio."""
        return self.mix_for_equivalence_ratio(normalized_phi / (1. - normalized_phi), fuel, oxy)

    def _get_atoms_in_stream(self, stream, atom_names):
        atom_amounts = dict()
        for atom in atom_names:
            atom_amounts[atom] = 0
        for i, species in enumerate(stream.species_names):
            mole_fraction = stream.X[i]
            for atom in atom_names:
                atom_amounts[atom] += mole_fraction * stream.n_atoms(species, atom)
        return atom_amounts

    def stoich_molar_fuel_to_oxy_ratio(self, fuel_stream, oxy_stream):
        """Get the molar ratio of fuel to oxidizer at stoichiometric conditions.
            Assumes C, O, and H combustion of single fuel and single oxidizer streams."""
        atom_names = ['H', 'O']
        if 'C' in self._cantera_wrapper.solution.element_names:
            atom_names.append('C')
        fuel_atoms = self._get_atoms_in_stream(fuel_stream, atom_names)
        oxy_atoms = self._get_atoms_in_stream(oxy_stream, atom_names)
        if 'C' not in self._cantera_wrapper.solution.element_names:
            fuel_atoms['C'] = 0
            oxy_atoms['C'] = 0
        return -(oxy_atoms['O'] - 0.5 * oxy_atoms['H'] - 2.0 * oxy_atoms['C']) / (
                fuel_atoms['O'] - 0.5 * fuel_atoms['H'] - 2.0 * fuel_atoms['C'])

    def stoich_mass_fuel_to_oxy_ratio(self, fuel_stream, oxy_stream):
        """Get the mass ratio of fuel to oxidizer at stoichiometric conditions.
            Assumes C, O, and H combustion of single fuel and single oxidizer streams."""
        mf = fuel_stream.mean_molecular_weight
        mx = oxy_stream.mean_molecular_weight
        return mf / mx * self.stoich_molar_fuel_to_oxy_ratio(fuel_stream, oxy_stream)

    def stoich_mixture_fraction(self, fuel_stream, oxy_stream):
        """Get the mixture fraction at stoichiometric conditions.
            Assumes C, O, and H combustion of single fuel and single oxidizer streams."""
        eta = self.stoich_mass_fuel_to_oxy_ratio(fuel_stream, oxy_stream)
        return eta / (1. + eta)

    def mix_fuels_for_stoich_mixture_fraction(self, fuel1, fuel2, zstoich, oxy):
        """Mix two fuel streams for a specified stoichiometric mixture fraction given a particular oxidizer.
            As an example, this can be used to dilute a fuel with Nitrogen to hit a particular stoichiometric mixture fraction.

            Note that it is not possible to reach all stoichiometric mixture fractions with any fuel combinations!
            In such a case this function will throw an error."""
        atom_names = ['H', 'O']
        if 'C' in self._cantera_wrapper.solution.element_names:
            atom_names.append('C')
        coeffs_a = self._get_atoms_in_stream(fuel1, atom_names)
        coeffs_b = self._get_atoms_in_stream(fuel2, atom_names)
        coeffs_x = self._get_atoms_in_stream(oxy, atom_names)
        if 'C' not in self._cantera_wrapper.solution.element_names:
            coeffs_a['C'] = 0
            coeffs_b['C'] = 0
            coeffs_x['C'] = 0

        mfa = fuel1.mean_molecular_weight
        mfb = fuel2.mean_molecular_weight
        dmf = mfa - mfb
        mx = oxy.mean_molecular_weight
        etax = coeffs_x['O'] - 0.5 * coeffs_x['H'] - 2. * coeffs_x['C']
        netab = -coeffs_b['O'] + 0.5 * coeffs_b['H'] + 2. * coeffs_b['C']
        netaa = -coeffs_a['O'] + 0.5 * coeffs_a['H'] + 2. * coeffs_a['C']
        lhs = mx / etax * zstoich / (1. - zstoich)
        gamma = netaa - netab
        kstar = (mfb - lhs * netab) / (lhs * gamma - dmf)

        if kstar > 1. or kstar < 0.:
            raise ValueError('The provided fuel combination cannot reach the desired stoichiometric mixture fraction!')
        else:
            return self.mix_streams([(fuel1, kstar), (fuel2, 1. - kstar)], 'mole')
