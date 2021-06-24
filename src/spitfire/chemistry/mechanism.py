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
from numpy import sum, array, abs
from spitfire.griffon.griffon import PyCombustionKernels


class CanteraLoadError(Exception):
    def __init__(self, mech_xml_path, group_name, error):
        super().__init__(f'Cantera failed to build a Solution object for the given XML and group name. '
                         f'XML path provided: {mech_xml_path}, '
                         f'group name provided: {group_name}, '
                         f'Error: {error}')


class _CanteraWrapper(object):
    def __init__(self, mech_xml_path, group_name, solution=None):
        self._mech_xml_path = mech_xml_path if mech_xml_path is not None else 'cantera-XML-not-given'
        self._group_name = group_name if group_name is not None else 'cantera-group-not-given'
        if solution is None:
            try:
                self._solution = ct.Solution(self._mech_xml_path, self._group_name)
            except Exception as error:
                raise CanteraLoadError(mech_xml_path, group_name, error)
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

        **Constructor**: specify a chemical mechanism file in cantera XML format

        Parameters
        ----------
        cantera_xml : str
            a cantera XML file describing the thermochemistry and (optionally) transport properties
        group_name : str
            the phase to use (e.g. a phase with transport properties vs without, if such a split exists in the XML file)
        cantera_solution: ct.Solution
            a ct.Solution object to use directly (optional, if specified the xml and group name are ignored if given)

    """

    def __init__(self, cantera_xml=None, group_name=None, cantera_solution=None):
        if cantera_solution is None:
            self._cantera_wrapper = _CanteraWrapper(cantera_xml, group_name)
        else:
            self._cantera_wrapper = _CanteraWrapper(None, None, cantera_solution)

        self._mech_data = dict()
        self._mech_data['ref_pressure'] = None
        self._mech_data['ref_temperature'] = None
        self._mech_data['elements'] = list()
        self._mech_data['species'] = dict()
        self._mech_data['reactions'] = list()

        self._griffon = PyCombustionKernels()
        self._populate_griffon_mechanism_data(*self._extract_cantera_mechanism_data(self._cantera_wrapper.solution))

    @property
    def mech_data(self):
        return self._mech_data

    def __getstate__(self):
        return dict({'mech_data': self._mech_data})

    def __setstate__(self, state):
        mech_data = state['mech_data']
        self.__init__(cantera_solution=ChemicalMechanismSpec._build_cantera_solution(mech_data))

    @classmethod
    def from_solution(cls, solution: ct.Solution):
        """Construct a ChemicalMechanismSpec directly from a cantera solution"""
        return ChemicalMechanismSpec(cantera_solution=solution)

    def _populate_griffon_mechanism_data(self,
                                         ct_element_mw_map,
                                         elem_list,
                                         ref_temperature,
                                         ref_pressure,
                                         spec_name_list,
                                         spec_dict,
                                         reac_list):
        self._griffon.mechanism_set_ref_pressure(ref_pressure)
        self._mech_data['ref_pressure'] = ref_pressure
        self._griffon.mechanism_set_ref_temperature(ref_temperature)
        self._mech_data['ref_temperature'] = ref_temperature
        gas_constant = ct.gas_constant
        self._griffon.mechanism_set_gas_constant(gas_constant)
        self._mech_data['gas_constant'] = gas_constant

        self._griffon.mechanism_set_element_mw_map(ct_element_mw_map)
        self._mech_data['element_mw_map'] = ct_element_mw_map

        for e in elem_list:
            self._griffon.mechanism_add_element(e)
            self._mech_data['elements'].append(e)

        for s in spec_name_list:
            self._griffon.mechanism_add_species(s, spec_dict[s]['atoms'])
            self._mech_data['species'][s] = dict()
            self._mech_data['species'][s]['atom_map'] = spec_dict[s]['atoms']

        self._griffon.mechanism_resize_heat_capacity_data()

        for s in spec_dict:
            cp = spec_dict[s]['heat-capacity']
            if cp['type'] == 'constant':
                self._griffon.mechanism_add_const_cp(s, cp['Tmin'], cp['Tmax'], cp['T0'], cp['h0'], cp['s0'], cp['cp'])
                self._mech_data['species'][s]['cp'] = (
                    'constant', cp['Tmin'], cp['Tmax'], cp['T0'], cp['h0'], cp['s0'], cp['cp'])
            elif cp['type'] == 'NASA7':
                self._griffon.mechanism_add_nasa7_cp(s, cp['Tmin'], cp['Tmid'], cp['Tmax'], cp['low-coeffs'].tolist(),
                                                     cp['high-coeffs'].tolist())
                self._mech_data['species'][s]['cp'] = (
                    'NASA7', cp['Tmin'], cp['Tmid'], cp['Tmax'], cp['low-coeffs'].tolist(), cp['high-coeffs'].tolist())

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
                    self._mech_data['reactions'].append(('simple-special', rx['reactants'], rx['products'],
                                                         rx['reversible'], rx['A'], rx['b'], rx['Ea'],
                                                         rx['orders']))
                else:
                    add_smpl(rx['reactants'], rx['products'], rx['reversible'],
                             rx['A'], rx['b'], rx['Ea'] / ct.gas_constant)
                    self._mech_data['reactions'].append(('simple', rx['reactants'], rx['products'], rx['reversible'],
                                                         rx['A'], rx['b'], rx['Ea']))
            elif rx['type'] == 'three-body':
                if 'orders' in rx:
                    add_3bdy_ord(rx['reactants'], rx['products'], rx['reversible'],
                                 rx['A'], rx['b'], rx['Ea'] / ct.gas_constant,
                                 rx['efficiencies'], rx['default-eff'], rx['orders'])
                    self._mech_data['reactions'].append(('three-body-special', rx['reactants'], rx['products'],
                                                         rx['reversible'],
                                                         rx['A'], rx['b'], rx['Ea'],
                                                         rx['efficiencies'], rx['default-eff'], rx['orders']))
                else:
                    add_3bdy(rx['reactants'], rx['products'], rx['reversible'],
                             rx['A'], rx['b'], rx['Ea'] / ct.gas_constant,
                             rx['efficiencies'], rx['default-eff'])
                    self._mech_data['reactions'].append(
                        ('three-body', rx['reactants'], rx['products'], rx['reversible'],
                         rx['A'], rx['b'], rx['Ea'],
                         rx['efficiencies'], rx['default-eff']))
            elif rx['type'] == 'Lindemann':
                if 'orders' in rx:
                    add_Lind_ord(rx['reactants'], rx['products'], rx['reversible'],
                                 rx['fwd-A'], rx['fwd-b'], rx['fwd-Ea'] / ct.gas_constant,
                                 rx['efficiencies'], rx['default-eff'],
                                 rx['flf-A'], rx['flf-b'], rx['flf-Ea'] / ct.gas_constant,
                                 rx['orders'])
                    self._mech_data['reactions'].append(('Lindemann-special', rx['reactants'], rx['products'],
                                                         rx['reversible'],
                                                         rx['fwd-A'], rx['fwd-b'], rx['fwd-Ea'],
                                                         rx['efficiencies'], rx['default-eff'],
                                                         rx['flf-A'], rx['flf-b'], rx['flf-Ea'],
                                                         rx['orders']))
                else:
                    add_Lind(rx['reactants'], rx['products'], rx['reversible'],
                             rx['fwd-A'], rx['fwd-b'], rx['fwd-Ea'] / ct.gas_constant,
                             rx['efficiencies'], rx['default-eff'],
                             rx['flf-A'], rx['flf-b'], rx['flf-Ea'] / ct.gas_constant)
                    self._mech_data['reactions'].append(('Lindemann', rx['reactants'], rx['products'], rx['reversible'],
                                                         rx['fwd-A'], rx['fwd-b'], rx['fwd-Ea'],
                                                         rx['efficiencies'], rx['default-eff'],
                                                         rx['flf-A'], rx['flf-b'], rx['flf-Ea']))
            elif rx['type'] == 'Troe':
                if 'orders' in rx:
                    add_Troe_ord(rx['reactants'], rx['products'], rx['reversible'],
                                 rx['fwd-A'], rx['fwd-b'], rx['fwd-Ea'] / ct.gas_constant,
                                 rx['efficiencies'], rx['default-eff'],
                                 rx['flf-A'], rx['flf-b'], rx['flf-Ea'] / ct.gas_constant,
                                 rx['Troe-params'].tolist(),
                                 rx['orders'])
                    self._mech_data['reactions'].append(('Troe-special', rx['reactants'], rx['products'],
                                                         rx['reversible'],
                                                         rx['fwd-A'], rx['fwd-b'], rx['fwd-Ea'],
                                                         rx['efficiencies'], rx['default-eff'],
                                                         rx['flf-A'], rx['flf-b'], rx['flf-Ea'],
                                                         rx['Troe-params'].tolist(),
                                                         rx['orders']))
                else:
                    add_Troe(rx['reactants'], rx['products'], rx['reversible'],
                             rx['fwd-A'], rx['fwd-b'], rx['fwd-Ea'] / ct.gas_constant,
                             rx['efficiencies'], rx['default-eff'],
                             rx['flf-A'], rx['flf-b'], rx['flf-Ea'] / ct.gas_constant,
                             rx['Troe-params'].tolist())
                    self._mech_data['reactions'].append(('Troe', rx['reactants'], rx['products'], rx['reversible'],
                                                         rx['fwd-A'], rx['fwd-b'], rx['fwd-Ea'],
                                                         rx['efficiencies'], rx['default-eff'],
                                                         rx['flf-A'], rx['flf-b'], rx['flf-Ea'],
                                                         rx['Troe-params'].tolist()))

    @classmethod
    def _build_cantera_solution(cls, gdata):
        p_ref = gdata['ref_pressure']
        species_list = list()
        for s in gdata['species']:
            spec = gdata['species'][s]
            ctspec = ct.Species(s, ','.join([f'{k}:{spec["atom_map"][k]}' for k in spec['atom_map']]))
            if spec['cp'][0] == 'constant':
                Tmin, Tmax, T0, h0, s0, cp = spec['cp'][1:]
                ctspec.thermo = ct.ConstantCp(Tmin, Tmax, p_ref, list([T0, h0, s0, cp]))
            elif spec['cp'][0] == 'NASA7':
                Tmin, Tmid, Tmax, low_coeffs, high_coeffs = spec['cp'][1:]
                coeffs = [Tmid] + high_coeffs + low_coeffs
                ctspec.thermo = ct.NasaPoly2(Tmin, Tmax, p_ref, coeffs)
            species_list.append(ctspec)

        reaction_list = list()
        for rxn in gdata['reactions']:
            type, reactants_stoich, products_stoich, reversible = rxn[:4]

            fwd_pre_exp_value, fwd_temp_exponent, fwd_act_energy = rxn[4:7]
            fwd_rate = ct.Arrhenius(fwd_pre_exp_value, fwd_temp_exponent, fwd_act_energy)

            if type == 'simple' or type == 'simple-special':
                ctrxn = ct.ElementaryReaction(reactants_stoich, products_stoich)
                ctrxn.rate = fwd_rate
            elif type == 'three-body' or type == 'three-body-special':
                ctrxn = ct.ThreeBodyReaction(reactants_stoich, products_stoich)
                ctrxn.rate = fwd_rate
                ctrxn.efficiencies = rxn[7]
                ctrxn.default_efficiency = rxn[8]
            elif type == 'Lindemann' or type == 'Lindemann-special':
                ctrxn = ct.FalloffReaction(reactants_stoich, products_stoich)
                ctrxn.efficiencies = rxn[7]
                ctrxn.default_efficiency = rxn[8]
                flf_pre_exp_value, flf_temp_exponent, flf_act_energy = rxn[9:12]
                ctrxn.high_rate = fwd_rate
                ctrxn.low_rate = ct.Arrhenius(flf_pre_exp_value, flf_temp_exponent, flf_act_energy)
                ctrxn.falloff = ct.Falloff()
            elif type == 'Troe' or type == 'Troe-special':
                ctrxn = ct.FalloffReaction(reactants_stoich, products_stoich)
                ctrxn.efficiencies = rxn[7]
                ctrxn.default_efficiency = rxn[8]
                flf_pre_exp_value, flf_temp_exponent, flf_act_energy = rxn[9:12]
                troe_params = rxn[12]
                ctrxn.high_rate = fwd_rate
                ctrxn.low_rate = ct.Arrhenius(flf_pre_exp_value, flf_temp_exponent, flf_act_energy)
                if len(troe_params) == 4 and abs(troe_params[3]) < 1e-300:
                    ctrxn.falloff = ct.TroeFalloff(troe_params[:3])
                else:
                    ctrxn.falloff = ct.TroeFalloff(troe_params)

            if 'special' in type:
                ctrxn.orders = rxn[-1]

            ctrxn.reversible = reversible
            reaction_list.append(ctrxn)

        return ct.Solution(thermo='IdealGas', kinetics='GasKinetics', species=species_list, reactions=reaction_list)

    @classmethod
    def _get_cantera_element_mw_map(cls, ctsol: ct.Solution):
        species_list = [ct.Species(element_name, f'{element_name}: 1') for element_name in ctsol.element_names]
        for i in range(len(species_list)):
            species_list[i].thermo = ct.ConstantCp(300., 3000., 101325., (300., 0., 0., 1.e4))
        element_only_ctsol = ct.Solution(thermo='IdealGas', kinetics='GasKinetics', species=species_list, reactions=[])
        return {name: mw for name, mw in zip(element_only_ctsol.species_names, element_only_ctsol.molecular_weights)}

    @classmethod
    def _extract_cantera_mechanism_data(cls, ctsol: ct.Solution):
        ct_element_mw_map = cls._get_cantera_element_mw_map(ctsol)
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
        return ct_element_mw_map, elem_list, ref_temperature, ref_pressure, spec_name_list, spec_dict, reac_list

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

    @property
    def molecular_weights(self):
        return array([self._cantera_wrapper.solution.molecular_weights[ni] for ni in range(self.n_species)])

    def molecular_weight(self, ni):
        """Obtain the molecular weight of a single species from its string name or integer index
        """
        if isinstance(ni, str):
            return self._cantera_wrapper.solution.molecular_weights[self._cantera_wrapper.solution.species_index(ni)]
        elif isinstance(ni, int):
            return self._cantera_wrapper.solution.molecular_weights[ni]
        else:
            raise TypeError('ChemicalMechanismSpec.molecular_weight(ni) takes a string or integer, given ' + str(ni))

    def stream(self, properties=None, values=None, stp_air=False):
        """Build a mixture of species with certain properties

        Parameters
        ----------
        properties : str
            a string of keys used in building a cantera Quantity (e.g., 'TPX' or 'TP' or 'X', etc.)
        values : tuple
            the values of the properties
        stp_air : bool
            special option to make a stream of air at standard temperature and pressure (default: False)
            This produces a stream of 3.74 mol N2 per mole O2 at 300 K and one atmosphere
        Returns
        -------
        mix : cantera.Quantity
            a cantera Quantity object with the specified properties

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

    @staticmethod
    def mix_streams(streams, basis, constant='HP'):
        """Mix a number of streams by mass/mole and at constant HP, TP, UV, etc. (as supported by Cantera)

        Parameters
        ----------
        streams : list(tuple)
            a list of tuples as [(stream, amount)] where amount is the mass/moles (depending on the basis)
        basis : str
            whether amounts are masses ('mass') or moles ('mole')
        constant : str
            property pair held constant, such as HP (default), TP, UV - any combination supported by Cantera
        Returns
        -------
        mix : cantera.Quantity
            the requested mixture

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
