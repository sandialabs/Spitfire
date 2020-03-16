from spitfire.chemistry.mechanism import ChemicalMechanismSpec as Mechanism
from spitfire.chemistry.reactors import HomogeneousReactor
import matplotlib.pyplot as plt
import cantera as ct

species_in_model = ['NXC7H16', 'O2', 'H2O', 'CO2', 'N2']
reaction_cti = '''
reaction(
'2 NXC7H16 + 22 O2 => 14 CO2 + 16 H2O', 
[2e7, 0, (30.0, 'kcal/mol')], 
order='NXC7H16:0.25 O2:1.5')
'''
xml_file_for_species = 'heptane-liu-hewson-chen-pitsch-highT.xml'

species_data = ct.Species.listFromFile(xml_file_for_species)

species_list = list()
for sp in species_data:
    if sp.name in species_in_model:
        species_list.append(sp)

s = ct.Solution(thermo='IdealGas',
                kinetics='GasKinetics',
                species=species_list,
                reactions=[ct.Reaction.fromCti(reaction_cti)])

mech = Mechanism.from_solution(s)

fuel = mech.stream('X', 'NXC7H16:1')
air = mech.stream(stp_air=True)

mix = mech.mix_for_equivalence_ratio(phi=1., fuel=fuel, oxy=air)
mix.TP = 1000, 101325

r = HomogeneousReactor(mech_spec=mech,
                       initial_mixture=mix,
                       configuration='isobaric',
                       heat_transfer='adiabatic',
                       mass_transfer='closed')
output = r.integrate_to_time(0.1)

t = output.time_values * 1.e3
T = output['temperature']
yH = output['mass fraction NXC7H16']
yO = output['mass fraction O2']
yC = output['mass fraction CO2']
yW = output['mass fraction H2O']

fig, axY = plt.subplots()

axY.plot(t, yH, label='$\\mathrm{C}_7\\mathrm{H}_{16}$')
axY.plot(t, yO, label='$\\mathrm{O}_2$')
axY.plot(t, yC, label='$\\mathrm{CO}_2$')
axY.plot(t, yW, label='$\\mathrm{H}_2\\mathrm{O}$')
axY.legend(loc='best')
axY.set_ylabel('mass fraction')
axY.set_xlabel('t (ms)')

axT = axY.twinx()
axT.plot(t, T, '--', label='temperature')
axT.set_ylabel('T (K)')
axT.legend(loc='center right')

fig.tight_layout()
# plt.savefig('one_step_ignition.png', dpi=300)
plt.show()
