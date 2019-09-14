from spitfire.chemistry.mechanism import ChemicalMechanismSpec
from spitfire.chemistry.reactors import HomogeneousReactor
import matplotlib.pyplot as plt
from numpy import linspace, zeros_like
from cantera import one_atm
from time import perf_counter as timer

b = ChemicalMechanismSpec(cantera_xml='dme-bhagatwala.xml', group_name='dme-bhagatwala')

air = b.stream(stp_air=True)
h2 = b.stream('X', 'CH3OCH3:1, CH4:1')
phi = 1.0
blend = b.mix_for_equivalence_ratio(phi, h2, air)

temperature_list = linspace(600., 1400., 20)
pressure_atm_list = [4., 10., 20., 50., 100., 200.]
markers_list = ['o', 's', '^', 'D', 'P', '*']

t0 = timer()
for pressure, marker in zip(pressure_atm_list, markers_list):
    tau_list = zeros_like(temperature_list)

    print(f'  p = {pressure:7.2f} atm | T = ', end='', flush=True)
    for idx, temperature in enumerate(temperature_list):
        mix = b.copy_stream(blend)
        mix.TP = temperature, pressure * one_atm

        r = HomogeneousReactor(b, mix,
                               'isobaric',
                               'adiabatic',
                               'closed')
        tau_list[idx] = r.compute_ignition_delay(first_time_step=1.e-9)
        print(f'{temperature:.1f} ', end='', flush=True)

    plt.semilogy(temperature_list, tau_list * 1.e6, '-' + marker, label='{:.1f} atm'.format(pressure))

    print(f'complete\n' + '-' * 153)

plt.xlabel('T (K)')
plt.ylabel('ignition delay (us)')
plt.legend()
plt.grid()
plt.show()
