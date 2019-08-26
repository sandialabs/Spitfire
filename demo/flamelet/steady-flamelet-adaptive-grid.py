from spitfire import ChemicalMechanismSpec, Flamelet
import numpy as np
import matplotlib.pyplot as plt

m = ChemicalMechanismSpec('hydrogen', 'burke')

pressure = 101325.

air = m.stream(stp_air=True)
air.TP = 300., pressure

zstoich = 0.2
fuel = m.mix_fuels_for_stoich_mixture_fraction(m.stream('X', 'H2:1'), m.stream('X', 'N2:1'), zstoich, air)
fuel.TP = 1200., pressure

npts_interior = 128

chimax_vec = np.hstack([0., np.logspace(-1, 5, 15)])

flamelet_specs = {'mech_spec': m,
                  'pressure': pressure,
                  'oxy_stream': air,
                  'fuel_stream': fuel,
                  'engine': 'griffon'}

grid_has_been_refined = False
plt.ion()
for idx, chimax in enumerate(chimax_vec):
    print('chimax:', chimax, 'Hz')
    if idx == 0:
        flamelet_specs['grid_points'] = npts_interior + 2
        flamelet_specs['grid_type'] = 'uniform'
        flamelet_specs['initial_condition'] = 'equilibrium'
    else:
        if grid_has_been_refined:
            flamelet_specs['initial_condition'] = running_state
            flamelet_specs['grid'] = new_grid
        else:
            flamelet_specs['initial_condition'] = running_state
            flamelet_specs['grid_points'] = npts_interior + 2
            flamelet_specs['grid_type'] = 'uniform'

    flamelet_specs['max_dissipation_rate'] = chimax
    flamelet = Flamelet(**flamelet_specs)
    flamelet.steady_state_solve_psitc(plot=False, log_rate=200)
    running_state = np.copy(flamelet.final_state)

    flamelet.insitu_process_quantity(['temperature', 'mass fractions', 'production rates', 'heat release rate'])
    running_state_data = flamelet.process_quantities_on_state(running_state)
    running_temperature = running_state_data['temperature']
    if idx > 0:
        plt.plot(flamelet.mixfrac_grid, np.hstack((air.T, running_temperature.ravel(), fuel.T)),
                 'o-', markersize=3, markerfacecolor='w')
        plt.pause(1.e-3)

    if idx == 10:
        new_grid, running_state = flamelet.redistribute_grid(running_state)
        del flamelet_specs['grid_type']
        del flamelet_specs['grid_points']
        flamelet_specs['initial_condition'] = running_state
        flamelet_specs['grid'] = new_grid
        flamelet = Flamelet(**flamelet_specs)
        print(' - re-converging on new grid')
        flamelet.steady_state_solve_psitc(plot=False, log_rate=200)
        running_state = np.copy(flamelet.final_state)
        flamelet.insitu_process_quantity(['temperature', 'mass fractions', 'production rates', 'heat release rate'])
        running_state_data = flamelet.process_quantities_on_state(running_state)
        running_temperature = running_state_data['temperature']
        if idx > 0:
            plt.plot(flamelet.mixfrac_grid, np.hstack((air.T, running_temperature.ravel(), fuel.T)),
                     'o-', markersize=3, markerfacecolor='w')
            plt.pause(1.e-3)
        grid_has_been_refined = True
        refine_grid = False

plt.ioff()
plt.show()
