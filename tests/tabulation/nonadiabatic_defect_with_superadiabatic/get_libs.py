import numpy as np
from spitfire import ChemicalMechanismSpec, build_nonadiabatic_defect_transient_slfm_library, build_nonadiabatic_defect_steady_slfm_library
from spitfire import build_nonadiabatic_defect_eq_library
from os.path import abspath, join


def get_flameletspecs():
    test_xml = abspath(join('tests', 'test_mechanisms', 'hydrogen_one_step.yaml'))
    mech = ChemicalMechanismSpec(cantera_input=test_xml, group_name='one_step_hydrogen')
    pressure = 101325.
    air = mech.stream(stp_air=True)
    air.TP = 1200., pressure
    fuel = mech.stream('TPY', (1200., pressure, 'H2:0.5'))
    flamelet_specs = {'mech_spec': mech, 'oxy_stream': air, 'fuel_stream': fuel, 'grid_points': 34}
    return flamelet_specs

def get_abs_relerr(pred, obs):
    err = pred - obs
    relerr = err / (np.abs(obs) + 0.01*np.max(np.abs(obs)) + np.finfo(float).eps)
    return np.abs(relerr)

def get_params():
    superad_dT = 50.
    subad_dT = 25.
    fuel_shift = 25.
    verbose = False
    return superad_dT, subad_dT, fuel_shift, verbose

def run_slfm(n_superad, n_subad, transient, num_procs=1):
    nonad_lib_fun = build_nonadiabatic_defect_transient_slfm_library if transient else build_nonadiabatic_defect_steady_slfm_library
    flamelet_specs = get_flameletspecs()
    chivals = np.array([1.e-4])
    superad_dT, subad_dT, fuel_shift, verbose = get_params()

    lib = nonad_lib_fun(flamelet_specs, verbose=verbose, num_procs=num_procs,
                        diss_rate_values=chivals,
                        n_defect_st = n_subad)
    
    lib_sup = nonad_lib_fun(flamelet_specs, verbose=verbose, num_procs=num_procs,
                            diss_rate_values=chivals,
                            n_defect_st={'n_superad':n_superad,'n_subad':n_subad-1},
                            superad_dT=superad_dT)
    
    lib_sub = nonad_lib_fun(flamelet_specs, verbose=verbose, num_procs=num_procs,
                            diss_rate_values=chivals,
                            n_defect_st=n_subad,
                            subad_dT=subad_dT)

    lib_both = nonad_lib_fun(flamelet_specs, verbose=verbose, num_procs=num_procs,
                             diss_rate_values=chivals,
                             n_defect_st={'n_superad':n_superad,'n_subad':n_subad-1},
                             superad_dT=superad_dT,
                             subad_dT=subad_dT)

    lib_shift = nonad_lib_fun(flamelet_specs, verbose=verbose, num_procs=num_procs,
                                diss_rate_values=chivals,
                                n_defect_st={'n_superad':n_superad,'n_subad':n_subad-1},
                                superad_dT={'fuel':superad_dT+fuel_shift,'oxy':superad_dT},
                                subad_dT={'fuel':subad_dT+fuel_shift,'oxy':subad_dT})

    return lib, lib_sup, lib_sub, lib_both, lib_shift

def run_eq(n_superad, n_subad):
    nonad_lib_fun = build_nonadiabatic_defect_eq_library
    flamelet_specs = get_flameletspecs()
    superad_dT, subad_dT, fuel_shift, verbose = get_params()

    lib = nonad_lib_fun(flamelet_specs, verbose=verbose,
                        n_defect_st=n_subad)

    lib_sup = nonad_lib_fun(flamelet_specs, verbose=verbose,
                            n_defect_st={'n_superad':n_superad,'n_subad':n_subad-1},
                            superad_dT=superad_dT)

    lib_sub = nonad_lib_fun(flamelet_specs, verbose=verbose,
                            n_defect_st=n_subad,
                            subad_dT=subad_dT)

    lib_both = nonad_lib_fun(flamelet_specs, verbose=verbose,
                             n_defect_st={'n_superad':n_superad,'n_subad':n_subad-1},
                             superad_dT=superad_dT,
                             subad_dT=subad_dT)
    
    lib_shift = nonad_lib_fun(flamelet_specs, verbose=verbose,
                              n_defect_st={'n_superad':n_superad,'n_subad':n_subad-1},
                              superad_dT={'fuel':superad_dT+fuel_shift,'oxy':superad_dT},
                              subad_dT={'fuel':subad_dT+fuel_shift,'oxy':subad_dT})

    return lib, lib_sup, lib_sub, lib_both, lib_shift

def run_inputs_check(type):
    if type=='eq':
        nonad_lib_fun=build_nonadiabatic_defect_eq_library
    elif type=='transient_slfm':
        nonad_lib_fun=build_nonadiabatic_defect_transient_slfm_library
    elif type=='steady_slfm':
        nonad_lib_fun=build_nonadiabatic_defect_steady_slfm_library
    else:
        raise ValueError("Unsupported function type.")

    flamelet_specs = get_flameletspecs()
    badT = -1.
    n_subad = 5
    results = []

    # negative temperatures
    try:
        lib = nonad_lib_fun(flamelet_specs,
                            n_defect_st=n_subad,
                            superad_dT=badT,
                            subad_dT=0.,
                            verbose=False)
        results.append(False)
    except:
        results.append(True)
    try:
        lib = nonad_lib_fun(flamelet_specs,
                            n_defect_st=n_subad,
                            superad_dT=0.,
                            subad_dT=badT,
                            verbose=False)
        results.append(False)
    except:
        results.append(True)
    try:
        lib = nonad_lib_fun(flamelet_specs,
                            n_defect_st=n_subad,
                            superad_dT={'fuel':0.,'oxy':badT},
                            subad_dT=0.,
                            verbose=False)
        results.append(False)
    except:
        results.append(True)
    try:
        lib = nonad_lib_fun(flamelet_specs,
                            n_defect_st=n_subad,
                            superad_dT={'fuel':badT,'oxy':0.},
                            subad_dT=0.,
                            verbose=False)
        results.append(False)
    except:
        results.append(True)
    try:
        lib = nonad_lib_fun(flamelet_specs,
                            n_defect_st=n_subad,
                            superad_dT={'fuel':0.,'oxy':0.},
                            subad_dT={'fuel':0.,'oxy':badT},
                            verbose=False)
        results.append(False)
    except:
        results.append(True)
    try:
        lib = nonad_lib_fun(flamelet_specs,
                            n_defect_st=n_subad,
                            superad_dT={'fuel':0.,'oxy':0.},
                            subad_dT={'fuel':badT,'oxy':0.},
                            verbose=False)
        results.append(False)
    except:
        results.append(True)

    # n_defect_st not the dictionary should be
    try:
        lib = nonad_lib_fun(flamelet_specs,
                            n_defect_st=n_subad,
                            superad_dT=10.,
                            subad_dT=0.,
                            verbose=False)
        results.append(False)
    except:
        results.append(True)
    try:
        lib = nonad_lib_fun(flamelet_specs,
                            n_defect_st={'n_superad':3},
                            superad_dT=10.,
                            subad_dT=0.,
                            verbose=False)
        results.append(False)
    except:
        results.append(True)

    results = np.array(results)
    return False if np.any(results==False) else True
