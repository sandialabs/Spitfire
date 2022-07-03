import pickle
from os.path import abspath, join


def run():
    from spitfire.chemistry.mechanism import ChemicalMechanismSpec
    from spitfire.chemistry.reactors import HomogeneousReactor
    import numpy as np

    xml = abspath(join('tests', 'test_mechanisms', 'h2-burke.yaml'))
    mechanism = ChemicalMechanismSpec(cantera_input=xml, group_name='h2-burke')

    air = mechanism.stream(stp_air=True)
    fuel = mechanism.stream('X', 'H2:1')

    mix = mechanism.mix_for_equivalence_ratio(1.0, fuel, air)
    mix.TP = 1200., 101325.

    reactor_dict = {'cp, adiabatic': HomogeneousReactor(mechanism, mix, 'isobaric', 'adiabatic', 'closed'),
                    'cp, isothermal': HomogeneousReactor(mechanism, mix, 'isobaric', 'isothermal', 'closed'),
                    'cv, adiabatic': HomogeneousReactor(mechanism, mix, 'isochoric', 'adiabatic', 'closed'),
                    'cv, isothermal': HomogeneousReactor(mechanism, mix, 'isochoric', 'isothermal', 'closed')}

    sol_dict = dict()
    for r in reactor_dict:
        output = reactor_dict[r].integrate_to_steady()
        Y = np.array([output['mass fraction ' + s].copy() for s in mechanism.species_names])
        sol_dict[r] = (output.time_values.copy(), output['temperature'], Y)

    return sol_dict


if __name__ == '__main__':
    output = run()
    gold_pkl = abspath(join('tests', 'reactor', 'closed_reactors', 'gold.pkl'))
    with open(gold_pkl, 'wb') as file_output:
        pickle.dump(output, file_output)
