import pickle
from os.path import abspath, join


def run():
    from spitfire.chemistry.mechanism import ChemicalMechanismSpec
    from spitfire.chemistry.reactors import HomogeneousReactor
    from numpy import sin as sin, pi as pi, array

    from os.path import abspath, join
    xml = abspath(join('spitfire_test', 'test_mechanisms', 'h2-burke.xml'))
    mechanism = ChemicalMechanismSpec(cantera_xml=xml,
                                      group_name='h2-burke')

    air = mechanism.stream(stp_air=True)
    fuel = mechanism.stream('X', 'H2:1')

    mix = mechanism.mix_for_equivalence_ratio(1.0, fuel, air)
    mix.TP = 800., 101325.

    feed = mechanism.copy_stream(mix)

    reactor = HomogeneousReactor(mechanism, mix,
                                 configuration='isobaric',
                                 heat_transfer='adiabatic',
                                 mass_transfer='open',
                                 mixing_tau=1.e-5,
                                 feed_temperature=lambda t: 800. + 400. * sin(2. * pi * 10. * t),
                                 feed_mass_fractions=feed.Y)

    output = reactor.integrate_to_time(0.1, transient_tolerance=1.e-10, write_log=False, log_rate=200)

    Y = array([output['mass fraction ' + s].copy() for s in mechanism.species_names])

    return output.time_values.copy(), output['temperature'], Y


if __name__ == '__main__':
    output = run()
    gold_pkl = abspath(join('spitfire_test', 'reactor', 'oscillating_feed_temperature', 'gold.pkl'))
    with open(gold_pkl, 'wb') as file_output:
        pickle.dump(output, file_output)
