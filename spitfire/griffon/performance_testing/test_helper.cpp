#include "test_helper.h"
#include <algorithm>

namespace griffon
{

void Timer::tic()
{
    tic_time = std::chrono::high_resolution_clock::now();
}

double
Timer::toc(const double multiplier, const int repeats)
{
    toc_time = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::duration<double>>(toc_time - tic_time).count() * multiplier / ((double)repeats);
}

void print_result(const int ns, const int nz, const int rp, const std::string name, const double dcput)
{
    std::cout << rp << " repeats of \"" << name << "\" with nspec = " << ns << " and ngrid = " << nz
              << " completed, average of " << dcput << " us each\n";
}

void print_result(const int ns, const int nr, const int nz, const int rp, const std::string name, const double dcput)
{
    std::cout << rp << " repeats of \"" << name << "\" with nspec = " << ns << " and nr = " << nr << " and ngrid = " << nz
              << " completed, average of " << dcput << " us each\n";
}

griffon::CombustionKernels make_fake_mechanism(const int ns, const int nr)
{
    griffon::CombustionKernels m;

    m.mechanism_add_element("H");
    m.mechanism_add_element("O");
    m.mechanism_add_element("C");
    m.mechanism_add_element("N");

    m.mechanism_set_ref_pressure(101325.);
    m.mechanism_set_ref_temperature(300.);

    using AtomMap = std::map<std::string, double>;
    std::vector<std::pair<std::string, AtomMap>> available_species;

    std::vector<std::string> elements = {{"H", "O", "C", "N"}};
    std::vector<std::string> species_names;
    for (const auto &e : elements)
    {
        {
            const auto species_name = e;
            if (std::find(species_names.begin(), species_names.end(), species_name) == species_names.end())
            {
                species_names.push_back(species_name);
                available_species.push_back(std::make_pair(species_name, AtomMap({{e, 1.}})));
            }
        }
        {
            const auto species_name = e + "2";
            if (std::find(species_names.begin(), species_names.end(), species_name) == species_names.end())
            {
                species_names.push_back(species_name);
                available_species.push_back(std::make_pair(species_name, AtomMap({{e, 2.}})));
            }
        }
    }
    for (const auto &e1 : elements)
    {
        for (const auto &e2 : elements)
        {
            {
                const auto species_name = e1 + e2;
                if (std::find(species_names.begin(), species_names.end(), species_name) == species_names.end())
                {
                    species_names.push_back(species_name);
                    available_species.push_back(std::make_pair(species_name, AtomMap({{e1, 1.}, {e2, 1.}})));
                }
            }
            {
                const auto species_name = e1 + "2" + e2;
                if (std::find(species_names.begin(), species_names.end(), species_name) == species_names.end())
                {
                    species_names.push_back(species_name);
                    available_species.push_back(std::make_pair(species_name, AtomMap({{e1, 2.}, {e2, 1.}})));
                }
            }
            {
                const auto species_name = e1 + e2 + "2";
                if (std::find(species_names.begin(), species_names.end(), species_name) == species_names.end())
                {
                    species_names.push_back(species_name);
                    available_species.push_back(std::make_pair(species_name, AtomMap({{e1, 1.}, {e2, 2.}})));
                }
            }
            {
                const auto species_name = e1 + "2" + e2 + "2";
                if (std::find(species_names.begin(), species_names.end(), species_name) == species_names.end())
                {
                    species_names.push_back(species_name);
                    available_species.push_back(std::make_pair(species_name, AtomMap({{e1, 2.}, {e2, 2.}})));
                }
            }
            {
                const auto species_name = e1 + "3" + e2;
                if (std::find(species_names.begin(), species_names.end(), species_name) == species_names.end())
                {
                    species_names.push_back(species_name);
                    available_species.push_back(std::make_pair(species_name, AtomMap({{e1, 3.}, {e2, 1.}})));
                }
            }
            {
                const auto species_name = e1 + "3" + e2 + "2";
                if (std::find(species_names.begin(), species_names.end(), species_name) == species_names.end())
                {
                    species_names.push_back(species_name);
                    available_species.push_back(std::make_pair(species_name, AtomMap({{e1, 3.}, {e2, 2.}})));
                }
            }
            {
                const auto species_name = e1 + e2 + "3";
                if (std::find(species_names.begin(), species_names.end(), species_name) == species_names.end())
                {
                    species_names.push_back(species_name);
                    available_species.push_back(std::make_pair(species_name, AtomMap({{e1, 1.}, {e2, 3.}})));
                }
            }
            {
                const auto species_name = e1 + "2" + e2 + "3";
                if (std::find(species_names.begin(), species_names.end(), species_name) == species_names.end())
                {
                    species_names.push_back(species_name);
                    available_species.push_back(std::make_pair(species_name, AtomMap({{e1, 2.}, {e2, 3.}})));
                }
            }
            {
                const auto species_name = e1 + "3" + e2 + "3";
                if (std::find(species_names.begin(), species_names.end(), species_name) == species_names.end())
                {
                    species_names.push_back(species_name);
                    available_species.push_back(std::make_pair(species_name, AtomMap({{e1, 3.}, {e2, 3.}})));
                }
            }
        }
    }
    for (const auto &e1 : elements)
    {
        for (const auto &e2 : elements)
        {
            for (const auto &e3 : elements)
            {
                {
                    const auto species_name = e1 + e2 + e3;
                    if (std::find(species_names.begin(), species_names.end(), species_name) == species_names.end())
                    {
                        species_names.push_back(species_name);
                        available_species.push_back(std::make_pair(species_name, AtomMap({{e1, 1.}, {e2, 1.}, {e3, 1.}})));
                    }
                }
                {
                    const auto species_name = "2" + e1 + e2 + e3;
                    if (std::find(species_names.begin(), species_names.end(), species_name) == species_names.end())
                    {
                        species_names.push_back(species_name);
                        available_species.push_back(std::make_pair(species_name, AtomMap({{e1, 2.}, {e2, 1.}, {e3, 1.}})));
                    }
                }
                {
                    const auto species_name = e1 + "2" + e2 + e3;
                    if (std::find(species_names.begin(), species_names.end(), species_name) == species_names.end())
                    {
                        species_names.push_back(species_name);
                        available_species.push_back(std::make_pair(species_name, AtomMap({{e1, 1.}, {e2, 2.}, {e3, 1.}})));
                    }
                }
                {
                    const auto species_name = e1 + e2 + "2" + e3;
                    if (std::find(species_names.begin(), species_names.end(), species_name) == species_names.end())
                    {
                        species_names.push_back(species_name);
                        available_species.push_back(std::make_pair(species_name, AtomMap({{e1, 1.}, {e2, 1.}, {e3, 2.}})));
                    }
                }
            }
        }
    }
    if (ns > species_names.size())
    {
        std::stringstream ss;
        ss << "Problem in make_fake_mechanism(nspecies) - only " << species_names.size() << " species available\n";
        throw std::logic_error(ss.str());
    }

    for (int is = 0; is < ns; ++is)
    {
        m.mechanism_add_species(available_species[is].first, available_species[is].second);
    }
    m.mechanism_resize_heat_capacity_data();

    for (int is = 0; is < ns; ++is)
    {
        const double idb = (double)is / ((double)ns);
        const std::vector<double> c_low = {{idb * 0.1, idb * 1.e-4, idb * -1.e-7, idb * -1.e-10, idb * 1.e-12, idb * -1.e3, idb * -1.}};
        const std::vector<double> c_high = {{idb * 0.2, idb * 2.e-3, idb * -2.e-8, idb * -2.e-9, idb * 2.e-11, idb * -2.e2, idb * -2.}};
        m.mechanism_add_nasa7_cp(available_species[is].first, 200., 1000., 3000., c_low, c_high);
    }

    for (int ir = 0; ir < nr * 8 / 10; ++ir)
    {
        const std::map<std::string, int> reactants = {{"H", 1}, {"O2", 1}};
        const std::map<std::string, int> products = {{"HO", 1}, {"O", 1}};
        m.mechanism_add_reaction_simple(reactants, products, true, 10., 1., 1.e5);
    }

    for (int ir = 0; ir < nr * 1 / 10; ++ir)
    {
        const std::map<std::string, int> reactants = {{"H", 1}, {"O2", 1}};
        const std::map<std::string, int> products = {{"HO", 1}, {"O", 1}};
        const std::map<std::string, double> thirdbodies = {{"H", 1.5}, {"O2", 1.5}};
        m.mechanism_add_reaction_three_body(reactants, products, true, 10., 1., 1.e5, thirdbodies, 1.);
    }

    for (int ir = 0; ir < nr * 1 / 20; ++ir)
    {
        const std::map<std::string, int> reactants = {{"H", 1}, {"O2", 1}};
        const std::map<std::string, int> products = {{"HO", 1}, {"O", 1}};
        const std::map<std::string, double> thirdbodies = {{"H", 1.5}, {"O2", 1.5}};
        m.mechanism_add_reaction_Lindemann(reactants, products, true, 10., 1., 1.e5, thirdbodies, 1., 0.1, 1.1, 1.e7);
    }

    for (int ir = 0; ir < nr * 1 / 20; ++ir)
    {
        const std::map<std::string, int> reactants = {{"H", 1}, {"O2", 1}};
        const std::map<std::string, int> products = {{"HO", 1}, {"O", 1}};
        const std::map<std::string, double> thirdbodies = {{"H", 1.5}, {"O2", 1.5}};
        const std::vector<double> troe_params = {{0.5, 1.e-30, 1.e30, 0.0}};
        m.mechanism_add_reaction_Troe(reactants, products, true, 10., 1., 1.e5, thirdbodies, 1., 0.1, 1.1, 1.e7, troe_params);
    }

    return m;
}
} // namespace griffon
