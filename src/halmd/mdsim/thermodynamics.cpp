/*
 * Copyright © 2010  Felix Höfling
 *
 * This file is part of HALMD.
 *
 * HALMD is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <halmd/mdsim/thermodynamics.hpp>
#include <halmd/utility/module.hpp>

using namespace boost;
using namespace std;

namespace halmd
{
namespace mdsim
{

/**
 * Assemble module options
 */
template <int dimension>
void thermodynamics<dimension>::options(po::options_description& desc)
{
    desc.add_options()
        ("disable-state-vars", po::bool_switch(),
         "disable evaluation and output of macroscopic state variables")
        ;
}

/**
 * Resolve module dependencies
 */
template <int dimension>
void thermodynamics<dimension>::depends()
{
    modules::depends<_Self, box_type>::required();
    modules::depends<_Self, writer_type>::required();
    modules::depends<_Self, profiler_type>::required();
}

template <int dimension>
void thermodynamics<dimension>::select(po::options const& vm)
{
    if (vm["disable-state-vars"].as<bool>()) {
        throw unsuitable_module("mismatching option: disable-state-vars");
    }
}

template <int dimension>
thermodynamics<dimension>::thermodynamics(modules::factory& factory, po::options const& vm)
  // dependency injection
  : box(modules::fetch<box_type>(factory, vm))
  , writer(modules::fetch<writer_type>(factory, vm))
  , profiler(modules::fetch<profiler_type>(factory, vm))
{
    writer->register_observable("TIME", &time_, "simulation time");
    writer->register_observable("EPOT", &en_pot_, "potential energy");
    writer->register_observable("EKIN", &en_kin_, "kinetic energy");
    writer->register_observable("ETOT", &en_tot_, "total energy");
//     writer->register_observable("VCM", &v_cm_, "centre-of-mass velocity");
    writer->register_observable("PRESS", &pressure_, "pressure");
    writer->register_observable("TEMP", &temp_, "temperature");
}

/**
 * Sample macroscopic state variables
 */
template <int dimension>
void thermodynamics<dimension>::sample(double time)
{
    // compute state variables and take care that
    // expensive functions are called only once
    en_pot_ = en_pot();
    en_kin_ = en_kin();
//     v_cm_ = v_cm();
    en_tot_ = en_pot_ + en_kin_;
    temp_ = 2 * en_kin_ / dimension;
    density_ = box->density();
    pressure_ = density_ * (temp_ + virial() / dimension);
    time_ = time;

    // call previously registered writer functions
    writer->write();
}


// explicit instantiation
template class thermodynamics<3>;
template class thermodynamics<2>;

} // namespace mdsim

} // namespace halmd
