/*
 * Copyright © 2008-2010  Peter Colberg
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

#include <algorithm>
#include <boost/array.hpp>
#include <boost/algorithm/string/join.hpp>
#include <boost/lexical_cast.hpp>
#include <exception>
#include <numeric>

#include <halmd/io/logger.hpp>
#include <halmd/mdsim/particle.hpp>
#include <halmd/utility/lua_wrapper/lua_wrapper.hpp>

using namespace boost;
using namespace boost::algorithm;
using namespace std;

namespace halmd
{
namespace mdsim
{

/**
 * Assemble module options
 */
template <int dimension>
void particle<dimension>::options(po::options_description& desc)
{
    multi_array<unsigned int, 1> default_particles(extents[1]);
    default_particles[0] = 1000;

    desc.add_options()
        ("particles,N", po::value<multi_array<unsigned int, 1> >()->default_value(default_particles),
         "number of particles")
        ;
}

/**
 * Register option value types with Lua
 */
static __attribute__((constructor)) void register_option_converters()
{
    register_any_converter<boost::multi_array<unsigned int, 1> >();
}

/**
 * Construct microscopic system state.
 *
 * @param particles number of particles per type or species
 */
template <int dimension>
particle<dimension>::particle(vector<unsigned int> const& particles)
  : nbox(accumulate(particles.begin(), particles.end(), 0))
  , ntype(particles.size())
  , ntypes(particles)
{
    if (*min_element(this->ntypes.begin(), this->ntypes.end()) < 1) {
        throw logic_error("invalid number of particles");
    }

    vector<string> ntypes_(ntypes.size());
    transform(
        ntypes.begin()
      , ntypes.end()
      , ntypes_.begin()
      , lexical_cast<string, unsigned int>
    );

    LOG("number of particles: " << nbox);
    LOG("number of particle types: " << ntype);
    LOG("number of particles per type: " << join(ntypes_, " "));
}

template <typename T>
static void register_lua(lua_State* L, char const* class_name)
{
    using namespace luabind;
    module(L)
    [
        namespace_("halmd_wrapper")
        [
            namespace_("mdsim")
            [
                class_<T, shared_ptr<T> >(class_name)
                    .def_readonly("nbox", &T::nbox)
                    .def_readonly("ntype", &T::ntype)
                    .def_readonly("ntypes", &T::ntypes)
                    .scope
                    [
                        def("options", &T::options)
                    ]
            ]
        ]
    ];
}

static __attribute__((constructor)) void register_lua()
{
    lua_wrapper::register_(0) //< distance of derived to base class
    [
        bind(&register_lua<particle<3> >, _1, "particle_3_")
    ]
    [
        bind(&register_lua<particle<2> >, _1, "particle_2_")
    ];
}

// explicit instantiation
template class particle<3>;
template class particle<2>;

} // namespace mdsim

} // namespace halmd