/*
 * Copyright © 2008-2010  Peter Colberg and Felix Höfling
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

#include <halmd/observables/host/trajectory.hpp>
#include <halmd/utility/lua_wrapper/lua_wrapper.hpp>

using namespace boost;
using namespace std;

namespace halmd
{
namespace observables { namespace host
{

template <int dimension, typename float_type>
trajectory<dimension, float_type>::trajectory(
    shared_ptr<sample_type> sample
  , shared_ptr<particle_type> particle
  , shared_ptr<box_type> box
)
  : sample(sample)
  , particle(particle)
  , box(box)
{
}

/**
 * Sample trajectory
 */
template <int dimension, typename float_type>
void trajectory<dimension, float_type>::acquire(double time)
{
    for (size_t i = 0; i < particle->nbox; ++i) {
        // periodically extended particle position
        (*sample->r[particle->type[i]])[particle->tag[i]] = particle->r[i] + element_prod(particle->image[i], box->length());
        // particle velocity
        (*sample->v[particle->type[i]])[particle->tag[i]] = particle->v[i];
    }
    sample->time = time;
}

template <int dimension, typename float_type>
void trajectory<dimension, float_type>::luaopen(lua_State* L)
{
    using namespace luabind;
    string class_name("trajectory_" + lexical_cast<string>(dimension) + "_");
    module(L, "halmd_wrapper")
    [
        namespace_("observables")
        [
            namespace_("host")
            [
                class_<trajectory, shared_ptr<_Base>, _Base>(class_name.c_str())
                    .def(constructor<
                         shared_ptr<sample_type>
                       , shared_ptr<particle_type>
                       , shared_ptr<box_type>
                    >())
            ]
        ]
    ];
}

static __attribute__((constructor)) void register_lua()
{
    lua_wrapper::register_(1) //< distance of derived to base class
#ifndef USE_HOST_SINGLE_PRECISION
    [
        &trajectory<3, double>::luaopen
    ]
    [
        &trajectory<2, double>::luaopen
    ];
#else
    [
        &trajectory<3, float>::luaopen
    ]
    [
        &trajectory<2, float>::luaopen
    ];
#endif
}

// explicit instantiation
#ifndef USE_HOST_SINGLE_PRECISION
template class trajectory<3, double>;
template class trajectory<2, double>;
#else
template class trajectory<3, float>;
template class trajectory<2, float>;
#endif

}} // namespace observables::host

} // namespace halmd