/*
 * Copyright © 2008-2011  Peter Colberg
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

#include <halmd/mdsim/gpu/particle_kernel.cuh>
#include <halmd/mdsim/gpu/positions/phase_space.hpp>
#include <halmd/mdsim/gpu/positions/phase_space_kernel.hpp>
#include <halmd/utility/lua/lua.hpp>

namespace halmd {
namespace mdsim {
namespace gpu {
namespace positions {

using namespace boost;
using namespace std;

template <int dimension, typename float_type>
phase_space<dimension, float_type>::phase_space(
    shared_ptr<particle_type> particle
  , shared_ptr<box_type const> box
  , shared_ptr<sample_type const> sample
  , shared_ptr<logger_type> logger
)
  // dependency injection
  : particle_(particle)
  , box_(box)
  , sample_(sample)
  , logger_(logger)
{
    try {
        cuda::copy(static_cast<vector_type>(box_->length()), phase_space_wrapper<dimension>::kernel.box_length);
    }
    catch (cuda::error const&)
    {
        LOG_ERROR("failed to copy box length to GPU");
        throw;
    }
}

/**
 * set particle positions
 */
template <int dimension, typename float_type>
void phase_space<dimension, float_type>::set()
{
    LOG("set particle positions from phase space sample");

    // assign particle coordinates and types
    size_t n = 0; // indicates the boundary to the next particle type
    for (size_t j = 0, i = 0; j < particle_->ntype; ++j) {
        typename sample_type::sample_vector const& r_sample = *sample_->r[j];
        n += particle_->ntypes[j];
        assert(particle_->ntypes[j] == r_sample.size());
        assert(n <= particle_->h_r.size());
        for (size_t k = 0; i < n; ++i, ++k) {
            particle_->h_r[i] = particle_kernel::tagged<vector_type>(r_sample[k], j);
        }
    }

    try {
#ifdef USE_VERLET_DSFUN
        // erase particle position vectors (double-single precision)
        cuda::memset(particle_->g_r, 0, particle_->g_r.capacity());
#endif
        cuda::copy(particle_->h_r, particle_->g_r);
    }
    catch (cuda::error const&)
    {
        LOG_ERROR("failed to copy particle positions to GPU");
        throw;
    }

    // shift particle positions to range (-L/2, L/2)
    try {
        cuda::configure(particle_->dim.grid, particle_->dim.block);
        phase_space_wrapper<dimension>::kernel.reduce_periodic(particle_->g_r, particle_->g_image);
    }
    catch (cuda::error const&)
    {
        LOG_ERROR("failed to reduce particle positions on GPU");
        throw;
    }
}

template <int dimension, typename float_type>
void phase_space<dimension, float_type>::luaopen(lua_State* L)
{
    using namespace luabind;
    static string class_name("phase_space_" + lexical_cast<string>(dimension) + "_");
    module(L, "libhalmd")
    [
        namespace_("mdsim")
        [
            namespace_("gpu")
            [
                namespace_("positions")
                [
                    class_<phase_space, shared_ptr<_Base>, _Base>(class_name.c_str())
                        .def(constructor<
                             shared_ptr<particle_type>
                           , shared_ptr<box_type const>
                           , shared_ptr<sample_type const>
                           , shared_ptr<logger_type>
                        >())
                ]
            ]
        ]
    ];
}

HALMD_LUA_API int luaopen_libhalmd_mdsim_gpu_positions_phase_space(lua_State* L)
{
    phase_space<3, float>::luaopen(L);
    phase_space<2, float>::luaopen(L);
    return 0;
}

// explicit instantiation
template class phase_space<3, float>;
template class phase_space<2, float>;

} // namespace mdsim
} // namespace gpu
} // namespace positions
} // namespace halmd
