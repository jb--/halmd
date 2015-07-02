/*
 * Copyright © 2014  Jörg Bartnick
 * Copyright © 2008-2011  Peter Colberg and Felix Höfling
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
#include <boost/bind.hpp>
#include <cmath>
#include <memory>

#include <halmd/mdsim/gpu/integrators/brownian.hpp>
#include <halmd/utility/lua/lua.hpp>

namespace halmd {
namespace mdsim {
namespace gpu {
namespace integrators {

template <int dimension, typename float_type, typename RandomNumberGenerator>
brownian<dimension, float_type, RandomNumberGenerator>::brownian(
    std::shared_ptr<particle_type> particle
  , std::shared_ptr<box_type const> box
  , std::shared_ptr<random_type> random
  , float_type timestep
  , float_type temperature
  , float_type mobility
  , std::shared_ptr<logger> logger
)
  : particle_(particle)
  , box_(box)
  , random_(random)
  , mobility_(mobility)
  , logger_(logger)
{
    set_timestep(timestep);
    set_temperature(temperature);
    LOG("mobility: " << mobility_);
}

/**
 * set integration time-step
 */
template <int dimension, typename float_type, typename RandomNumberGenerator>
void brownian<dimension, float_type, RandomNumberGenerator>::set_timestep(double timestep)
{
    timestep_ = timestep;
    timestep_by_mobility_ = timestep / mobility_;
}

template <int dimension, typename float_type, typename RandomNumberGenerator>
void brownian<dimension, float_type, RandomNumberGenerator>::set_temperature(double temperature)
{
    temperature_ = temperature;
    width_rand_force_ = std::sqrt(2 * temperature_ / timestep_by_mobility_);
    LOG("temperature of heat bath: " << temperature_);
}

/**
 * This is not needed for brownian dynamics
 */
template <int dimension, typename float_type, typename RandomNumberGenerator>
void brownian<dimension, float_type, RandomNumberGenerator>::integrate()
{
    LOG_TRACE("update positions");
    force_array_type const& force = read_cache(particle_->force());

    // invalidate the particle caches after accessing the force!
    auto position = make_cache_mutable(particle_->position());
    auto image = make_cache_mutable(particle_->image());
    auto velocity = make_cache_mutable(particle_->velocity());

    scoped_timer_type timer(runtime_.integrate);

    try {
        // use CUDA execution dimensions of 'random' since
        // the kernel makes use of the random number generator
        cuda::configure(random_->rng().dim.grid, random_->rng().dim.block);
        wrapper_type::kernel.integrate(
            &*position->begin()
          , &*image->begin()
          , &*velocity->begin()
          , &*force.begin()
          , width_rand_force_
          , timestep_by_mobility_
          , particle_->nparticle()
          , particle_->dim.threads()
          , random_->rng().rng()
          , static_cast<vector_type>(box_->length())
        );
        cuda::thread::synchronize();
    }
    catch (cuda::error const&) {
        LOG_ERROR("failed to stream bd integration step on GPU");
        throw;
    }
}

/**
 * Second leapfrog half-step of velocity-Verlet algorithm
 */
template <int dimension, typename float_type, typename RandomNumberGenerator>
void brownian<dimension, float_type, RandomNumberGenerator>::finalize()
{}

template <int dimension, typename float_type, typename RandomNumberGenerator>
void brownian<dimension, float_type, RandomNumberGenerator>::luaopen(lua_State* L)
{
    using namespace luaponte;
    module(L, "libhalmd")
    [
        namespace_("mdsim")
        [
            namespace_("integrators")
            [
                class_<brownian>()
                    .def("integrate", &brownian::integrate)
                    .def("finalize", &brownian::finalize)
                    .def("set_timestep", &brownian::set_timestep)
                    .def("set_temperature", &brownian::set_temperature)
                    .property("timestep", &brownian::timestep)
                    .property("temperature", &brownian::temperature)
                    .property("mobility", &brownian::mobility)
                    .scope
                    [
                        class_<runtime>()
                            .def_readonly("integrate", &runtime::integrate)
                            .def_readonly("finalize", &runtime::finalize)
                    ]
                    .def_readonly("runtime", &brownian::runtime_)

              , def("brownian", &std::make_shared<brownian
                  , std::shared_ptr<particle_type>
                  , std::shared_ptr<box_type const>
                  , std::shared_ptr<random_type>
                  , float_type
                  , float_type
                  , float_type
                  , std::shared_ptr<logger>
                >)
            ]
        ]
    ];
}

HALMD_LUA_API int luaopen_libhalmd_mdsim_gpu_integrators_brownian(lua_State* L)
{
    brownian<3, float, random::gpu::rand48>::luaopen(L);
    brownian<2, float, random::gpu::rand48>::luaopen(L);
    return 0;
}

// explicit instantiation
template class brownian<3, float, random::gpu::rand48>;
template class brownian<2, float, random::gpu::rand48>;

} // namespace integrators
} // namespace gpu
} // namespace mdsim
} // namespace halmd
