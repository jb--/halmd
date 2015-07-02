/*
 * Copyright © 2014 Jörg Bartnick
 * Copyright © 2008-2012  Peter Colberg and Felix Höfling
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

#include <halmd/mdsim/host/integrators/brownian.hpp>
#include <halmd/utility/lua/lua.hpp>

namespace halmd {
namespace mdsim {
namespace host {
namespace integrators {

template <int dimension, typename float_type>
brownian<dimension, float_type>::brownian(
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
    LOG("mobility set as: " << mobility_);
}

template <int dimension, typename float_type>
void brownian<dimension, float_type>::set_timestep(double timestep)
{
    timestep_ = timestep; 
    timestep_by_mobility_ = timestep / mobility_;
    LOG("timestep: " << timestep_);
}

template <int dimension, typename float_type>
void brownian<dimension, float_type>::set_temperature(double temperature)
{
    temperature_ = temperature;
    width_rand_force_ = std::sqrt(2 * temperature_ / timestep_by_mobility_);
    LOG("temperature of heat bath: " << temperature_);
}

template <int dimension, typename float_type>
void brownian<dimension, float_type>::integrate()
{
    LOG_TRACE("update positions and velocities")
    force_array_type const& force = read_cache(particle_->force());
    mass_array_type const& mass = read_cache(particle_->mass());
    size_type nparticle = particle_->nparticle();

    // invalidate the particle caches after accessing the force!
    auto position = make_cache_mutable(particle_->position());
    auto image = make_cache_mutable(particle_->image());
    auto velocity = make_cache_mutable(particle_->velocity());

    scoped_timer_type timer(runtime_.integrate);

    // cache random numbers
    float_type rng_cache = 0;
    bool rng_cache_valid = false;
    // displacement due to random force 
    vector_type f_rand;
 

    // loop over all particles
    for (size_type i = 0; i < nparticle; ++i) {
        vector_type& r = (*position)[i];
        vector_type& v = (*velocity)[i];
        f_rand = 0;
        
        // stochastic coupling with heat bath
        if (temperature_ != 0) {
            // assign two velocity components at a time
            for (unsigned int i = 0; i < dimension - 1; i += 2) {
                boost::tie(f_rand[i], f_rand[i + 1]) = random_->normal(width_rand_force_);
            }
            // handle last component separately for odd dimensions
            if (dimension % 2 == 1) {
                if (rng_cache_valid) {
                    f_rand[dimension - 1] = rng_cache;
                }
                else {
                    std::tie(f_rand[dimension - 1], rng_cache) = random_->normal(width_rand_force_);
                }
                rng_cache_valid = !rng_cache_valid;
            }
        }
        v = (force[i] + f_rand);
        r += v * timestep_by_mobility_;
        (*image)[i] += box_->reduce_periodic(r);
    }
}

template <int dimension, typename float_type>
void brownian<dimension, float_type>::finalize()
{
    scoped_timer_type timer(runtime_.finalize);
}

template <int dimension, typename float_type>
void brownian<dimension, float_type>::luaopen(lua_State* L)
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

HALMD_LUA_API int luaopen_libhalmd_mdsim_host_integrators_brownian(lua_State* L)
{
#ifndef USE_HOST_SINGLE_PRECISION
    brownian<3, double>::luaopen(L);
    brownian<2, double>::luaopen(L);
#else
    brownian<3, float>::luaopen(L);
    brownian<2, float>::luaopen(L);
#endif
    return 0;
}

// explicit instantiation
#ifndef USE_HOST_SINGLE_PRECISION
template class brownian<3, double>;
template class brownian<2, double>;
#else
template class brownian<3, float>;
template class brownian<2, float>;
#endif

} // namespace integrators
} // namespace host
} // namespace mdsim
} // namespace halmd
