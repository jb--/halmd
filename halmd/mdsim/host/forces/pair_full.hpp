/*
 * Copyright © 2010-2011 Felix Höfling
 * Copyright © 2013-2014 Nicolas Höft
 * Copyright © 2008-2012 Peter Colberg
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

#ifndef HALMD_MDSIM_HOST_FORCES_PAIR_FULL_HPP
#define HALMD_MDSIM_HOST_FORCES_PAIR_FULL_HPP

#include <halmd/io/logger.hpp>
#include <halmd/mdsim/box.hpp>
#include <halmd/mdsim/force_kernel.hpp>
#include <halmd/mdsim/host/particle.hpp>
#include <halmd/utility/lua/lua.hpp>
#include <halmd/utility/profiler.hpp>
#include <halmd/utility/signal.hpp>

#include <memory>
#include <tuple>

namespace halmd {
namespace mdsim {
namespace host {
namespace forces {

/**
 * template class for modules implementing short ranged potential forces
 */
template <int dimension, typename float_type, typename potential_type>
class pair_full
{
public:
    typedef particle<dimension, float_type> particle_type;
    typedef box<dimension> box_type;

    pair_full(
        std::shared_ptr<potential_type const> potential
      , std::shared_ptr<particle_type> particle1
      , std::shared_ptr<particle_type const> particle2
      , std::shared_ptr<box_type const> box
      , float_type aux_weight = 1
      , std::shared_ptr<halmd::logger> logger = std::make_shared<halmd::logger>()
    );

    /**
     * Check if the force cache (of the particle module) is up-to-date and if
     * not, mark the cache as dirty.
     */
    void check_cache();

    /**
     * Compute and apply the force to the particles.
     */
    void apply();

    /**
     * Bind class to Lua.
     */
    static void luaopen(lua_State* L);

private:
    typedef typename particle_type::position_array_type position_array_type;
    typedef typename particle_type::position_type position_type;
    typedef typename particle_type::species_array_type species_array_type;
    typedef typename particle_type::species_type species_type;
    typedef typename particle_type::size_type size_type;
    typedef typename particle_type::force_array_type force_array_type;
    typedef typename particle_type::en_pot_array_type en_pot_array_type;
    typedef typename particle_type::stress_pot_array_type stress_pot_array_type;
    typedef typename particle_type::stress_pot_type stress_pot_type;
    typedef typename particle_type::en_pot_type en_pot_type;

    /** compute forces */
    void compute_();
    /** compute forces with auxiliary variables */
    void compute_aux_();

    /** pair potential */
    std::shared_ptr<potential_type const> potential_;
    /** state of first system */
    std::shared_ptr<particle_type> particle1_;
    /** state of second system */
    std::shared_ptr<particle_type const> particle2_;
    /** simulation domain */
    std::shared_ptr<box_type const> box_;
    /** weight for auxiliary variables */
    float_type aux_weight_;
    /** module logger */
    std::shared_ptr<logger> logger_;

    /** cache observer of force per particle */
    std::tuple<cache<>, cache<>, cache<>, cache<>> force_cache_;
    /** cache observer of auxiliary variables */
    std::tuple<cache<>, cache<>, cache<>, cache<>> aux_cache_;

    typedef utility::profiler::accumulator_type accumulator_type;
    typedef utility::profiler::scoped_timer_type scoped_timer_type;

    struct runtime
    {
        accumulator_type compute;
        accumulator_type compute_aux;
    };
    /** profiling runtime accumulators */
    runtime runtime_;
};

template <int dimension, typename float_type, typename potential_type>
pair_full<dimension, float_type, potential_type>::pair_full(
    std::shared_ptr<potential_type const> potential
  , std::shared_ptr<particle_type> particle1
  , std::shared_ptr<particle_type const> particle2
  , std::shared_ptr<box_type const> box
  , float_type aux_weight
  , std::shared_ptr<logger> logger
)
  : potential_(potential)
  , particle1_(particle1)
  , particle2_(particle2)
  , box_(box)
  , aux_weight_(aux_weight)
  , logger_(logger)
{
    if (std::min(potential_->size1(), potential_->size2()) < std::max(particle1_->nspecies(), particle2_->nspecies())) {
        throw std::invalid_argument("size of potential coefficients less than number of particle species");
    }
}

template <int dimension, typename float_type, typename potential_type>
inline void pair_full<dimension, float_type, potential_type>::check_cache()
{
    cache<position_array_type> const& position1_cache = particle1_->position();
    cache<position_array_type> const& position2_cache = particle2_->position();
    cache<species_array_type> const& species1_cache = particle1_->species();
    cache<species_array_type> const& species2_cache = particle2_->species();

    auto current_state = std::tie(position1_cache, position2_cache, species1_cache, species2_cache);

    if (force_cache_ != current_state) {
        particle1_->mark_force_dirty();
    }

    if (aux_cache_ != current_state) {
        particle1_->mark_aux_dirty();
    }
}

template <int dimension, typename float_type, typename potential_type>
inline void pair_full<dimension, float_type, potential_type>::apply()
{
    cache<position_array_type> const& position1_cache = particle1_->position();
    cache<position_array_type> const& position2_cache = particle2_->position();
    cache<species_array_type> const& species1_cache = particle1_->species();
    cache<species_array_type> const& species2_cache = particle2_->species();

    auto current_state = std::tie(position1_cache, position2_cache, species1_cache, species2_cache);

    if (particle1_->aux_enabled()) {
        compute_aux_();
        force_cache_ = current_state;
        aux_cache_ = force_cache_;
    }
    else {
        compute_();
        force_cache_ = current_state;
    }
    particle1_->force_zero_disable();
}

template <int dimension, typename float_type, typename potential_type>
inline void pair_full<dimension, float_type, potential_type>::compute_()
{
    auto force = make_cache_mutable(particle1_->mutable_force());

    position_array_type const& position1 = read_cache(particle1_->position());
    position_array_type const& position2 = read_cache(particle2_->position());
    species_array_type const& species1   = *particle1_->species();
    species_array_type const& species2   = *particle2_->species();
    size_type nparticle1 = particle1_->nparticle();
    size_type nparticle2 = particle2_->nparticle();

    LOG_TRACE("compute forces");

    scoped_timer_type timer(runtime_.compute);

    // reset the force and auxiliary variables to zero if necessary
    if (particle1_->force_zero()) {
        std::fill(force->begin(), force->end(), 0);
    }

    // whether Newton's third law applies
    bool const reactio = false;

    for (size_type i = 0; i < nparticle1; ++i) {
        // calculate untruncated pairwise force with all other particles
        for (size_type j = reactio ? (i + 1) : 0; j < nparticle2; ++j) {
            // particle distance vector
            position_type r = position1[i] - position2[j];
            box_->reduce_periodic(r);
            // particle types
            species_type a = species1[i];
            species_type b = species2[j];
            // squared particle distance
            float_type rr = inner_prod(r, r);

            float_type fval, pot;
            std::tie(fval, pot) = (*potential_)(rr, a, b);

            // add force contribution to both particles
            (*force)[i] += r * fval;
            if (reactio) {
                (*force)[j] -= r * fval;
            }
        }
    }
}

template <int dimension, typename float_type, typename potential_type>
inline void pair_full<dimension, float_type, potential_type>::compute_aux_()
{
    auto force      = make_cache_mutable(particle1_->mutable_force());
    auto en_pot     = make_cache_mutable(particle1_->mutable_potential_energy());
    auto stress_pot = make_cache_mutable(particle1_->mutable_stress_pot());

    position_array_type const& position1 = read_cache(particle1_->position());
    position_array_type const& position2 = read_cache(particle2_->position());
    species_array_type const& species1   = *particle1_->species();
    species_array_type const& species2   = *particle2_->species();
    size_type nparticle1 = particle1_->nparticle();
    size_type nparticle2 = particle2_->nparticle();

    LOG_TRACE("compute forces with auxiliary variables");

    scoped_timer_type timer(runtime_.compute_aux);

    // reset the force and auxiliary variables to zero if necessary
    if (particle1_->force_zero()) {
        std::fill(force->begin(), force->end(), 0);
        std::fill(en_pot->begin(), en_pot->end(), 0);
        std::fill(stress_pot->begin(), stress_pot->end(), 0);
    }

    // whether Newton's third law applies
    bool const reactio = false;

    float_type weight = aux_weight_;
    if (reactio) {
        weight /= 2;
    }

    for (size_type i = 0; i < nparticle1; ++i) {
        // calculate untruncated pairwise force with all other particles
        for (size_type j = reactio ? (i + 1) : 0; j < nparticle2; ++j) {
            // particle distance vector
            position_type r = position1[i] - position2[j];
            box_->reduce_periodic(r);
            // particle types
            species_type a = species1[i];
            species_type b = species2[j];
            // squared particle distance
            float_type rr = inner_prod(r, r);

            float_type fval, pot;
            std::tie(fval, pot) = (*potential_)(rr, a, b);

            // add force contribution to both particles
            (*force)[i] += r * fval;
            if (reactio) {
                (*force)[j] -= r * fval;
            }

            // contribution to potential energy
            en_pot_type en = weight * pot;
            // potential part of stress tensor
            stress_pot_type stress = weight * fval * make_stress_tensor(r);

            // add contributions for first particle
            (*en_pot)[i]      += en;
            (*stress_pot)[i]  += stress;

            // add contributions for second particle
            if (reactio) {
                (*en_pot)[j]      += en;
                (*stress_pot)[j]  += stress;
            }
        }
    }
}


template <int dimension, typename float_type, typename potential_type>
void pair_full<dimension, float_type, potential_type>::luaopen(lua_State* L)
{
    using namespace luaponte;
    module(L, "libhalmd")
    [
        namespace_("mdsim")
        [
            namespace_("forces")
            [
                class_<pair_full>()
                    .def("check_cache", &pair_full::check_cache)
                    .def("apply", &pair_full::apply)
                    .scope
                    [
                        class_<runtime>("runtime")
                            .def_readonly("compute", &runtime::compute)
                            .def_readonly("compute_aux", &runtime::compute_aux)
                    ]
                    .def_readonly("runtime", &pair_full::runtime_)

              , def("pair_full", &std::make_shared<pair_full,
                    std::shared_ptr<potential_type const>
                  , std::shared_ptr<particle_type>
                  , std::shared_ptr<particle_type const>
                  , std::shared_ptr<box_type const>
                  , float
                  , std::shared_ptr<logger>
                >)
            ]
        ]
    ];
}

} // namespace forces
} // namespace host
} // namespace mdsim
} // namespace halmd

#endif /* ! HALMD_MDSIM_HOST_FORCES_PAIR_FULL_HPP */
