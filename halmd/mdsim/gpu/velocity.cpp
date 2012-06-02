/*
 * Copyright © 2010  Peter Colberg
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

#include <halmd/mdsim/gpu/velocity.hpp>
#include <halmd/mdsim/gpu/velocity_kernel.hpp>
#include <halmd/numeric/mp/dsfloat.hpp>

namespace halmd {
namespace mdsim {
namespace gpu {

template <int dimension, typename float_type>
velocity<dimension, float_type>::velocity(
    std::shared_ptr<particle_type> particle
  , std::shared_ptr<logger_type> logger
)
  // dependency injection
  : particle_(particle)
  , logger_(logger)
  // set parameters
  , dim_(particle_->dim) // FIXME not used?
{
    cuda::copy(particle_->nparticle(), get_velocity_kernel<dimension>().nbox);
}

/**
 * Rescale magnitude of all velocities by 'factor'
 */
template <int dimension, typename float_type>
void velocity<dimension, float_type>::rescale(double factor)
{
    LOG_TRACE("rescale particle velocities by a factor of " << factor);
    cuda::configure(dim_.grid, dim_.block);
    get_velocity_kernel<dimension>().rescale(
        particle_->velocity()
      , particle_->dim.threads()
      , factor
    );
}

/**
 * Shift all velocities by 'delta'
 */
template <int dimension, typename float_type>
void velocity<dimension, float_type>::shift(vector_type const& delta)
{
    LOG_TRACE("shift particle velocities by " << delta);
    cuda::configure(dim_.grid, dim_.block);
    get_velocity_kernel<dimension>().shift(
        particle_->velocity()
      , particle_->dim.threads()
      , delta
    );
}

/**
 * First shift, then rescale all velocities
 */
template <int dimension, typename float_type>
void velocity<dimension, float_type>::shift_rescale(vector_type const& delta, double factor)
{
    LOG_TRACE("shift particle velocities by " << delta << " and rescale by a factor of " << factor);
    cuda::configure(dim_.grid, dim_.block);
    get_velocity_kernel<dimension>().shift_rescale(
        particle_->velocity()
      , particle_->dim.threads()
      , delta
      , factor
    );
}

template class velocity<3, float>;
template class velocity<2, float>;

} // namespace mdsim
} // namespace gpu
} // namespace halmd
