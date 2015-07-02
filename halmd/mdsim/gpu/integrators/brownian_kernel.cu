/*
 * Copyright © 2014 Jörg Bartnick
 * Copyright © 2008-2014 Felix Höfling
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

// uncomment this line for a thread-divergent, slower implementation
// of the original thermostat introduced by H. C. Andersen (1978).
// #define USE_ORIGINAL_ANDERSEN_THERMOSTAT

#include <halmd/mdsim/gpu/box_kernel.cuh>
#include <halmd/mdsim/gpu/integrators/brownian_kernel.hpp>
#include <halmd/numeric/blas/blas.hpp>
#include <halmd/numeric/mp/dsfloat.hpp>
#include <halmd/random/gpu/normal_distribution.cuh>
#include <halmd/random/gpu/random_number_generator.cuh>
#include <halmd/utility/gpu/thread.cuh>

#if __CUDA_ARCH__ < 120
# define USE_ORIGINAL_ANDERSEN_THERMOSTAT
#endif

namespace halmd {
namespace mdsim {
namespace gpu {
namespace integrators {
namespace brownian_kernel {


/**
 *
 * CUDA execution dimensions must agree with random number generator
 *
 * @param g_velocity particle velocities (array of size \code{} 2 * nplace \endcode for dsfloat arithmetic)
 * @param g_force particle forces (array of size \code{} nplace \endcode)
 * @param timestep integration time-step
 * @param width_rand_force Width of the random force ( = sqrt(2 kT xi / dt) )
 * @param coll_prob collision probability with heat bath
 * @param npart number of particles
 * @param nplace number of placeholder particles
 * @param rng random number generator
 * 
 * 
 */
template <int dimension, typename float_type, typename gpu_vector_type, typename rng_type>
__global__ void integrate(
    float4* g_position
  , gpu_vector_type* g_image
  , float4* g_velocity
  , gpu_vector_type const* g_force
  , float width_rand_force
  , float timestep_by_mobility
  , unsigned int npart
  , unsigned int nplace
  , rng_type rng
  , fixed_vector<float, dimension> box_length
)
{
    typedef fixed_vector<float_type, dimension> vector_type;
    typedef fixed_vector<float, dimension> float_vector_type;
 
    
    // read random number generator state from global device memory
    typename rng_type::state_type state = rng[GTID];

    // cache second normal variate for odd dimensions
    bool cached = false;
    float_type cache;
    
    for (uint i = GTID; i < npart; i += GTDIM) {
        // read position, species, velocity, mass, image, force from global memory
        vector_type r;
        unsigned int species;
        fixed_vector<float_type, dimension> v;
        float mass;

#ifdef USE_VERLET_DSFUN
        tie(r, species) <<= tie(g_position[i], g_position[i + nplace]);
        tie(v, mass) <<= tie(g_velocity[i], g_velocity[i + nplace]);
#else
        tie(r, species) <<= g_position[i];
        tie(v, mass) <<= g_velocity[i];
#endif

        float_vector_type f_rand;
        float_vector_type f = g_force[i];
        f_rand = 0;
        if (width_rand_force != 0) {
            /* Width of the random force = sqrt(2 kT xi / dt) 
             * => random displacement with sigma = sqrt (2 D dt) = sqrt(2 kT dt / xi))
             * the random force is multiplied again by timestep_by_mobility, thus the difference
             */
            // parameters for normal distribution
            float const mean = 0;
            float const sigma = width_rand_force;
            // assign random velocity according to Maxwell-Boltzmann distribution
            for (uint j = 0; j < dimension - 1; j += 2) {
            tie(f_rand[j], f_rand[j + 1]) = normal(rng, state, mean, sigma);
            }
            if (dimension % 2) {
            if ((cached = !cached)) {
                tie(f_rand[dimension - 1], cache) = normal(rng, state, mean, sigma);
            }   
            else {
                f_rand[dimension - 1] = cache;
            }
            }
        }
        // advance position
        v = (f + f_rand);
        r += v * timestep_by_mobility;

        float_vector_type image = box_kernel::reduce_periodic(r, box_length);
        // store position, species, velocity, mass, image in global memory
#ifdef USE_VERLET_DSFUN
        tie(g_position[i], g_position[i + nplace]) <<= tie(r, species);
        tie(g_velocity[i], g_velocity[i + nplace]) <<= tie(v, mass);
#else
        g_position[i] <<= tie(r, species);
        g_velocity[i] <<= tie(v, mass);
#endif
        
        if (!(image == float_vector_type(0))) {
            g_image[i] = image + static_cast<float_vector_type>(g_image[i]);
        }
    } 
    // store random number generator state in global device memory
    rng[GTID] = state; 
} 
} // namespace brownian_kernel

template <int dimension, typename rng_type>
brownian_wrapper<dimension, rng_type> const
brownian_wrapper<dimension, rng_type>::kernel = {
#ifdef USE_VERLET_DSFUN
   brownian_kernel::integrate<dimension, dsfloat>
#else
   brownian_kernel::integrate<dimension, float>
#endif
};

template class brownian_wrapper<3, random::gpu::rand48_rng>;
template class brownian_wrapper<2, random::gpu::rand48_rng>;

} // namespace mdsim
} // namespace gpu
} // namespace integrators
} // namespace halmd
