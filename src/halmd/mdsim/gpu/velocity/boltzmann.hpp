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

#ifndef HALMD_MDSIM_GPU_VELOCITY_BOLTZMANN_HPP
#define HALMD_MDSIM_GPU_VELOCITY_BOLTZMANN_HPP

#include <utility>

#include <halmd/mdsim/gpu/particle.hpp>
#include <halmd/mdsim/gpu/velocity/boltzmann_kernel.hpp>
#include <halmd/mdsim/velocity.hpp>
#include <halmd/numeric/gpu/blas/dsfloat.cuh>
#include <halmd/random/gpu/random.hpp>
#include <halmd/utility/options.hpp>

namespace halmd
{
namespace mdsim { namespace gpu { namespace velocity
{

using numeric::gpu::blas::dsfloat;

template <int dimension, typename float_type, typename RandomNumberGenerator>
class boltzmann
  : public mdsim::velocity<dimension>
{
public:
    // module definitions
    typedef boltzmann _Self;
    typedef mdsim::velocity<dimension> _Base;
    static void options(po::options_description& desc);
    static void depends();
    static void select(po::options const& vm);

    typedef gpu::particle<dimension, float_type> particle_type;
    typedef typename particle_type::vector_type vector_type;
    typedef typename particle_type::gpu_vector_type gpu_vector_type;
    typedef random::gpu::random<RandomNumberGenerator> random_type;
    typedef typename random_type::rng_type rng_type;
#ifdef USE_VERLET_DSFUN
    typedef boltzmann_wrapper<dimension, dsfloat, rng_type> wrapper_type;
#else
    typedef boltzmann_wrapper<dimension, float, rng_type> wrapper_type;
#endif
    typedef typename wrapper_type::gaussian_impl_type gaussian_impl_type;

    shared_ptr<particle_type> particle;
    shared_ptr<random_type> random;

    boltzmann(modules::factory& factory, po::options const& vm);
    virtual ~boltzmann() {};
    void set();

    gaussian_impl_type const gaussian_impl;
    static gaussian_impl_type get_gaussian_impl(int threads);

// private:
    /** assign new velocities from Gaussian distribution of width sigma,
      * return mean velocity and mean-square velocity */
    std::pair<vector_type, float_type> gaussian(float_type sigma);
    /** shift all velocities by 'v_shift' */
    void shift(vector_type const& v_shift);
    /** rescale magnitude of all velocities by factor 'scale' */
    void rescale(float_type scale);
    /** first shift, then rescale all velocities */
    void shift_rescale(vector_type const& v_shift, float_type scale);

protected:
    /** temperature */
    float_type temp_;
    /** block sum of velocity */
    cuda::vector<gpu_vector_type> g_vcm_;
    /** block sum of squared velocity */
    cuda::vector<dsfloat> g_vv_;
};

}}} // namespace mdsim::gpu::velocity

} // namespace halmd

#endif /* ! HALMD_MDSIM_GPU_VELOCITY_BOLTZMANN_HPP */
