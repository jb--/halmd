/*
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

#include <halmd/mdsim/gpu/forces/pair_full_kernel.cuh>
#include <halmd/mdsim/gpu/forces/pair_trunc_kernel.cuh>
#include <halmd/mdsim/gpu/potentials/pair/parabolic_kernel.hpp>
#include <halmd/numeric/blas/blas.hpp>
#include <halmd/numeric/pow.hpp>  // std::pow is not a device function
#include <halmd/utility/tuple.hpp>
#include <halmd/mdsim/forces/trunc/local_r4.hpp>

namespace halmd {
namespace mdsim {
namespace gpu {
namespace potentials {
namespace pair {
namespace parabolic_kernel {

/** array of potential parameters for all combinations of particle types */
static texture<float4> param_;
/** squares of potential cutoff radius and energy shift for all combinations of particle types */
static texture<float2> rr_en_cut_;

/**
 * power law interaction potential of a pair of particles.
 *
 * @f[  U(r) = \epsilon (r/\sigma)^{-n} @f]
 */
class parabolic
{
public:
    /**
     * Construct power law potential.
     *
     * Fetch potential parameters from texture cache for particle pair.
     *
     * @param type1 type of first interacting particle
     * @param type2 type of second interacting particle
     */
    HALMD_GPU_ENABLED parabolic(
        unsigned int type1, unsigned int type2
      , unsigned int ntype1, unsigned int ntype2
    )
      : pair_(tex1Dfetch(param_, type1 * ntype2 + type2))
      , pair_rr_en_cut_(tex1Dfetch(rr_en_cut_, type1 * ntype2 + type2))
    {}

    /**
     * Returns square of cutoff distance.
     */
    HALMD_GPU_ENABLED float rr_cut() const
    {
        return pair_rr_en_cut_[0];
    }

    /**
     * Check whether particles are in interaction range.
     *
     * @param rr squared distance between particles
     */
    template <typename float_type>
    HALMD_GPU_ENABLED bool within_range(float_type rr) const
    {
        return (rr < pair_rr_en_cut_[0]);
    }

    /**
     * Compute force and potential for interaction.
     *
     * @param rr squared distance between particles
     * @returns tuple of unit "force" @f$ -U'(r)/r @f$ and potential @f$ U(r) @f$
     *
     * @f{eqnarray*}{
     *   - U'(r) / r &=& n r^{-2} \epsilon (r/\sigma)^{-n} \\
     *   U(r) &=& \epsilon (r/\sigma)^{-n}
     * @f}
     */
    template <typename float_type>
    HALMD_GPU_ENABLED tuple<float_type, float_type> operator()(float_type rr) const
    {
        float_type r = sqrt(rr);
        float_type one_minus_r = 1 - r /  pair_[CUTOFF];
        unsigned short n = static_cast<unsigned short>(pair_[INDEX]);

        float_type real_force = pair_[EPSILON] * halmd::pow(one_minus_r, n - 1);
        float_type fval = real_force / r;

        float_type en_pot = real_force * one_minus_r / n;

        return make_tuple(fval, en_pot);
    }

private:
    /** potential parameters for particle pair */
    fixed_vector<float, 4> pair_;
    /** squared cutoff radius and energy shift for particle pair */
    fixed_vector<float, 2> pair_rr_en_cut_;
};

} // namespace parabolic_kernel

cuda::texture<float4> parabolic_wrapper::param = parabolic_kernel::param_;
cuda::texture<float2> parabolic_wrapper::rr_en_cut = parabolic_kernel::rr_en_cut_;

} // namespace pair
} // namespace potentials

// explicit instantiation of force kernels
namespace forces {

using namespace halmd::mdsim::gpu::potentials::pair::parabolic_kernel;
using namespace halmd::mdsim::forces::trunc;

template class pair_full_wrapper<3, parabolic>;
template class pair_full_wrapper<2, parabolic>;

template class pair_trunc_wrapper<3, parabolic>;
template class pair_trunc_wrapper<2, parabolic>;
template class pair_trunc_wrapper<3, parabolic, local_r4<float> >;
template class pair_trunc_wrapper<2, parabolic, local_r4<float> >;

} // namespace forces

} // namespace gpu
} // namespace mdsim
} // namespace halmd
