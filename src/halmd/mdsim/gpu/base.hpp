/* Lennard-Jones fluid kernel
 *
 * Copyright © 2008-2009  Peter Colberg
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

#ifndef HALMD_MDSIM_GPU_LJFLUID_BASE_HPP
#define HALMD_MDSIM_GPU_LJFLUID_BASE_HPP

#include <cuda_wrapper.hpp>
#include <halmd/math/gpu/dsfloat.cuh>
#include <halmd/mdsim/impl.hpp>
#include <halmd/mdsim/variant.hpp>
#include <halmd/rng/gpu/uint48.cuh>

namespace halmd { namespace gpu
{

enum { VIRTUAL_PARTICLE = -1U };

template <typename ljfluid_impl>
struct ljfluid_base;

template <>
struct ljfluid_base<ljfluid_impl_gpu_base>
{
    static cuda::symbol<uint> npart;
    static cuda::symbol<float> box;
    static cuda::symbol<float> timestep;
    static cuda::symbol<float[]> r_cut;
    static cuda::symbol<float[]> rr_cut;
    static cuda::symbol<float[]> en_cut;
    /** binary mixture */
    static cuda::symbol<uint[]> mpart;
    static cuda::symbol<float[]> epsilon;
    static cuda::symbol<float[]> sigma2;
    /** C² potential */
    static cuda::symbol<float> rri_smooth;

    static cuda::function<void (float3*, const float2)> sample_smooth_function;
    static cuda::function<void (float3*, const float2)> sample_potential;
    static cuda::function<void (float3*, const float2)> sample_smooth_potential;
};

template <typename ljfluid_impl, int dimension>
struct ljfluid;

template <>
struct ljfluid<halmd::ljfluid_impl_gpu_base, 3>
: public ljfluid_base<ljfluid_impl_gpu_base>
{
    static cuda::function<void (float4*, unsigned int*)> init_tags;
    static cuda::function<void (float4*, dsfloat)> rescale_velocity;
};

template <>
struct ljfluid<halmd::ljfluid_impl_gpu_base, 2>
: public ljfluid_base<ljfluid_impl_gpu_base>
{
    static cuda::function<void (float4*, unsigned int*)> init_tags;
    static cuda::function<void (float2*, dsfloat)> rescale_velocity;
};

}} // namespace halmd::gpu

#endif /* ! HALMD_MDSIM_GPU_LJFLUID_BASE_HPP */
