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

#ifndef HALMD_MDSIM_GPU_FORCES_LENNARD_JONES_SIMPLE_KERNEL_HPP
#define HALMD_MDSIM_GPU_FORCES_LENNARD_JONES_SIMPLE_KERNEL_HPP

#include <cuda_wrapper/cuda_wrapper.hpp>

namespace halmd {
namespace mdsim {
namespace gpu {
namespace forces {
namespace lennard_jones_simple_kernel {

// forward declaration for host code
class lennard_jones_simple;

} // namespace lennard_jones_kernel

struct lennard_jones_simple_wrapper
{
    /** Lennard-Jones potential parameters */
    static cuda::symbol<float> rr_cut;
    static cuda::symbol<float> en_cut;
};

} // namespace mdsim
} // namespace gpu
} // namespace forces
} // namespace halmd

#endif /* ! HALMD_MDSIM_GPU_FORCES_LENNARD_JONES_KERNEL_HPP */
