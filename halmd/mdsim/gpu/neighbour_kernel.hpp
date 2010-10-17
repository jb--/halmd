/*
 * Copyright © 2008-2010  Peter Colberg
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

#ifndef HALMD_MDSIM_GPU_NEIGHBOUR_KERNEL_HPP
#define HALMD_MDSIM_GPU_NEIGHBOUR_KERNEL_HPP

#include <cuda_wrapper/cuda_wrapper.hpp>
#include <halmd/mdsim/type_traits.hpp>

namespace halmd
{
namespace mdsim { namespace gpu
{

template <int dimension>
struct neighbour_wrapper
{
    typedef typename type_traits<dimension, float>::gpu::vector_type vector_type;
    typedef typename type_traits<dimension, unsigned int>::gpu::vector_type cell_index_type;
    typedef cuda::function<void (float4* g_r, float4* g_r0, float* g_rr)> displacement_impl_type;

    /** (cutoff lengths + neighbour list skin)² */
    cuda::texture<float> rr_cut_skin;
    /** number of cells per dimension */
    cuda::symbol<cell_index_type> ncell;
    /** neighbour list length */
    cuda::symbol<unsigned int> neighbour_size;
    /** neighbour list stride */
    cuda::symbol<unsigned int> neighbour_stride;
    /** number of particles in simulation box */
    cuda::symbol<unsigned int> nbox;
    /** positions, tags */
    cuda::texture<float4> r;
    /** cubic box edgle length */
    cuda::symbol<vector_type> box_length;
    /** cell edge lengths */
    cuda::symbol<vector_type> cell_length;
    /** assign particles to cells */
    cuda::function<void (int*, unsigned int const*, unsigned int const*, unsigned int const*, unsigned int*)> assign_cells;
    /** compute global cell offsets in particle list */
    cuda::function<void (unsigned int*, unsigned int*)> find_cell_offset;
    /** generate ascending index sequence */
    cuda::function<void (unsigned int*)> gen_index;
    /** update neighbour lists */
    cuda::function<void (int*, unsigned int*, unsigned int const*)> update_neighbours;
    /** compute cell indices for particle positions */
    cuda::function<void (float4 const*, unsigned int*)> compute_cell;
    /** maximum squared particle distance */
    displacement_impl_type displacement_impl[5];

    static neighbour_wrapper kernel;
};

template <int dimension>
neighbour_wrapper<dimension> const& get_neighbour_kernel()
{
    return neighbour_wrapper<dimension>::kernel;
}

}} // namespace mdsim::gpu

} // namespace halmd

#endif /* ! HALMD_MDSIM_GPU_NEIGHBOUR_KERNEL_HPP */