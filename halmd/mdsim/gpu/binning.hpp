/*
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

#ifndef HALMD_MDSIM_GPU_BINNING_HPP
#define HALMD_MDSIM_GPU_BINNING_HPP

#include <halmd/config.hpp>

#include <algorithm>
#include <boost/bind.hpp>
#include <boost/make_shared.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/shared_ptr.hpp>
#include <lua.hpp>

#include <halmd/io/logger.hpp>
#include <halmd/algorithm/gpu/radix_sort.hpp>
#include <halmd/algorithm/multi_range.hpp>
#include <halmd/mdsim/box.hpp>
#include <halmd/mdsim/gpu/binning_kernel.hpp>
#include <halmd/mdsim/gpu/particle.hpp>
#include <halmd/utility/multi_index.hpp>
#include <halmd/utility/profiler.hpp>

namespace halmd {
namespace mdsim {
namespace gpu {

template <int dimension, typename float_type>
class binning
{
public:
    typedef gpu::particle<dimension, float_type> particle_type;
    typedef typename particle_type::vector_type vector_type;
    typedef boost::numeric::ublas::matrix<float_type> matrix_type;
    typedef mdsim::box<dimension> box_type;
    struct defaults;
    typedef logger logger_type;

    typedef fixed_vector<unsigned int, dimension> cell_size_type;
    typedef fixed_vector<int, dimension> cell_diff_type;

    static void luaopen(lua_State* L);

    binning(
        boost::shared_ptr<particle_type const> particle
      , boost::shared_ptr<box_type const> box
      , matrix_type const& r_cut
      , double skin
      , boost::shared_ptr<logger_type> logger = boost::make_shared<logger_type>()
      , double cell_occupancy = defaults::occupancy()
    );
    void update();

    //! returns neighbour list skin in MD units
    float_type r_skin() const
    {
        return r_skin_;
    }

    //! returns average desired cell occupancy
    float_type cell_occupancy() const
    {
        return nu_cell_;
    }

    //! returns average effective cell occupancy
    float_type effective_cell_occupancy() const
    {
        return nu_cell_eff_;
    }

    //! returns cell size
    unsigned int cell_size() const
    {
        return cell_size_;
    }

    /**
     * number of cells per dimension
     */
    cell_size_type ncell() const
    {
        return ncell_;
    }

    /**
     * cell kernel dimensions
     */
    cuda::config const& dim_cell() const
    {
        return dim_cell_;
    }

    /**
     * cell lists
     */
    cuda::vector<unsigned int> const& g_cell() const
    {
        return g_cell_;
    }

    /**
     * Copy cells to multi-range output iterator.
     *
     * @param output multi-range output iterator
     *
     * A multi-range iterator is a functor that accepts a multi-dimensional
     * index of array type, and returns an output iterator for the given
     * index. The particle indices in the cell of the given index are
     * then copied to the returned output iterator.
     */
    template <typename output_iterator>
    void get_cell(output_iterator output) const;

private:
    typedef utility::profiler profiler_type;
    typedef typename profiler_type::accumulator_type accumulator_type;
    typedef typename profiler_type::scoped_timer_type scoped_timer_type;

    struct runtime
    {
        accumulator_type update;
    };

    boost::shared_ptr<particle_type const> particle_;
    boost::shared_ptr<box_type const> box_;
    /** module logger */
    boost::shared_ptr<logger_type> logger_;
    /** neighbour list skin in MD units */
    float_type r_skin_;
    /** average desired cell occupancy */
    float_type nu_cell_;
    /** average effective cell occupancy */
    float_type nu_cell_eff_;
    /** number of cells per dimension */
    cell_size_type ncell_;
    /** number of placeholders per cell */
    unsigned int cell_size_;
    /** cell edge lengths */
    vector_type cell_length_;
    /** CUDA cell kernel execution configuration */
    cuda::config dim_cell_;
    /** cell lists in global device memory */
    cuda::vector<unsigned int> g_cell_;

    /** GPU radix sort */
    algorithm::gpu::radix_sort<unsigned int> sort_;
    /** cell indices for particles */
    cuda::vector<unsigned int> g_cell_index_;
    /** particle permutation */
    cuda::vector<unsigned int> g_cell_permutation_;
    /** cell offsets in sorted particle list */
    cuda::vector<unsigned int> g_cell_offset_;
    /** profiling runtime accumulators */
    runtime runtime_;
};

#ifndef HALMD_NO_CXX11

template <int dimension, typename float_type>
template <typename output_iterator>
inline void binning<dimension, float_type>::get_cell(output_iterator output) const
{
    cuda::host::vector<unsigned int> h_cell(g_cell_.size());
    cuda::copy(g_cell_, h_cell);
    multi_range_for_each(
        cell_size_type(0)
      , ncell_
      , [&](cell_size_type const& index) {
            auto first = h_cell.begin() + cell_size_ * multi_index_to_offset(index, ncell_);
            auto last = first + cell_size_;
            std::remove_copy(first, last, output(index), -1u);
        }
    );
}

#endif /* ! HALMD_NO_CXX11 */

template <int dimension, typename float_type>
struct binning<dimension, float_type>::defaults
{
    static float_type occupancy();
};

} // namespace mdsim
} // namespace gpu
} // namespace halmd

#endif /* ! HALMD_MDSIM_GPU_BINNING_HPP */
