/* Lennard-Jones fluid simulation using CUDA
 *
 * Copyright © 2008-2010  Peter Colberg
 *                        Felix Höfling
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

#ifndef HALMD_MDSIM_LJFLUID_GPU_NBR_HPP
#define HALMD_MDSIM_LJFLUID_GPU_NBR_HPP

#include <algorithm>
#include <boost/lexical_cast.hpp>
#include <limits>

#include <halmd/algorithm/reduce.hpp>
#include <halmd/mdsim/ljfluid_gpu_base.hpp>
#include <halmd/mdsim/gpu/hilbert.hpp>

#define foreach BOOST_FOREACH

namespace halmd
{

template <typename ljfluid_impl, int dimension>
class ljfluid;

template <int dimension>
class ljfluid<ljfluid_impl_gpu_neighbour, dimension>
    : public ljfluid_gpu_base<ljfluid_impl_gpu_neighbour, dimension>
{
public:
    typedef ljfluid_gpu_base<ljfluid_impl_gpu_neighbour, dimension> _Base;
    typedef gpu::ljfluid<ljfluid_impl_gpu_neighbour, dimension> _gpu;
    typedef typename _Base::float_type float_type;
    typedef typename _Base::vector_type vector_type;
    typedef typename _Base::gpu_vector_type gpu_vector_type;
    typedef typename _Base::host_sample_type host_sample_type;
    typedef typename _Base::gpu_sample_type gpu_sample_type;
    typedef gpu_sample_type trajectory_sample_type;
    typedef boost::variant<host_sample_type, gpu_sample_type> trajectory_sample_variant;
    typedef typename _Base::energy_sample_type energy_sample_type;
    typedef typename _Base::virial_tensor virial_tensor;

public:
    /** set number of particles in system */
    template <typename T>
    void particles(T const& value);
    /** set number of CUDA execution threads */
    void threads(unsigned int value);
    /** set desired average cell occupancy */
    void cell_occupancy(float_type value);
    /** set neighbour list skin */
    void nbl_skin(float value);

    /** restore system state from phase space sample */
    void state(host_sample_type& sample, float_type box);
    /** rescale particle velocities */
    void rescale_velocities(double coeff);
    /** place particles on a face-centered cubic (fcc) lattice */
    void lattice();
    /** set system temperature according to Maxwell-Boltzmann distribution */
    void temperature(float_type temp);

    /** MD integration step */
    void mdstep();
    /** sample phase space on host */
    void sample(host_sample_type& sample) const;
    /** sample phase space on GPU */
    void sample(gpu_sample_type& sample) const;
    /** sample thermodynamic equilibrium properties */
    void sample(energy_sample_type& sample) const;

    /** returns number of particles */
    unsigned int particles() const { return npart; }
    /** get number of CUDA execution threads */
    unsigned int threads() const { return dim_.threads_per_block(); }
    /** get effective average cell occupancy */
    float_type cell_occupancy() const { return cell_occupancy_; }
    /** get number of cells per dimension */
    unsigned int cells() const { return ncell; }
    /** get total number of cell placeholders */
    unsigned int placeholders() const { return nplace; }
    /** get cell length */
    float_type cell_length() const { return cell_length_; }
    /** get number of placeholders per cell */
    unsigned int cell_size() const { return cell_size_; }
    /** get total number of placeholders per neighbour list */
    unsigned int neighbours() const { return nbl_size; }
    /** write parameters to HDF5 parameter group */
    void param(H5param& param) const;

private:
    /** assign particle positions */
    void assign_positions();
    /** assign single-precision velocities according to particle order */
    void order_velocities();
    /** first leapfrog step of integration of differential equations of motion */
    void velocity_verlet();
    /** Lennard-Jones force calculation */
    void update_forces();
    /** assign particles to cells */
    void assign_cells();
    /** update neighbour lists */
    void update_neighbours();
    /** order particles after Hilbert space-filling curve */
    void hilbert_order();
    /** generate permutation for phase space sampling */
    void permutation();
    /** generate Maxwell-Boltzmann distributed velocities */
    void boltzmann(float temp);

private:
    using _Base::box_;
    using _Base::density_;
    using _Base::dim_;
    using _Base::m_times;
    using _Base::mixture_;
    using _Base::mpart;
    using _Base::npart;
    using _Base::potential_;
    using _Base::r_cut;
    using _Base::timestep_;
    using _Base::thermostat_steps;
    using _Base::thermostat_count;
    using _Base::thermostat_temp;

    /** CUDA execution dimensions for cell-specific kernels */
    cuda::config dim_cell_;
    /** CUDA execution dimensions for phase space sampling */
    std::vector<cuda::config> dim_sample;

    /** GPU radix sort */
    radix_sort<unsigned int> radix_;

    using _Base::reduce_squared_velocity;
    using _Base::reduce_velocity;
    using _Base::reduce_en;
    using _Base::reduce_virial;
    using _Base::reduce_helfand;

    /** number of cells per dimension */
    unsigned int ncell;
    /** total number of cell placeholders */
    unsigned int nplace;
    /** cell length */
    float_type cell_length_;
    /** effective average cell occupancy */
    float_type cell_occupancy_;
    /** number of placeholders per cell */
    unsigned int cell_size_;

    /** neighbour list skin */
    float_type r_skin;
    /** number of placeholders per neighbour list */
    unsigned int nbl_size;

    /** blockwise maximum absolute particle displacement */
    reduce<tag::max, float> reduce_r_max;
    /** maximum absolute particle displacement */
    float r_max_;
    /** upper boundary for maximum particle displacement */
    float_type r_skin_half;

    /** system state in page-locked host memory */
    struct {
        /** tagged periodically reduced particle positions */
        cuda::host::vector<float4> mutable r;
        /** periodic box traversal vectors */
        cuda::host::vector<gpu_vector_type> mutable R;
        /** particle velocities */
        cuda::host::vector<gpu_vector_type> mutable v;
        /** particle tags */
        cuda::host::vector<unsigned int> mutable tag;
    } h_part;

    /** system state in global device memory */
    struct {
        /** tagged periodically reduced particle positions */
        cuda::vector<float4> r;
        /** particle displacements */
        cuda::vector<gpu_vector_type> dr;
        /** periodic box traversal vectors */
        cuda::vector<gpu_vector_type> R;
        /** particle velocities */
        cuda::vector<gpu_vector_type> v;
        /** particle forces */
        cuda::vector<gpu_vector_type> f;
        /** particle tags */
        cuda::vector<unsigned int> tag;
        /** potential energies per particle */
        cuda::vector<float> en;
        /** virial equation sums per particle */
        cuda::vector<gpu_vector_type> virial;
        /** time integral of virial stress tensor to calculate Helfand moment */
        cuda::vector<gpu_vector_type> helfand;
    } g_part;

    /** double buffers for particle sorting */
    struct {
        /** tagged periodically reduced particle positions */
        cuda::vector<float4> r;
        /** periodic box traversal vectors */
        cuda::vector<gpu_vector_type> R;
        /** particle velocities */
        cuda::vector<gpu_vector_type> v;
    } g_part_buf;

    /** auxiliary device memory arrays for particle sorting */
    struct {
        /** particle cells */
        cuda::vector<unsigned int> cell;
        /** cell offsets in sorted particle list */
        cuda::vector<unsigned int> offset;
        /** permutation indices */
        cuda::vector<unsigned int> index;
    } g_aux;

    /** cell lists in global device memory */
    cuda::vector<unsigned int> g_cell;
    /** neighbour lists in global device memory */
    cuda::vector<unsigned int> g_nbl;

    /** CUDA kernel return values */
    struct {
        cuda::vector<unsigned int> assign_cells;
        cuda::vector<unsigned int> update_neighbours;
    } g_ret;
    struct {
        cuda::host::vector<unsigned int> assign_cells;
        cuda::host::vector<unsigned int> update_neighbours;
    } h_ret;
};

template <int dimension>
template <typename T>
void ljfluid<ljfluid_impl_gpu_neighbour, dimension>::particles(T const& value)
{
    _Base::particles(value);

    // allocate page-locked host memory for system state
    try {
        h_part.r.resize(npart);
        h_part.R.resize(npart);
        h_part.v.resize(npart);
        h_part.tag.resize(npart);
        // particle forces reside only in GPU memory
    }
    catch (cuda::error const& e) {
        LOG_ERROR("CUDA: " << e.what());
        throw exception("failed to allocate page-locked host memory for system state");
    }
}

template <int dimension>
void ljfluid<ljfluid_impl_gpu_neighbour, dimension>::cell_occupancy(float_type value)
{
    cell_occupancy_ = value;
    LOG("desired average cell occupancy: " << cell_occupancy_);
}

template <int dimension>
void ljfluid<ljfluid_impl_gpu_neighbour, dimension>::nbl_skin(float value)
{
    float r_cut_max = *std::max_element(r_cut.begin(), r_cut.end());
#ifndef __DEVICE_EMULATION__
    cuda::device::properties prop(cuda::driver::context::device());
#else
    cuda::device::properties prop(cuda::device::get());
#endif

    for (unsigned int i = prop.warp_size(); i <= prop.max_threads_per_block(); i += prop.warp_size()) {
        // number of placeholders per cell
        cell_size_ = i;
        // optimal number of cells with given cell occupancy as upper boundary
        ncell = static_cast<unsigned int>(std::ceil(std::pow(npart / (cell_occupancy_ * cell_size_), 1.f / dimension)));
        // set number of cells per dimension, respecting cutoff radius
        ncell = std::min(ncell, static_cast<unsigned int>(box_ / r_cut_max));
        // derive cell length from number of cells
        cell_length_ = box_ / ncell;

        if (value < (cell_length_ - r_cut_max)) {
            // cell size is adequately large
            break;
        }
    }

    LOG("number of placeholders per cell: " << cell_size_);
    LOG("number of cells per dimension: " << ncell);

    if (ncell < 3) {
        throw exception("number of cells per dimension must be at least 3");
    }

    LOG("cell length: " << cell_length_);
    // set total number of cell placeholders
    nplace = static_cast<unsigned int>(pow(ncell, dimension)) * cell_size_;
    LOG("total number of cell placeholders: " << nplace);
    // set effective average cell occupancy
    cell_occupancy_ = npart * 1.f / nplace;
    // round up this value by ULP, so that setting an exact desired cell
    // occupancy will yield the exact same effective cell occupancy,
    // e.g. if reading the value from a HDF5 attribute
    cell_occupancy_ *= (1.f + std::numeric_limits<float>::epsilon());
    LOG("effective average cell occupancy: " << cell_occupancy_);

    if (cell_occupancy_ > 1) {
        throw exception("average cell occupancy must not be larger than 1.0");
    }
    else if (cell_occupancy_ > 0.5f) {
        LOG_WARNING("average cell occupancy is larger than 0.5");
    }

    try {
        cuda::copy(ncell, _gpu::ncell);
    }
    catch (cuda::error const& e) {
        LOG_ERROR("CUDA: " << e.what());
        throw exception("failed to copy cell parameters to device symbols");
    }

    r_skin = std::min(value, cell_length_ - r_cut_max);

    if (r_skin < value) {
        LOG_WARNING("reducing neighbour list skin to fixed-size cell skin");
    }
    LOG("neighbour list skin: " << r_skin);

    // upper boundary for maximum particle displacement since last neighbour list update
    r_skin_half = r_skin / 2;

    float const r_nbl = r_cut_max + r_skin;
    // volume of n-dimensional sphere with neighbour list radius
    float const v_nbl = ((dimension + 1) * M_PI / 3) * std::pow(r_nbl, dimension);

    nbl_size = static_cast<unsigned int>(std::ceil(v_nbl * (density_ / cell_occupancy_)));
    LOG("number of placeholders per neighbour list: " << nbl_size);

    try {
        cuda::copy(std::pow(r_nbl, 2), _gpu::rr_nbl);
        cuda::copy(nbl_size, _gpu::nbl_size);
    }
    catch (cuda::error const& e) {
        LOG_ERROR("CUDA: " << e.what());
        throw exception("failed to copy neighbour list parameters to device symbols");
    }

#if defined(USE_HILBERT_ORDER)
    // set Hilbert space-filling curve recursion depth
    unsigned int depth = static_cast<unsigned int>(ceilf(logf(box_) / M_LN2));
    // 32-bit integer for 2D Hilbert code allows a maximum of 16/10 levels
    depth = std::min((dimension == 3) ? 10U : 16U, depth);

    LOG("Hilbert space-filling curve recursion depth: " << depth);

    try {
        cuda::copy(box_, gpu::hilbert<dimension>::box);
        cuda::copy(depth, gpu::hilbert<dimension>::depth);
    }
    catch (cuda::error const& e) {
        LOG_ERROR("CUDA: " << e.what());
        throw exception("failed to copy Hilbert curve recursion depth to device symbol");
    }
#endif
}

template <int dimension>
void ljfluid<ljfluid_impl_gpu_neighbour, dimension>::threads(unsigned int value)
{
    _Base::threads(value);

    // set CUDA execution dimensions for cell-specific kernels
    dim_cell_ = cuda::config(dim3(static_cast<unsigned int>(powf(ncell, dimension - 1)), ncell), cell_size_);
    LOG("number of cell CUDA execution blocks: " << dim_cell_.blocks_per_grid());
    LOG("number of cell CUDA execution threads: " << dim_cell_.threads_per_block());

    // allocate global device memory for placeholder particles
    try {
#ifdef USE_VERLET_DSFUN
        LOG("using double-single arithmetic in Verlet integration");
        g_part.r.reserve(2 * dim_.threads());
        g_part.v.reserve(2 * dim_.threads());
#else
        g_part.r.reserve(dim_.threads());
        g_part.v.reserve(dim_.threads());
#endif
        g_part.r.resize(npart);
        g_part.v.resize(npart);
        g_part.dr.reserve(dim_.threads());
        g_part.dr.resize(npart);
        g_part.R.reserve(dim_.threads());
        g_part.R.resize(npart);
        g_part.f.reserve(dim_.threads());
        g_part.f.resize(npart);
        g_part.tag.reserve(dim_.threads());
        g_part.tag.resize(npart);
        g_part.en.reserve(dim_.threads());
        g_part.en.resize(npart);
        g_part.virial.reserve(dim_.threads());
        g_part.virial.resize(npart);
        g_part.helfand.reserve(dim_.threads());
        g_part.helfand.resize(npart);
    }
    catch (cuda::error const& e) {
        LOG_ERROR("CUDA: " << e.what());
        throw exception("failed to allocate global device memory for system state");
    }

    // allocate global device memory for cell placeholders
    try {
        g_cell.resize(dim_cell_.threads());
        g_nbl.resize(dim_.threads() * nbl_size);
        cuda::copy(g_nbl.data(), _gpu::g_nbl);
        cuda::copy(static_cast<unsigned int>(dim_.threads()), _gpu::nbl_stride);
    }
    catch (cuda::error const& e) {
        LOG_ERROR("CUDA: " << e.what());
        throw exception("failed to allocate global device memory cell placeholders");
    }

    // bind GPU textures to global device memory arrays
    try {
        _gpu::r.bind(g_part.r);
        _gpu::v.bind(g_part.v);
        _gpu::R.bind(g_part.R);
    }
    catch (cuda::error const& e) {
        LOG_ERROR("CUDA: " << e.what());
        throw exception("failed to bind GPU textures to global device memory arrays");
    }

    // allocate global device memory for sorting buffers
    try {
        g_part_buf.r.reserve(g_part.r.capacity());
        g_part_buf.r.resize(npart);
        g_part_buf.R.reserve(g_part.R.capacity());
        g_part_buf.R.resize(npart);
        g_part_buf.v.reserve(g_part.v.capacity());
        g_part_buf.v.resize(npart);
        g_aux.cell.reserve(dim_.threads());
        g_aux.cell.resize(npart);
        g_aux.offset.resize(dim_cell_.blocks_per_grid());
        g_aux.index.reserve(dim_.threads());
        // allocate sufficient memory for binary mixture sampling
        for (size_t n = 0, i = 0; n < npart; n += mpart[i], ++i) {
            cuda::config dim((mpart[i] + threads() - 1) / threads(), threads());
            g_aux.index.reserve(n + dim.threads());
            dim_sample.push_back(dim);
        }
        g_aux.index.resize(npart);
    }
    catch (cuda::error const& e) {
        LOG_ERROR("CUDA: " << e.what());
        throw exception("failed to allocate global device memory for sorting buffers");
    }


    try {
        radix_.resize(npart, dim_.threads_per_block());
    }
    catch (cuda::error const& e) {
        LOG_ERROR("CUDA: " << e.what());
        throw exception("failed to allocate global device memory for radix sort");
    }

    g_ret.assign_cells.resize(1);
    h_ret.assign_cells.resize(1);
    g_ret.update_neighbours.resize(1);
    h_ret.update_neighbours.resize(1);
}

template <int dimension>
void ljfluid<ljfluid_impl_gpu_neighbour, dimension>::state(host_sample_type& sample, float_type box)
{
    _Base::state(sample, box, h_part.r, h_part.v);
#ifdef USE_VERLET_DSFUN
    cuda::memset(g_part.r, 0, g_part.r.capacity());
#endif
    cuda::copy(h_part.r, g_part.r);
    assign_positions();
#ifdef USE_VERLET_DSFUN
    cuda::memset(g_part.v, 0, g_part.v.capacity());
#endif
    cuda::copy(h_part.v, g_part.v);
#ifdef USE_HILBERT_ORDER
    order_velocities();
#endif
    // init accumulator for Helfand moment
    cuda::memset(g_part.helfand, 0, g_part.helfand.capacity());
}

template <int dimension>
void ljfluid<ljfluid_impl_gpu_neighbour, dimension>::rescale_velocities(double coeff)
{
    LOG("rescaling velocities with coefficient: " << coeff);
    _Base::rescale_velocities(g_part.v, coeff, dim_);
}

template <int dimension>
void ljfluid<ljfluid_impl_gpu_neighbour, dimension>::lattice()
{
#ifdef USE_VERLET_DSFUN
    cuda::memset(g_part.r, 0, g_part.r.capacity());
#endif
    // place particles on an fcc lattice
    _Base::lattice(g_part.r);
    // randomly permute particle coordinates for binary mixture
    _Base::random_permute(g_part.r);
    // init accumulator for Helfand moment
    cuda::memset(g_part.helfand, 0, g_part.helfand.capacity());

    assign_positions();
}

template <int dimension>
void ljfluid<ljfluid_impl_gpu_neighbour, dimension>::temperature(float_type temp)
{
    LOG("initialising velocities from Boltzmann distribution at temperature: " << temp);

    boost::array<high_resolution_timer, 2> timer;
    cuda::thread::synchronize();
    try {
        timer[0].record();
        boltzmann(temp);
        cuda::thread::synchronize();
        timer[1].record();
    }
    catch (cuda::error const& e) {
        LOG_ERROR("CUDA: " << e.what());
        throw exception("failed to compute Boltzmann distributed velocities on GPU");
    }
    m_times["boltzmann"] += timer[1] - timer[0];
}

/**
 * MD integration step on GPU
 */
template <int dimension>
void ljfluid<ljfluid_impl_gpu_neighbour, dimension>::mdstep()
{
    boost::array<high_resolution_timer, 10> timer;
    cuda::thread::synchronize();
    timer[1].record();

    // first leapfrog step of integration of differential equations of motion
    try {
        velocity_verlet();
        cuda::thread::synchronize();
    }
    catch (cuda::error const& e) {
        LOG_ERROR("CUDA: " << e.what());
        throw exception("failed to stream first leapfrog step on GPU");
    }
    timer[2].record();

    // maximum absolute particle displacement reduction
    try {
        reduce_r_max(g_part.dr);
        cuda::thread::synchronize();
    }
    catch (cuda::error const& e) {
        LOG_ERROR("CUDA: " << e.what());
        throw exception("failed to stream maximum particle displacement reduction on GPU");
    }
    timer[3].record();

    // update cell lists
    if ((r_max_ = reduce_r_max.value()) > r_skin_half) {
#ifdef USE_HILBERT_ORDER
        try {
            hilbert_order();
            cuda::thread::synchronize();
        }
        catch (cuda::error const& e) {
            LOG_ERROR("CUDA: " << e.what());
            throw exception("failed to stream hilbert space-filling curve sort on GPU");
        }
#endif
        timer[4].record();

        try {
            assign_cells();
            cuda::thread::synchronize();
        }
        catch (cuda::error const& e) {
            LOG_ERROR("CUDA: " << e.what());
            throw exception("failed to stream cell list update on GPU");
        }
        timer[5].record();

        try {
            update_neighbours();
            cuda::thread::synchronize();
        }
        catch (cuda::error const& e) {
            LOG_ERROR("CUDA: " << e.what());
            throw exception("failed to stream neighbour lists update on GPU");
        }
        timer[6].record();

#ifdef USE_HILBERT_ORDER
        try {
            permutation();
            cuda::thread::synchronize();
        }
        catch (cuda::error const& e) {
            LOG_ERROR("CUDA: " << e.what());
            throw exception("failed to generate permutation for phase space sampling");
        }
#endif
    }
    timer[7].record();

    // Lennard-Jones force calculation
    try {
        update_forces();
        cuda::thread::synchronize();
    }
    catch (cuda::error const& e) {
        LOG_ERROR("CUDA: " << e.what());
        throw exception("failed to stream force calculation on GPU");
    }
    timer[8].record();

    // heat bath coupling
    if (thermostat_steps && ++thermostat_count > thermostat_steps) {
        try {
            boltzmann(thermostat_temp);
            cuda::thread::synchronize();
        }
        catch (cuda::error const& e) {
            LOG_ERROR("CUDA: " << e.what());
            throw exception("failed to compute Boltzmann distributed velocities on GPU");
        }
    }
    timer[9].record();

    // potential energy sum calculation
    try {
        reduce_en(g_part.en);
        cuda::thread::synchronize();
    }
    catch (cuda::error const& e) {
        LOG_ERROR("CUDA: " << e.what());
        throw exception("failed to stream potential energy sum calculation on GPU");
    }
    timer[0].record();

    // CUDA time for MD integration step
    m_times["mdstep"] += timer[0] - timer[1];
    // GPU time for velocity-Verlet integration
    m_times["velocity_verlet"] += timer[2] - timer[1];
    // GPU time for maximum velocity calculation
    m_times["maximum_displacement"] += timer[3] - timer[2];

    if (r_max_ > r_skin_half) {
#ifdef USE_HILBERT_ORDER
        // GPU time for Hilbert curve sort
        m_times["hilbert_sort"] += timer[4] - timer[3];
#endif
        // GPU time for cell lists update
        m_times["update_cells"] += timer[5] - timer[4];
        // GPU time for neighbour lists update
        m_times["update_neighbours"] += timer[6] - timer[5];
#if defined(USE_HILBERT_ORDER)
        // GPU time for permutation sort
        m_times["permutation"] += timer[7] - timer[6];
#endif
    }

    // GPU time for Lennard-Jones force update
    m_times["update_forces"] += timer[8] - timer[7];

    if (thermostat_steps && thermostat_count > thermostat_steps) {
        // reset MD steps since last heatbath coupling
        thermostat_count = 0;
        // GPU time for Maxwell-Boltzmann distribution
        m_times["boltzmann"] += timer[9] - timer[8];
    }

    // GPU time for potential energy sum calculation
    m_times["potential_energy"] += timer[0] - timer[9];

    if (r_max_ > r_skin_half) {
        if (h_ret.assign_cells.front() != EXIT_SUCCESS) {
            throw exception("overcrowded placeholders in cell lists update");
        }
        if (h_ret.update_neighbours.front() != EXIT_SUCCESS) {
            throw exception("overcrowded placeholders in neighbour lists update");
        }
    }
    if (!std::isfinite(reduce_en.value())) {
        throw potential_energy_divergence();
    }
}

template <int dimension>
void ljfluid<ljfluid_impl_gpu_neighbour, dimension>::sample(host_sample_type& sample) const
{
    typedef typename host_sample_type::value_type sample_type;
    typedef typename sample_type::position_sample_vector position_sample_vector;
    typedef typename sample_type::position_sample_ptr position_sample_ptr;
    typedef typename sample_type::velocity_sample_vector velocity_sample_vector;
    typedef typename sample_type::velocity_sample_ptr velocity_sample_ptr;

    boost::array<high_resolution_timer, 2> timer;
    cuda::thread::synchronize();
    timer[0].record();
    try {
        cuda::copy(g_part.r, h_part.r);
        cuda::copy(g_part.R, h_part.R);
        cuda::copy(g_part.v, h_part.v);
        cuda::copy(g_part.tag, h_part.tag);
    }
    catch (cuda::error const& e) {
        LOG_ERROR("CUDA: " << e.what());
        throw exception("failed to copy MD integration step results from GPU to host");
    }
    timer[1].record();
    m_times["sample_memcpy"] += timer[1] - timer[0];

    // allocate memory for phase space sample
    for (size_t n = 0, i = 0; n < npart; n += mpart[i], ++i) {
        position_sample_ptr r(new position_sample_vector(mpart[i]));
        velocity_sample_ptr v(new velocity_sample_vector(mpart[i]));
        sample.push_back(sample_type(r, v));
    }

    // copy particle positions and velocities in binary mixture
    for (size_t i = 0, tag; i < h_part.tag.size(); ++i) {
        if ((tag = h_part.tag[i]) != gpu::VIRTUAL_PARTICLE) {
            unsigned int const type = (tag >= mpart[0]);
            unsigned int const n = type ? (tag - mpart[0]) : tag;
            (*sample[type].r)[n] = h_part.r[i] + box_ * static_cast<vector_type>(h_part.R[i]);
            (*sample[type].v)[n] = h_part.v[i];
        }
    }
}

template <int dimension>
void ljfluid<ljfluid_impl_gpu_neighbour, dimension>::sample(gpu_sample_type& sample) const
{
    typedef typename gpu_sample_type::value_type sample_type;
    typedef typename sample_type::position_sample_vector position_sample_vector;
    typedef typename sample_type::position_sample_ptr position_sample_ptr;
    typedef typename sample_type::velocity_sample_vector velocity_sample_vector;
    typedef typename sample_type::velocity_sample_ptr velocity_sample_ptr;

    boost::array<high_resolution_timer, 2> timer;
    cuda::thread::synchronize();
    timer[0].record();

    for (size_t n = 0, i = 0; n < npart; n += mpart[i], ++i) {
        // allocate global device memory for phase space sample
        position_sample_ptr r(new position_sample_vector);
        r->reserve(dim_sample[i].threads());
        r->resize(mpart[i]);
        velocity_sample_ptr v(new velocity_sample_vector);
        v->reserve(dim_sample[i].threads());
        v->resize(mpart[i]);
        sample.push_back(sample_type(r, v));
        // order particles by permutation
        cuda::configure(dim_sample[i].grid, dim_sample[i].block);
        _gpu::sample(g_aux.index.data() + n, *r, *v);
    }

    cuda::thread::synchronize();
    timer[1].record();
    m_times["sample"] += timer[1] - timer[0];
}

template <int dimension>
void ljfluid<ljfluid_impl_gpu_neighbour, dimension>::sample(energy_sample_type& sample) const
{
    boost::array<high_resolution_timer, 2> timer;
    cuda::thread::synchronize();

    // mean potential energy per particle
    sample.en_pot = reduce_en.value() / npart;

    // virial tensor trace and off-diagonal elements for particle species
    try {
        timer[0].record();
        if (mixture_ == BINARY) {
            reduce_virial(g_part.virial, g_part.tag, mpart);
            reduce_helfand(g_part.helfand, g_part.tag, mpart);
        }
        else {
            reduce_virial(g_part.virial);
            reduce_helfand(g_part.helfand);
        }
        cuda::thread::synchronize();
        timer[1].record();
        m_times["virial_sum"] += timer[1] - timer[0];
    }
    catch (cuda::error const& e) {
        LOG_ERROR("CUDA: " << e.what());
        throw exception("failed to calculate virial equation sum on GPU");
    }
    sample.virial = reduce_virial.value();
    for (size_t i = 0; i < sample.virial.size(); ++i) {
        sample.virial[i] /= mpart[i];
    }
    sample.helfand = reduce_helfand.value();
    for (size_t i = 0; i < sample.helfand.size(); ++i) {
        sample.helfand[i] /= mpart[i];
    }

    // mean squared velocity per particle
    try {
        timer[0].record();
        reduce_squared_velocity(g_part.v);
        cuda::thread::synchronize();
        timer[1].record();
        m_times["reduce_squared_velocity"] += timer[1] - timer[0];
    }
    catch (cuda::error const& e) {
        LOG_ERROR("CUDA: " << e.what());
        throw exception("failed to calculate mean squared velocity on GPU");
    }
    sample.vv = reduce_squared_velocity.value() / npart;

    // mean velocity per particle
    try {
        timer[0].record();
        reduce_velocity(g_part.v);
        cuda::thread::synchronize();
        timer[1].record();
        m_times["reduce_velocity"] += timer[1] - timer[0];
    }
    catch (cuda::error const& e) {
        LOG_ERROR("CUDA: " << e.what());
        throw exception("failed to calculate mean velocity on GPU");
    }
    sample.v_cm = reduce_velocity.value() / npart;
}

template <int dimension>
void ljfluid<ljfluid_impl_gpu_neighbour, dimension>::assign_positions()
{
    // assign ascending particle numbers
    _Base::init_tags(g_part.r, g_part.tag);

    try {
        // set periodic box traversal vectors to zero
        cuda::memset(g_part.R, 0);
#ifdef USE_HILBERT_ORDER
        // order particles after Hilbert space-filling curve
        hilbert_order();
#endif
        // assign particles to cells
        assign_cells();
        // update neighbour lists
        update_neighbours();
        // generate permutation for phase space sampling
        permutation();
        // calculate forces
        update_forces();
        // calculate potential energy
        reduce_en(g_part.en);

        // wait for CUDA operations to finish
        cuda::thread::synchronize();
    }
    catch (cuda::error const& e) {
        LOG_ERROR("CUDA: " << e.what());
        throw exception("failed to assign particle positions on GPU");
    }

    if (h_ret.assign_cells.front() != EXIT_SUCCESS) {
        throw exception("overcrowded placeholders in cell lists update");
    }
    if (h_ret.update_neighbours.front() != EXIT_SUCCESS) {
        throw exception("overcrowded placeholders in neighbour lists update");
    }
}

template <int dimension>
void ljfluid<ljfluid_impl_gpu_neighbour, dimension>::velocity_verlet()
{
    cuda::configure(dim_.grid, dim_.block);
    _gpu::inteq(g_part.r, g_part.dr, g_part.R, g_part.v, g_part.f, g_part.virial);
}

template <int dimension>
void ljfluid<ljfluid_impl_gpu_neighbour, dimension>::update_forces()
{
    cuda::configure(dim_.grid, dim_.block);
    _Base::update_forces(g_part.r, g_part.v, g_part.f, g_part.en, g_part.virial, g_part.helfand);
}

template <int dimension>
void ljfluid<ljfluid_impl_gpu_neighbour, dimension>::assign_cells()
{
    // compute cell indices for particle positions
    cuda::configure(dim_.grid, dim_.block);
    _gpu::compute_cell(g_part.r, g_aux.cell);

    // generate permutation
    cuda::configure(dim_.grid, dim_.block);
    _gpu::gen_index(g_aux.index);
    radix_(g_aux.cell, g_aux.index);

    // compute global cell offsets in sorted particle list
    cuda::memset(g_aux.offset, 0xFF);
    cuda::configure(dim_.grid, dim_.block);
    _gpu::find_cell_offset(g_aux.cell, g_aux.offset);

    // assign particles to cells
    cuda::memset(g_ret.assign_cells, EXIT_SUCCESS);
    cuda::configure(dim_cell_.grid, dim_cell_.block);
    _gpu::assign_cells(g_ret.assign_cells, g_aux.cell, g_aux.offset, g_aux.index, g_cell);
    cuda::copy(g_ret.assign_cells, h_ret.assign_cells);
}

template <int dimension>
void ljfluid<ljfluid_impl_gpu_neighbour, dimension>::update_neighbours()
{
    // set particle displacements to floating-point zero
    cuda::memset(g_part.dr, 0);
    // mark neighbour list placeholders as virtual particles
    cuda::memset(g_nbl, 0xFF);
    // reset return value
    cuda::memset(g_ret.update_neighbours, EXIT_SUCCESS);
    // build neighbour lists
    cuda::configure(dim_cell_.grid, dim_cell_.block, cell_size_ * (dimension + 1) * sizeof(float));
    _gpu::update_neighbours(g_ret.update_neighbours, g_cell);
    cuda::copy(g_ret.update_neighbours, h_ret.update_neighbours);
}

template <int dimension>
void ljfluid<ljfluid_impl_gpu_neighbour, dimension>::hilbert_order()
{
    // compute Hilbert space-filling curve for particles
    cuda::configure(dim_.grid, dim_.block);
    gpu::hilbert<dimension>::curve(g_part.r, g_aux.cell);

    // generate permutation
    cuda::configure(dim_.grid, dim_.block);
    _gpu::gen_index(g_aux.index);
    radix_(g_aux.cell, g_aux.index);

    // order particles by permutation
    cuda::configure(dim_.grid, dim_.block);
    _gpu::order_particles(g_aux.index, g_part_buf.r, g_part_buf.R, g_part_buf.v, g_part.tag);
    cuda::copy(g_part_buf.r, g_part.r, g_part.r.capacity());
    cuda::copy(g_part_buf.R, g_part.R, g_part.R.capacity());
    cuda::copy(g_part_buf.v, g_part.v, g_part.v.capacity());
}

/**
 * generate permutation for phase space sampling
 */
template <int dimension>
void ljfluid<ljfluid_impl_gpu_neighbour, dimension>::permutation()
{
    cuda::configure(dim_.grid, dim_.block);
    _gpu::gen_index(g_aux.index);
#ifdef USE_HILBERT_ORDER
    cuda::copy(g_part.tag, g_aux.cell);
    radix_(g_aux.cell, g_aux.index);
#endif
}

template <int dimension>
void ljfluid<ljfluid_impl_gpu_neighbour, dimension>::boltzmann(float temp)
{
#ifdef USE_VERLET_DSFUN
    cuda::memset(g_part.v, 0, g_part.v.capacity());
#endif
    _Base::boltzmann(g_part.v, temp);

#ifdef USE_HILBERT_ORDER
    // make thermostat independent of neighbour list update frequency or skin
    order_velocities();
#endif
}

/**
 * assign single-precision velocities according to particle order
 */
template <int dimension>
void ljfluid<ljfluid_impl_gpu_neighbour, dimension>::order_velocities()
{
    cuda::configure(dim_.grid, dim_.block);
    _gpu::order_velocities(g_part.tag, g_part_buf.v);
    cuda::copy(g_part_buf.v, g_part.v);
}

template <int dimension>
void ljfluid<ljfluid_impl_gpu_neighbour, dimension>::param(H5param& param) const
{
    _Base::param(param);

    H5xx::group node(param["mdsim"]);
    node["cells"] = ncell;
    node["placeholders"] = nplace;
    node["neighbours"] = nbl_size;
    node["cell_length"] = cell_length_;
    node["cell_occupancy"] = cell_occupancy_;
    node["neighbour_skin"] = r_skin;
}

} // namespace halmd

#undef foreach

#endif /* ! HALMD_MDSIM_LJFLUID_GPU_NBR_HPP */