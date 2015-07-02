/*
 * Copyright © 2008-2013 Felix Höfling
 * Copyright © 2008-2011 Peter Colberg
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

#include <boost/numeric/ublas/io.hpp>
#include <cuda_wrapper/cuda_wrapper.hpp>
#include <cmath>
#include <stdexcept>
#include <string>

#include <halmd/mdsim/forces/trunc/local_r4.hpp>
#include <halmd/mdsim/gpu/forces/pair_full.hpp>
#include <halmd/mdsim/gpu/forces/pair_trunc.hpp>
#include <halmd/mdsim/gpu/potentials/pair/parabolic.hpp>
#include <halmd/mdsim/gpu/potentials/pair/parabolic_kernel.hpp>
#include <halmd/utility/lua/lua.hpp>

namespace halmd {
namespace mdsim {
namespace gpu {
namespace potentials {
namespace pair {

template <typename T, typename S>
static T const&
check_shape(T const& m1, S const& m2)
{
    if (m1.size1() != m2.size1() || m1.size2() != m2.size2()) {
        throw std::invalid_argument("parameter matrix has invalid shape");
    }
    return m1;
}

/**
 * Initialise parabolic potential parameters
 */
template <typename float_type>
parabolic<float_type>::parabolic(
    matrix_type const& cutoff
  , matrix_type const& epsilon
  , uint_matrix_type const& index
  , std::shared_ptr<logger> logger
)
  // allocate potential parameters
  : epsilon_(epsilon)
  , index_(check_shape(index, epsilon))
  , r_cut_(check_shape(cutoff, epsilon))
  , rr_cut_(element_prod(r_cut_, r_cut_))
  , en_cut_(size1(), size2())
  , g_param_(size1() * size2())
  , g_rr_en_cut_(size1() * size2())
  , logger_(logger)
{
    // energy shift due to truncation at cutoff length
    for (unsigned i = 0; i < en_cut_.size1(); ++i) {
        for (unsigned j = 0; j < en_cut_.size2(); ++j) {
            en_cut_(i, j) = 0;
        }
    }

    LOG("interaction strength: ε = " << epsilon_);
    LOG("parabolic index: n = " << index_);
    LOG("cutoff length: r_c = " << r_cut_);
    LOG("cutoff energy: U = " << en_cut_);

    cuda::host::vector<float4> param(g_param_.size());
    for (size_t i = 0; i < param.size(); ++i) {
        fixed_vector<float, 4> p(0); // initialise unused elements as well
        p[parabolic_kernel::EPSILON] = epsilon_.data()[i];
        p[parabolic_kernel::INDEX] = index_.data()[i];
        p[parabolic_kernel::CUTOFF] = r_cut_.data()[i];
        param[i] = p;
    }
    cuda::copy(param, g_param_);

    cuda::host::vector<float2> rr_en_cut(g_rr_en_cut_.size());
    for (size_t i = 0; i < rr_en_cut.size(); ++i) {
        rr_en_cut[i].x = rr_cut_.data()[i];
        rr_en_cut[i].y = en_cut_.data()[i];
    }
    cuda::copy(rr_en_cut, g_rr_en_cut_);
}

template <typename float_type>
void parabolic<float_type>::luaopen(lua_State* L)
{
    using namespace luaponte;
    module(L, "libhalmd")
    [
        namespace_("mdsim")
        [
            namespace_("gpu")
            [
                namespace_("potentials")
                [
                    namespace_("pair")
                    [
                        class_<parabolic, std::shared_ptr<parabolic> >("parabolic")
                            .def(constructor<
                                matrix_type const&
                              , matrix_type const&
                              , uint_matrix_type const&
                              , std::shared_ptr<logger>
                            >())
                            .property("r_cut", (matrix_type const& (parabolic::*)() const) &parabolic::r_cut)
                            .property("epsilon", &parabolic::epsilon)
                            .property("index", &parabolic::index)
                    ]
                ]
            ]
        ]
    ];
}

HALMD_LUA_API int luaopen_libhalmd_mdsim_gpu_potentials_pair_parabolic(lua_State* L)
{
    parabolic<float>::luaopen(L);
    forces::pair_full<3, float, parabolic<float> >::luaopen(L);
    forces::pair_full<2, float, parabolic<float> >::luaopen(L);
    forces::pair_trunc<3, float, parabolic<float> >::luaopen(L);
    forces::pair_trunc<2, float, parabolic<float> >::luaopen(L);
    forces::pair_trunc<3, float, parabolic<float>, mdsim::forces::trunc::local_r4<float> >::luaopen(L);
    forces::pair_trunc<2, float, parabolic<float>, mdsim::forces::trunc::local_r4<float> >::luaopen(L);
    return 0;
}

// explicit instantiation
template class parabolic<float>;

} // namespace pair
} // namespace potentials

namespace forces {

// explicit instantiation of force modules
template class pair_full<3, float, potentials::pair::parabolic<float> >;
template class pair_full<2, float, potentials::pair::parabolic<float> >;
template class pair_trunc<3, float, potentials::pair::parabolic<float> >;
template class pair_trunc<2, float, potentials::pair::parabolic<float> >;
template class pair_trunc<3, float, potentials::pair::parabolic<float>, mdsim::forces::trunc::local_r4<float> >;
template class pair_trunc<2, float, potentials::pair::parabolic<float>, mdsim::forces::trunc::local_r4<float> >;

} // namespace forces
} // namespace gpu
} // namespace mdsim
} // namespace halmd
