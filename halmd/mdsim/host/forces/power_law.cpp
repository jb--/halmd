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

#include <boost/numeric/ublas/io.hpp>
#include <cmath>
#include <string>

#include <halmd/io/logger.hpp>
#include <halmd/io/utility/hdf5.hpp>
#include <halmd/mdsim/host/forces/power_law.hpp>
#include <halmd/utility/lua_wrapper/lua_wrapper.hpp>

using namespace boost;
using namespace boost::numeric::ublas;
using namespace std;

namespace halmd
{
namespace mdsim { namespace host { namespace forces
{

/**
 * Write module parameters to HDF5 group
 */
template <typename float_type>
void power_law<float_type>::write_parameters(H5::Group const& group) const
{
    h5xx::write_attribute(group, "index", index_);
    h5xx::write_attribute(group, "epsilon", epsilon_.data());
    h5xx::write_attribute(group, "sigma", sigma_.data());
    h5xx::write_attribute(group, "cutoff", r_cut_sigma_.data());
}

/**
 * Initialise potential parameters
 */
template <typename float_type>
power_law<float_type>::power_law(
    unsigned int ntype
  , int index
  , array<float, 3> const& cutoff
  , array<float, 3> const& epsilon
  , array<float, 3> const& sigma
)
  // allocate potential parameters
  : index_(index)
  , epsilon_(scalar_matrix<float_type>(ntype, ntype, 1))
  , sigma_(scalar_matrix<float_type>(ntype, ntype, 1))
  , r_cut_(ntype, ntype)
  , r_cut_sigma_(ntype, ntype)
  , rr_cut_(ntype, ntype)
  , en_cut_(scalar_matrix<float_type>(ntype, ntype, 0))
{
    // FIXME support any number of types
    for (unsigned i = 0; i < std::min(ntype, 2U); ++i) {
        for (unsigned j = i; j < std::min(ntype, 2U); ++j) {
            epsilon_(i, j) = epsilon[i + j];
            sigma_(i, j) = sigma[i + j];
            r_cut_sigma_(i, j) = cutoff[i + j];
        }
    }

    // precalculate derived parameters
    for (unsigned i = 0; i < ntype; ++i) {
        for (unsigned j = i; j < ntype; ++j) {
            r_cut_(i, j) = r_cut_sigma_(i, j) * sigma_(i, j);
            rr_cut_(i, j) = std::pow(r_cut_(i, j), 2);
            // energy shift due to truncation at cutoff length
            en_cut_(i, j) = (*this)(rr_cut_(i, j), i, j).second;
        }
    }

    LOG("potential: power law index: n = " << index_);
    LOG("potential: interaction strength ε = " << epsilon_);
    LOG("potential: interaction range σ = " << sigma_);
    LOG("potential: cutoff length r_c = " << r_cut_sigma_);
    LOG("potential: cutoff energy U = " << en_cut_);
}

template <typename float_type>
void power_law<float_type>::luaopen(lua_State* L)
{
    using namespace luabind;
    module(L, "halmd_wrapper")
    [
        namespace_("mdsim")
        [
            namespace_("host")
            [
                namespace_("forces")
                [
                    class_<power_law, shared_ptr<power_law> >(module_name())
                        .def(constructor<
                            unsigned int
                          , int
                          , array<float, 3> const&
                          , array<float, 3> const&
                          , array<float, 3> const&
                        >())
                        .def("write_parameters", &power_law::write_parameters)
                ]
            ]
        ]
    ];
}

static __attribute__((constructor)) void register_lua()
{
#ifndef USE_HOST_SINGLE_PRECISION
    typedef double float_type;
#else
    typedef float float_type;
#endif

    lua_wrapper::register_(0) //< distance of derived to base class
    [
        &power_law<float_type>::luaopen
    ];

    lua_wrapper::register_(2) //< distance of derived to base class
    [
        &pair_trunc<3, float_type, power_law<float_type> >::luaopen
    ]
    [
        &pair_trunc<2, float_type, power_law<float_type> >::luaopen
    ];
}

// explicit instantiation
#ifndef USE_HOST_SINGLE_PRECISION
template class power_law<double>;
template class pair_trunc<3, double, power_law<double> >;
template class pair_trunc<2, double, power_law<double> >;
#else
template class power_law<float>;
template class pair_trunc<3, float, power_law<float> >;
template class pair_trunc<2, float, power_law<float> >;
#endif

}}} // namespace mdsim::host::forces

} // namespace halmd
