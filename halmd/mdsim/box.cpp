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

#include <boost/bind.hpp>
#include <boost/shared_ptr.hpp>
#include <cmath>
#include <functional> // std::multiplies
#include <luabind/luabind.hpp>
#include <luabind/out_value_policy.hpp>
#include <numeric> // std::accumulate

#include <halmd/io/logger.hpp>
#include <halmd/io/utility/hdf5.hpp>
#include <halmd/mdsim/box.hpp>
#include <halmd/utility/lua/lua.hpp>

using namespace boost;
using namespace std;

namespace halmd {
namespace mdsim {

/**
 * Set box edge lengths
 */
template <int dimension>
box<dimension>::box(vector_type const& length)
  : length_(length)
  , length_half_(0.5 * length_)
{
    LOG("edge lengths of simulation box: " << length_);
}

template <int dimension>
typename box<dimension>::vector_type
box<dimension>::origin() const
{
    return -length_half_;
}

template <int dimension>
vector<typename box<dimension>::vector_type>
box<dimension>::edges() const
{
    vector<vector_type> edges(dimension, 0);
    for (int i = 0; i < dimension; ++i) {
        edges[i][i] = length_[i];
    }
    return edges;
}

template <int dimension>
double box<dimension>::volume() const
{
    return accumulate(length_.begin(), length_.end(), 1., multiplies<double>());
}

template <typename box_type>
static std::function<typename box_type::vector_type ()>
wrap_origin(boost::shared_ptr<box_type const> self)
{
    return [=]() {
        return self->origin();
    };
}

template <typename box_type>
static std::function<vector<typename box_type::vector_type> ()>
wrap_edges(boost::shared_ptr<box_type const> self)
{
    return [=]() {
        return self->edges();
    };
}

template <typename box_type>
static std::function<std::vector<typename box_type::vector_type>& ()>
edges_to_length(std::function<typename box_type::vector_type ()>& length)
{
    typedef typename box_type::vector_type vector_type;
    typedef std::vector<vector_type> edges_type;
    boost::shared_ptr<edges_type> edges = boost::make_shared<edges_type>();
    length = [=]() -> vector_type {
        vector_type length;
        if (edges->size() != box_type::vector_type::static_size) {
            throw std::runtime_error("edges have mismatching dimension");
        }
        for (unsigned int i = 0; i < box_type::vector_type::static_size; ++i) {
            length[i] = (*edges)[i][i];
        }
        return length;
    };
    return [=]() -> edges_type& {
        return *edges;
    };
}

template <int dimension>
void box<dimension>::luaopen(lua_State* L)
{
    using namespace luabind;
    static std::string class_name("box_" + std::to_string(dimension));
    module(L, "libhalmd")
    [
        namespace_("mdsim")
        [
            class_<box, boost::shared_ptr<box> >(class_name.c_str())
                .def(constructor<vector_type const&>())
                .property("length", &box::length)
                .property("volume", &box::volume)
                .property("origin", &box::origin)
                .property("edges", &box::edges)
                .scope
                [
                    def("edges", &wrap_edges<box>)
                  , def("origin", &wrap_origin<box>)
                  , def("edges_to_length", &edges_to_length<box>, pure_out_value(_1))
                ]
        ]
    ];
}

HALMD_LUA_API int luaopen_libhalmd_mdsim_box(lua_State* L)
{
    box<3>::luaopen(L);
    box<2>::luaopen(L);
    return 0;
}

// explicit instantiation
template class box<3>;
template class box<2>;

} // namespace mdsim
} // namespace halmd
