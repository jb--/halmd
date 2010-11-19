/*
 * Copyright © 2010  Peter Colberg
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

#ifndef HALMD_UTILITY_LUA_WRAPPER_UBLAS_HPP
#define HALMD_UTILITY_LUA_WRAPPER_UBLAS_HPP

#include <boost/numeric/ublas/storage.hpp>
#include <luabind/luabind.hpp>

namespace halmd
{
namespace lua_wrapper { namespace ublas
{

int luaopen(lua_State* L);

}} // namespace lua_wrapper::ublas

} // namespace luabind

namespace luabind
{

/**
 * Luabind converter for Boost uBLAS unbounded storage array
 */
template <typename T>
struct default_converter<boost::numeric::ublas::unbounded_array<T> >
  : native_converter_base<boost::numeric::ublas::unbounded_array<T> >
{
    //! compute Lua to C++ conversion score
    static int compute_score(lua_State* L, int index)
    {
        return lua_type(L, index) == LUA_TTABLE ? 0 : -1;
    }

    //! convert from Lua to C++
    boost::numeric::ublas::unbounded_array<T> from(lua_State* L, int index)
    {
        std::size_t size = luaL_getn(L, index);
        boost::numeric::ublas::unbounded_array<T> v(size);
        object table(from_stack(L, index));
        for (std::size_t i = 0; i < v.size(); ++i) {
            v[i] = object_cast<T>(table[i + 1]);
        }
        return v;
    }

    //! convert from C++ to Lua
    void to(lua_State* L, boost::numeric::ublas::unbounded_array<T> const& v)
    {
        object table = newtable(L);
        for (std::size_t i = 0; i < v.size(); ++i) {
            // default_converter<T> only invoked with reference wrapper
            table[i + 1] = boost::cref(v[i]);
        }
        table.push(L);
    }
};

template <typename T>
struct default_converter<boost::numeric::ublas::unbounded_array<T> const&>
  : default_converter<boost::numeric::ublas::unbounded_array<T> > {};


} // namespace luabind

#endif /* ! HALMD_UTILITY_LUA_WRAPPER_UBLAS_HPP */