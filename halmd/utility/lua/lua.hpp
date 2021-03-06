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

#ifndef HALMD_UTILITY_LUA_LUA_HPP
#define HALMD_UTILITY_LUA_LUA_HPP

#include <boost/lexical_cast.hpp> // Lua class names from template parameters
#include <memory> // pointer holder for luaponte::class_
#include <lua.hpp>
#include <luaponte/luaponte.hpp>
#include <luaponte/exception_handler.hpp>
#include <luaponte/shared_ptr_converter.hpp> //< std::shared_ptr up- and down-casts

#include <halmd/config.hpp> // HALMD_LUA_API
#include <halmd/utility/lua/array_converter.hpp>
#include <halmd/utility/lua/cache_converter.hpp>
#include <halmd/utility/lua/error.hpp>
#include <halmd/utility/lua/fixed_vector_converter.hpp>
#include <halmd/utility/lua/function.hpp>
#include <halmd/utility/lua/pair_converter.hpp>
#include <halmd/utility/lua/raw_array_converter.hpp>
#include <halmd/utility/lua/optional_converter.hpp>
#include <halmd/utility/lua/ublas.hpp>
#include <halmd/utility/lua/vector_converter.hpp>

#endif /* ! HALMD_UTILITY_LUA_LUA_HPP */
