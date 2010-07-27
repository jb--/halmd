/*
 * Copyright © 2010  Felix Höfling
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

#ifndef HALMD_UTILITY_DEMANGLE_HPP
#define HALMD_UTILITY_DEMANGLE_HPP

#include <cxxabi.h>
#include <string>
#include <typeinfo>

namespace halmd
{

/**
 * return type name in human readable format
 */
inline std::string demangled_name(std::type_info const& type)
{
    int status;
    char* buf = abi::__cxa_demangle(type.name(), 0, 0, &status);

    if(!status) {
        std::string s(buf);
        free(buf);
        return s;
    }
    else {
        return type.name();
    }
}

template <typename T>
inline std::string demangled_name()
{
    return demangled_name(typeid(T));
}

} // namespace halmd

#endif /* ! HALMD_UTILITY_DEMANGLE_HPP */
