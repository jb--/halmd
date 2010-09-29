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

#include <halmd/io/trajectory/writer.hpp>
#include <halmd/utility/lua_wrapper/lua_wrapper.hpp>

using namespace boost;
using namespace std;

namespace halmd
{
namespace io { namespace trajectory
{

template <typename T>
static void register_lua(char const* class_name)
{
    using namespace luabind;
    lua_wrapper::register_(0) //< distance of derived to base class
    [
        namespace_("halmd_wrapper")
        [
            namespace_("io")
            [
                namespace_("trajectory")
                [
                    class_<T, shared_ptr<T> >(class_name)
                        .def("flush", &T::flush)
                        .def("append", &T::append)
                ]
            ]
        ]
    ];
}

static __attribute__((constructor)) void register_lua()
{
    register_lua<writer<3> >("writer_3_");
    register_lua<writer<2> >("writer_2_");
}

}} // namespace io::trajectory

} // namespace halmd
