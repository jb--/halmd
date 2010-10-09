/* HDF5 C++ extensions
 *
 * Copyright © 2008-2009  Peter Colberg
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

#ifndef HALMD_UTIL_H5XX_EXCEPTION_HPP
#define HALMD_UTIL_H5XX_EXCEPTION_HPP

#include <H5xx/hdf5.hpp>

namespace H5
{

template <typename Exception>
class no_autoprint : public Exception
{
public:
    no_autoprint()
    {
        Exception::getAutoPrint(func, &client_data);
        Exception::dontPrint();
    }

    ~no_autoprint()
    {
        Exception::setAutoPrint(func, client_data);
    }

private:
    H5E_auto_t func;
    void* client_data;
};

#define H5XX_NO_AUTO_PRINT(exception) H5::no_autoprint<exception> __no_autoprint;

} // namespace H5

#endif /* ! HALMD_UTIL_H5XX_EXCEPTION_HPP */