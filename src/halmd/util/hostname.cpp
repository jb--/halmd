/*
 * Copyright © 2009  Peter Colberg
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

#include <boost/asio.hpp>

#include <halmd/util/hostname.hpp>

namespace halmd
{

/**
 * get fully qualified host name
 */
std::string get_fqdn_host_name()
{
    boost::asio::io_service ios;
    namespace bai = boost::asio::ip;
    bai::tcp::resolver resolver(ios);
    bai::tcp::resolver::query query(bai::host_name(), "");
    bai::tcp::resolver::iterator addr_iter = resolver.resolve(query);
    bai::tcp::resolver::iterator end;
    bai::tcp::endpoint ep;
    while (addr_iter != end) {
        bai::tcp::resolver::iterator name_iter = resolver.resolve(addr_iter->endpoint());
        if (name_iter != end) {
            return (*name_iter).host_name();
        }
        ++addr_iter;
    }
    return query.host_name(); // failed to resolve FQDN
}

} // namespace halmd