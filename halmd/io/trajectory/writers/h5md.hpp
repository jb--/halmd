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

#ifndef HALMD_IO_TRAJECTORY_H5MD_WRITER_HPP
#define HALMD_IO_TRAJECTORY_H5MD_WRITER_HPP

#include <boost/bind.hpp>
#include <boost/unordered_map.hpp>
#include <lua.hpp>

#include <h5xx/h5xx.hpp>

#include <halmd/io/trajectory/writer.hpp>
#include <halmd/observables/host/samples/phase_space.hpp>

namespace halmd
{
namespace io { namespace trajectory { namespace writers
{

template <int dimension, typename float_type>
class h5md
  : public trajectory::writer<dimension>
{
public:
    typedef trajectory::writer<dimension> _Base;
    typedef observables::host::samples::phase_space<dimension, float_type> sample_type;
    typedef typename sample_type::sample_vector sample_vector_type;
    typedef typename sample_vector_type::value_type vector_type;

    static void luaopen(lua_State* L);

    h5md(
        boost::shared_ptr<sample_type> sample
      , std::string const& file_name
    );
    virtual void append();
    virtual void flush();

    boost::shared_ptr<sample_type> sample;

    H5::H5File const& file() const
    {
        return file_;
    }

private:
    /** H5MD file */
    H5::H5File file_;
    /** dataset write functors */
    std::vector<boost::function<void ()> > writers_;
};

}}} // namespace io::trajectory::writers

} // namespace halmd

#endif /* ! HALMD_IO_TRAJECTORY_H5MD_WRITER_HPP */