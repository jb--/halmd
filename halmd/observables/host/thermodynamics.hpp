/*
 * Copyright © 2010-2012  Felix Höfling
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

#ifndef HALMD_OBSERVABLES_HOST_THERMODYNAMICS_HPP
#define HALMD_OBSERVABLES_HOST_THERMODYNAMICS_HPP

#include <boost/make_shared.hpp>
#include <lua.hpp>

#include <halmd/io/logger.hpp>
#include <halmd/mdsim/box.hpp>
#include <halmd/mdsim/clock.hpp>
#include <halmd/mdsim/host/force.hpp>
#include <halmd/observables/host/samples/particle_group.hpp>
#include <halmd/observables/thermodynamics.hpp>
#include <halmd/utility/data_cache.hpp>
#include <halmd/utility/profiler.hpp>

namespace halmd {
namespace observables {
namespace host {

template <int dimension, typename float_type>
class thermodynamics
    : public observables::thermodynamics<dimension>
{
private:
    typedef observables::thermodynamics<dimension> _Base;

public:
    typedef typename _Base::vector_type vector_type;
    typedef mdsim::box<dimension> box_type;
    typedef mdsim::clock clock_type;
    typedef mdsim::host::force<dimension, float_type> force_type;
    typedef samples::particle_group<dimension, float_type> particle_group_type;
    typedef typename particle_group_type::particle_type particle_type;
    typedef logger logger_type;

    static void luaopen(lua_State* L);

    thermodynamics(
        boost::shared_ptr<particle_group_type const> particle_group
      , boost::shared_ptr<box_type const> box
      , boost::shared_ptr<clock_type const> clock
      , boost::shared_ptr<force_type const> force
      , boost::shared_ptr<logger_type> logger = boost::make_shared<logger_type>()
    );

    thermodynamics(
        boost::shared_ptr<particle_type const> particle
      , boost::shared_ptr<box_type const> box
      , boost::shared_ptr<clock_type const> clock
      , boost::shared_ptr<force_type const> force
      , boost::shared_ptr<logger_type> logger = boost::make_shared<logger_type>()
    );

    virtual unsigned int nparticle() const
    {
        return particle_group_->size();
    }

    virtual double volume() const
    {
        return box_->volume();
    }

    virtual double en_kin();
    virtual vector_type const& v_cm();

    virtual double en_pot()
    {
        return force_->potential_energy();
    }

    virtual double virial()
    {
        return force_->stress_tensor_pot()[0];
    }

    virtual double hypervirial()
    {
        return force_->hypervirial();
    }

    virtual void clear_cache();

private:
    typedef halmd::utility::profiler profiler_type;
    typedef profiler_type::accumulator_type accumulator_type;
    typedef profiler_type::scoped_timer_type scoped_timer_type;

    struct runtime
    {
        accumulator_type en_kin;
        accumulator_type v_cm;
    };

    /** module dependencies */
    boost::shared_ptr<box_type const> box_;
    boost::shared_ptr<particle_group_type const> particle_group_;
    boost::shared_ptr<force_type const> force_;
    /** module logger */
    boost::shared_ptr<logger_type> logger_;

    /** cached results */
    data_cache<double> en_kin_;
    data_cache<vector_type> v_cm_;

    /** profiling runtime accumulators */
    runtime runtime_;
};

} // namespace observables
} // namespace host
} // namespace halmd

#endif /* ! HALMD_OBSERVABLES_THERMODYNAMICS_HPP */
