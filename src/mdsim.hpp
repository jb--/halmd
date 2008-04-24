/* Molecular dynamics simulation
 *
 * Copyright (C) 2008  Peter Colberg
 *
 * This program is free software: you can redistribute it and/or modify
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

#ifndef MDSIM_MDSIM_HPP
#define MDSIM_MDSIM_HPP

#include "ljfluid.hpp"
#include "accumulator.hpp"
#include <stdint.h>
#include <ostream>


namespace mdsim
{

/**
 * Molecular dynamics simulation
 */
template <typename T>
class mdsim
{
public:
    mdsim() : steps_(0), time_(0.)
    {
    }

    /**
     * MD simulation step
     */
    void step(ljfluid<T>& fluid)
    {
	float en_pot, virial, vel2_sum;
	T vel_cm;

	// MD simulation step
	fluid.step(en_pot, virial, vel_cm, vel2_sum);
	// advance total simulation time
	time_ = ++steps_ * fluid.timestep();

	// accumulate properties
	en_pot_ += en_pot;
	en_kin_ += vel2_sum / 2;
	en_tot_ += en_pot + vel2_sum / 2;
	temp_ += vel2_sum / T::dim();
	pressure_ += fluid.density() * (vel2_sum + virial);
	vel_cm_ += vel_cm;
    }

    /**
     * clear accumulators
     */
    void clear()
    {
	en_pot_.clear();
	en_kin_.clear();
	en_tot_.clear();
	temp_.clear();
	pressure_.clear();
	vel_cm_.clear();
    }

    /**
     * output measurement properties to stream
     */
    friend std::ostream& operator<<(std::ostream& os, mdsim<T> const& sim)
    {
	os << sim.time_ << "\t";
	os << sim.en_tot_.mean() << "\t" << sim.en_tot_.std() << "\t";
	os << sim.en_pot_.mean() << "\t" << sim.en_pot_.std() << "\t";
	os << sim.en_kin_.mean() << "\t" << sim.en_kin_.std() << "\t";
	os << sim.temp_.mean() << "\t" << sim.temp_.std() << "\t";
	os << sim.pressure_.mean() << "\t" << sim.pressure_.std() << "\t";
	os << sim.vel_cm_.mean() << "\t" << sim.vel_cm_.std();
	return os;
    }

    /**
     * get potential energy per particle
     */
    accumulator<float> const& en_pot() const
    {
	return en_pot_;
    }

    /**
     * get kinetic energy per particle
     */
    accumulator<float> const& en_kin() const
    {
	return en_kin_;
    }

    /**
     * get potential energy
     */
    accumulator<float> const& en_tot() const
    {
	return en_tot_;
    }

    /**
     * get temperature
     */
    accumulator<float> const& temp() const
    {
	return temp_;
    }

    /**
     * get pressure
     */
    accumulator<float> const& pressure() const
    {
	return pressure_;
    }

    /**
     * get center of mass velocity
     */
    accumulator<T> const& vel_cm() const
    {
	return vel_cm_;
    }

    /**
     * get total number of simulation steps
     */
    uint64_t steps() const
    {
	return steps_;
    }

    /**
     * get total simulation time
     */
    float time() const
    {
	return time_;
    }

private:
    /** potential energy per particle */
    accumulator<float> en_pot_;
    /** kinetic energy per particle */
    accumulator<float> en_kin_;
    /** total energy per particle */
    accumulator<float> en_tot_;
    /** temperature */
    accumulator<float> temp_;
    /** pressure */
    accumulator<float> pressure_;
    /** center of mass velocity */
    accumulator<T> vel_cm_;
    /** total number of simulation steps */
    uint64_t steps_;
    /** total simulation time */
    float time_;
};

} // namespace mdsim

#endif /* ! MDSIM_MDSIM_HPP */
