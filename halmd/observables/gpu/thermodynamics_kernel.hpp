/*
 * Copyright © 2012  Peter Colberg
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

#ifndef HALMD_OBSERVABLES_GPU_THERMODYNAMICS_KERNEL_HPP
#define HALMD_OBSERVABLES_GPU_THERMODYNAMICS_KERNEL_HPP

#include <halmd/config.hpp>
#include <halmd/numeric/blas/fixed_vector.hpp>
#include <halmd/mdsim/type_traits.hpp>

namespace halmd {
namespace observables {
namespace gpu {

/**
 * Compute total kinetic energy.
 */
template <int dimension, typename float_type>
class kinetic_energy
{
public:
    /** element type of input array */
    typedef float4 argument_type;

    /**
     * Initialise kinetic energy to zero.
     */
    kinetic_energy() : mv2_(0) {}

    /**
     * Accumulate kinetic energy of a particle.
     */
    HALMD_GPU_ENABLED void operator()(argument_type const& velocity)
    {
        fixed_vector<float, dimension> v;
        float mass;
        tie(v, mass) <<= velocity;
        mv2_ += mass * inner_prod(v, v);
    }

    /**
     * Accumulate kinetic energy of another accumulator.
     */
    HALMD_GPU_ENABLED void operator()(kinetic_energy const& acc)
    {
        mv2_ += acc.mv2_;
    }

    /**
     * Returns total kinetic energy.
     */
    float_type operator()() const
    {
        return 0.5 * mv2_;
    }

private:
    /** sum over mass × square of velocity vector */
    float_type mv2_;
};

/**
 * Compute velocity of centre of mass.
 */
template <int dimension, typename float_type>
class velocity_of_centre_of_mass
{
private:
    typedef fixed_vector<float_type, dimension> vector_type;

public:
    /** element type of input array */
    typedef float4 argument_type;

    /**
     * Initialise momentan and mass to zero.
     */
    velocity_of_centre_of_mass() : mv_(0), m_(0) {}

    /**
     * Accumulate momentum and mass of a particle.
     */
    HALMD_GPU_ENABLED void operator()(argument_type const& value)
    {
        fixed_vector<float, dimension> v;
        float mass;
        tie(v, mass) <<= value;
        mv_ += mass * v;
        m_ += mass;
    }

    /**
     * Accumulate velocity centre of mass of another accumulator.
     */
    HALMD_GPU_ENABLED void operator()(velocity_of_centre_of_mass const& acc)
    {
        mv_ += acc.mv_;
        m_ += acc.m_;
    }

    /**
     * Returns velocity centre of mass.
     */
    vector_type operator()() const
    {
        return mv_ / m_;
    }

private:
    /** sum over momentum vector */
    vector_type mv_;
    /** sum over mass */
    float_type m_;
};

/**
 * Compute total potential energy.
 */
template <typename float_type>
class potential_energy
{
public:
    typedef float argument_type;

    /**
     * Accumulate potential energy of a particle.
     */
    HALMD_GPU_ENABLED void operator()(argument_type const& en_pot)
    {
        en_pot_ += en_pot;
    }

    /**
     * Accumulate potential energy of another accumulator.
     */
    HALMD_GPU_ENABLED void operator()(potential_energy const& acc)
    {
        en_pot_ += acc.en_pot_;
    }

    /**
     * Returns total potential energy.
     */
    float_type operator()() const
    {
        return en_pot_;
    }

private:
    /** total potential energy */
    float_type en_pot_;
};

/**
 * Compute total virial sum.
 */
template <int dimension, typename float_type>
class virial
{
private:
    typedef typename mdsim::type_traits<dimension, float>::stress_tensor_type stress_pot_type;

public:
    /** element type of input array */
    typedef typename mdsim::type_traits<dimension, float>::gpu::stress_tensor_type argument_type;

    /**
     * Accumulate stress tensor diagonal of a particle.
     */
    HALMD_GPU_ENABLED void operator()(argument_type const& stress_pot)
    {
        stress_pot_type s = stress_pot;
        virial_ += s[0];
    }

    /**
     * Accumulate virial sum of another accumulator.
     */
    HALMD_GPU_ENABLED void operator()(virial const& acc)
    {
        virial_ += acc.virial_;
    }

    /**
     * Returns total virial sum.
     */
    float_type operator()() const
    {
        return virial_;
    }

private:
    /** total virial sum */
    float_type virial_;
};

} // namespace observables
} // namespace gpu
} // namespace halmd

#endif /* ! HALMD_OBSERVABLES_THERMODYNAMICS_KERNEL_HPP */