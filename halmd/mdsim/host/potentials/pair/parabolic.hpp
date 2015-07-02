/*
 * Copyright © 2008-2013 Felix Höfling
 * Copyright © 2008-2011 Peter Colberg
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

#ifndef HALMD_MDSIM_HOST_POTENTIALS_PAIR_PARABOLIC_HPP
#define HALMD_MDSIM_HOST_POTENTIALS_PAIR_PARABOLIC_HPP

#include <boost/numeric/ublas/matrix.hpp>
#include <lua.hpp>
#include <tuple>
#include <memory>

#include <halmd/io/logger.hpp>
#include <halmd/numeric/pow.hpp>

namespace halmd {
namespace mdsim {
namespace host {
namespace potentials {
namespace pair {

/**
 * A power-law potential @f$ r^{-n} @f$ is often used for
 * repulsive smooth spheres. A big advantage is
 * its scale invariance (in the absence of a cutoff).
 */

template <typename float_type>
class parabolic
{
public:
    typedef boost::numeric::ublas::matrix<float_type> matrix_type;
    typedef boost::numeric::ublas::matrix<unsigned int> uint_matrix_type;

    parabolic(
        matrix_type const& cutoff
      , matrix_type const& epsilon
      , uint_matrix_type const& index
      , std::shared_ptr<halmd::logger> logger = std::make_shared<halmd::logger>()
    );

    /** optimise pow() function by providing the index at compile time
     * @param rr squared distance between particles
     * @param a type of first interacting particle
     * @param b type of second interacting particle
     * @returns tuple of unit "force" @f$ -U'(r)/r @f$ and potential @f$ U(r) @f$
     */
    std::tuple<float_type, float_type> operator()(float_type rr, unsigned a, unsigned b) const
    {
        unsigned int n = index_(a, b);
        float_type r = std::sqrt(rr);
        float_type one_minus_r_ratio = 1 - r /  r_cut_(a, b);
        float_type fval = epsilon_(a, b) * std::pow(one_minus_r_ratio, n-1);
        float_type en_pot = fval * one_minus_r_ratio / n;

        return std::make_tuple(fval / r, en_pot);
    }

    matrix_type const& r_cut() const
    {
        return r_cut_;
    }

    float_type r_cut(unsigned a, unsigned b) const
    {
        return r_cut_(a, b);
    }

    float_type rr_cut(unsigned a, unsigned b) const
    {
        return rr_cut_(a, b);
    }

    matrix_type const& epsilon() const
    {
        return epsilon_;
    }

    uint_matrix_type const& index() const
    {
        return index_;
    }

    unsigned int size1() const
    {
        return epsilon_.size1();
    }

    unsigned int size2() const
    {
        return epsilon_.size2();
    }

    /**
     * Bind class to Lua.
     **/
    static void luaopen(lua_State* L);

private:


    /** interaction strength in MD units */
    matrix_type epsilon_;
    /** power law index */
    uint_matrix_type index_;
    /** cutoff length in MD units */
    matrix_type r_cut_;
    /** square of cutoff length */
    matrix_type rr_cut_;
    /** potential energy at cutoff in MD units */
    matrix_type en_cut_;
    /** module logger */
    std::shared_ptr<logger> logger_;
};

} // namespace pair
} // namespace potentials
} // namespace host
} // namespace mdsim
} // namespace halmd

#endif /* ! HALMD_MDSIM_HOST_POTENTIALS_PAIR_PARABOLIC_HPP */
