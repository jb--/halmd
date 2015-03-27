/*
 * Copyright © 2011-2013 Felix Höfling
 * Copyright © 2011-2012 Michael Kopp
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

#ifndef HALMD_MDSIM_HOST_POTENTIALS_PAIR_YUKAWA_SHIFTED_HPP
#define HALMD_MDSIM_HOST_POTENTIALS_PAIR_YUKAWA_SHIFTED_HPP

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
 * YUKAWA_SHIFTED-Potential
 */

template <typename float_type>
class yukawa_shifted
{
public:
    typedef boost::numeric::ublas::matrix<float_type> matrix_type;
    typedef boost::numeric::ublas::matrix<unsigned int> uint_matrix_type;

    yukawa_shifted(
        matrix_type const& cutoff
      , matrix_type const& gamma
      , matrix_type const& kappa
      , matrix_type const& delta
      , std::shared_ptr<halmd::logger> logger = std::make_shared<halmd::logger>()
    );
    std::tuple<float_type, float_type> operator()(float_type rr, unsigned a, unsigned b) const
    {
        float_type r_ = sqrt(rr + delta_sq_(a, b) );
        if (r_ == 0) {
            return std::make_tuple(0, 0);
        }
        float_type pot_ = gamma_(a, b)  *  exp(-kappa_(a, b) * r_) / r_;
        // the last part "sqrt(rr) / r_" is the projection of the force on the xy-plane, the sqrt (rr) the normalization for the force module
        float_type fval = pot_  * (1. / r_ + kappa_(a, b) ) / r_; // * sqrt(rr) / sqrt(rr);
        
        
        float_type en_pot = pot_ - en_cut_(a, b);
        
        return std::make_tuple(fval, en_pot);
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

    matrix_type const& gamma() const
    {
        return gamma_;
    }

    matrix_type const& kappa() const
    {
        return kappa_;
    }

    matrix_type const& delta() const
    {
        return delta_;
    }
    
    matrix_type const& delta_sq() const
    {
        return delta_sq_;
    }

    unsigned int size1() const
    {
        return gamma_.size1();
    }

    unsigned int size2() const
    {
        return gamma_.size2();
    }

    /**
     * Bind class to Lua.
     */
    static void luaopen(lua_State* L);

private:

    /** interaction strength in MD units */
    matrix_type gamma_;
    /** interaction range in MD units */
    matrix_type kappa_;
    /** layer distance in MD units */
    matrix_type delta_;
    /** layer distance squared in MD units */
    matrix_type delta_sq_;
    /** cutoff length in MD units */
    matrix_type r_cut_;
    /** square of cutoff length */
    matrix_type rr_cut_;
    /** core radius in units of sigma */
    matrix_type r_core_sigma_;
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

#endif /* ! HALMD_MDSIM_HOST_POTENTIALS_PAIR_YUKAWA_SHIFTED_HPP */
