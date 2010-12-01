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

#ifndef HALMD_RANDOM_HOST_RANDOM_HPP
#define HALMD_RANDOM_HOST_RANDOM_HPP

#include <algorithm>
#include <boost/random/uniform_01.hpp>
#include <lua.hpp>
#include <iterator>
#include <utility>

#include <halmd/numeric/blas/fixed_vector.hpp>
#include <halmd/random/host/gsl_rng.hpp>
#include <halmd/random/random.hpp>

namespace halmd
{
namespace random { namespace host
{

class random
  : public halmd::random::random
{
public:
    typedef halmd::random::random _Base;
    typedef host::gfsr4 random_generator; // FIXME template parameter

    static void luaopen(lua_State* L);

    random(unsigned int seed);

    template <typename input_iterator>
    void shuffle(input_iterator first, input_iterator last);
    template <typename value_type>
    value_type uniform();
    template <typename value_type>
    std::pair<value_type, value_type> normal(value_type sigma2);
    template <typename value_type>
    void unit_vector(fixed_vector<value_type, 2>& v);
    template <typename value_type>
    void unit_vector(fixed_vector<value_type, 3>& v);
    template <typename value_type>
    void unit_vector(fixed_vector<value_type, 4>& v);

protected:
    /** pseudo-random number generator */
    random_generator rng_;
};

/**
 * Shuffle sequence in-place
 *
 * D.E. Knuth, Art of Computer Programming, Volume 2:
 * Seminumerical Algorithms, 3rd Edition, 1997,
 * Addison-Wesley, pp. 124-125.
 */
template <typename input_iterator>
void random::shuffle(input_iterator first, input_iterator last)
{
    typedef typename std::iterator_traits<input_iterator>::difference_type difference_type;
    for (difference_type i = last - first; i > 1; --i) {
        std::iter_swap(first + rng_(i), first + (i - 1));
    }
}

template <typename value_type>
value_type random::uniform()
{
    boost::uniform_01<random_generator&, value_type> variate(rng_);
    return variate();
}

/**
 * Generate two random numbers from normal distribution
 *
 * The Box-Muller transformation for generating random numbers
 * in the normal distribution was originally described in
 *
 *   G.E.P. Box and M.E. Muller, A Note on the Generation of
 *   Random Normal Deviates, The Annals of Mathematical Statistics,
 *   1958, 29, p. 610-611
 *
 * Here, we use instead the faster polar method of the Box-Muller
 * transformation, see
 *
 *   D.E. Knuth, Art of Computer Programming, Volume 2: Seminumerical
 *   Algorithms, 3rd Edition, 1997, Addison-Wesley, p. 122
 */
template <typename value_type>
std::pair<value_type, value_type> random::normal(value_type sigma)
{
    boost::uniform_01<random_generator&, value_type> variate(rng_);
    value_type x, y, s;
    do {
        x = 2. * variate() - 1.;
        y = 2. * variate() - 1.;
        s = x * x + y * y;
    } while (s >= 1.);

    s = sigma * std::sqrt(-2. * std::log(s) / s);
    x *= s;
    y *= s;
    return std::make_pair(x, y);
}

/**
 * Generate 2-dimensional random unit vector
 */
template <typename value_type>
void random::unit_vector(fixed_vector<value_type, 2>& v)
{
    value_type s;
    do {
        v[0] = 2 * uniform<value_type>() - 1;
        v[1] = 2 * uniform<value_type>() - 1;
        s = v[0] * v[0] + v[1] * v[1];
    } while (s >= 1);

    s = 1 / std::sqrt(s);
    v[0] *= s;
    v[1] *= s;
}

/**
 * Generate 3-dimensional random unit vector
 *
 * The following method requires an average of 8/Pi =~ 2.55
 * uniform random numbers. It is described in
 *
 * G. Marsaglia, Choosing a Point from the Surface of a Sphere,
 * The Annals of Mathematical Statistics, 1972, 43, p. 645-646
 *
 * http://projecteuclid.org/euclid.aoms/1177692644#
 *
 */
template <typename value_type>
void random::unit_vector(fixed_vector<value_type, 3>& v)
{
    value_type s;

    do {
        v[0] = 2 * uniform<value_type>() - 1;
        v[1] = 2 * uniform<value_type>() - 1;
        s = v[0] * v[0] + v[1] * v[1];
    } while (s >= 1);

    v[2] = 1 - 2 * s;
    s = 2 * std::sqrt(1 - s);
    v[0] *= s;
    v[1] *= s;
}

/**
 * Choose point from surface of 4-sphere
 *
 * G. Marsaglia, Choosing a Point from the Surface of a Sphere,
 * The Annals of Mathematical Statistics, 1972, 43, p. 645-646
 *
 * http://projecteuclid.org/euclid.aoms/1177692644#
 *
 */
template <typename value_type>
void random::unit_vector(fixed_vector<value_type, 4>& v)
{
    value_type s1;
    do {
        v[0] = 2 * uniform<value_type>() - 1;
        v[1] = 2 * uniform<value_type>() - 1;
        s1 = v[0] * v[0] + v[1] * v[1];
    } while (s1 >= 1);

    value_type s2;
    do {
        v[2] = 2 * uniform<value_type>() - 1;
        v[3] = 2 * uniform<value_type>() - 1;
        s2 = v[2] * v[2] + v[3] * v[3];
    } while (s2 >= 1);

    value_type s = std::sqrt((1 - s1) / s2);
    v[2] *= s;
    v[3] *= s;
}

}} // namespace random::host

} // namespace halmd

#endif /* ! HALMD_RANDOM_HOST_RANDOM_HPP */
