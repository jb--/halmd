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

#ifndef HALMD_NUMERIC_BLAS_FIXED_ARRAY_HPP
#define HALMD_NUMERIC_BLAS_FIXED_ARRAY_HPP

#ifndef __CUDACC__
# include <boost/array.hpp>
#endif
#include <boost/type_traits/is_pod.hpp>
#include <boost/utility/enable_if.hpp>

#include <halmd/config.hpp>
#include <halmd/numeric/mp/dsfloat.hpp>

namespace halmd
{
namespace detail { namespace numeric { namespace blas
{

// import into current namespace
using namespace boost;

#ifndef __CUDACC__

template <typename T, size_t N>
struct fixed_array
  : boost::array<T, N>
{
    fixed_array() : boost::array<T, N>() {}
    /**
     * implicit conversion from base class
     */
    fixed_array(boost::array<T, N> const& a) {
        for (size_t i = 0; i < N; ++i) {
            (*this)[i] = a[i];
        }
    }
};

#else /* __CUDACC__ */

template <typename T, size_t N>
struct _fixed_array_impl;

template <typename T, typename Enable = void>
struct _fixed_array_pod_type;

//
// The purpose of a bounded array is to serve as the underlying
// array type to a fixed-length algebraic vector. It defines
// operator[] to allow convenient access of its components.
//
template <typename T, size_t N>
struct fixed_array
  : _fixed_array_impl<typename _fixed_array_pod_type<T>::type, N>
{
    typedef T value_type;
    enum { static_size = N };

    HALMD_GPU_ENABLED value_type& operator[](size_t i)
    {
        return reinterpret_cast<value_type*>(this)[i];
    }

    HALMD_GPU_ENABLED value_type const& operator[](size_t i) const
    {
        return reinterpret_cast<value_type const*>(this)[i];
    }
};

//
// CUDA shared memory arrays only allow POD-type data members, so we
// cannot use structs with non-default constructors as a data members.
// Instead, we define an equivalent POD type here and cast to the
// non-POD type upon accessing an element of the bounded array.
//

template <typename T>
struct _fixed_array_pod_type<T,
  typename enable_if<is_pod<T> >::type>
{
    typedef T type;
};

template <>
struct _fixed_array_pod_type<dsfloat>
{
    typedef float2 type;
};

//
// These specializations define the data members of bounded array.
//
template <typename T>
struct _fixed_array_impl<T, 1>
{
    T x;
};

template <typename T>
struct _fixed_array_impl<T, 2>
{
    T x, y;
};

template <typename T>
struct _fixed_array_impl<T, 3>
{
    T x, y, z;
};

template <typename T>
struct _fixed_array_impl<T, 4>
{
    T x, y, z, w;
};

#endif /* __CUDACC__ */

}}} // namespace detail::numeric::blas

} // namespace halmd

#endif /* ! HALMD_NUMERIC_BLAS_FIXED_ARRAY_HPP */