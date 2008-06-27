/* 2-dimensional floating-point vector
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

#ifndef MDSIM_VECTOR2D_HPP
#define MDSIM_VECTOR2D_HPP

#include <boost/array.hpp>
#include <cmath>
#include <iostream>


/**
 * 2-dimensional floating-point vector
 */
template <typename T>
class vector2d
{
public:
    typedef T value_type;

public:
    T x, y;

public:
    vector2d()
    {
    }

    /**
     * initialization by vector
     */
    template <typename U>
    vector2d(vector2d<U> const& v) : x(v.x), y(v.y)
    {
    }

    /**
     * initialization by scalar
     */
    template <typename U>
    vector2d(U const& s) : x(s), y(s)
    {
    }

    /**
     * initialization by scalar components
     */
    template <typename U, typename V>
    vector2d(U const& x, V const& y) : x(x), y(y)
    {
    }

    /**
     * initialization by array
     */
    template <typename U>
    vector2d(boost::array<U, 2> const& v) : x(v[0]), y(v[1])
    {
    }

    /**
     * dimension of vector space
     */
    static unsigned int dim()
    {
	return 2;
    }

    /**
     * equality comparison
     */
    bool operator==(vector2d<T> const& v) const
    {
	return (v.x == x && v.y == y);
    }

    /**
     * inequality comparison
     */
    bool operator!=(vector2d<T> const& v) const
    {
	return (v.x != x || v.y != y);
    }

    /**
     * componentwise less than comparison
     */
    bool operator<(vector2d<T> const& v) const
    {
	return (v.x < x && v.y < y);
    }

    /**
     * componentwise greater than comparison
     */
    bool operator>(vector2d<T> const& v) const
    {
	return (v.x > x && v.y > y);
    }

    /**
     * componentwise less than or equal to comparison
     */
    bool operator<=(vector2d<T> const& v) const
    {
	return (v.x <= x && v.y <= y);
    }

    /**
     * componentwise greater than or equal to comparison
     */
    bool operator>=(vector2d<T> const& v) const
    {
	return (v.x >= x && v.y >= y);
    }

    /**
     * assignment by vector
     */
    vector2d<T>& operator=(vector2d<T> const& v)
    {
	x = v.x;
	y = v.y;
	return *this;
    }

    /**
     * assignment by scalar
     */
    vector2d<T>& operator=(T const& s)
    {
	x = s;
	y = s;
	return *this;
    }

    /**
     * assignment by componentwise vector addition
     */
    vector2d<T>& operator+=(vector2d<T> const& v)
    {
	x += v.x;
	y += v.y;
	return *this;
    }

    /**
     * assignment by componentwise vector subtraction
     */
    vector2d<T>& operator-=(vector2d<T> const& v)
    {
	x -= v.x;
	y -= v.y;
	return *this;
    }

    /**
     * assignment by scalar multiplication
     */
    vector2d<T>& operator*=(T const& s)
    {
	x *= s;
	y *= s;
	return *this;
    }

    /**
     * assignment by scalar division
     */
    vector2d<T>& operator/=(T const& s)
    {
	x /= s;
	y /= s;
	return *this;
    }

    /**
     * componentwise vector addition
     */
    friend vector2d<T> operator+(vector2d<T> v, vector2d<T> const& w)
    {
	v.x += w.x;
	v.y += w.y;
	return v;
    }

    /**
     * componentwise vector subtraction
     */
    friend vector2d<T> operator-(vector2d<T> v, vector2d<T> const& w)
    {
	v.x -= w.x;
	v.y -= w.y;
	return v;
    }

    /**
     * scalar product
     */
    T operator*(vector2d<T> const& v) const
    {
	return x * v.x + y * v.y;
    }

    /**
     * scalar multiplication
     */
    friend vector2d<T> operator*(vector2d<T> v, T const& s)
    {
	v.x *= s;
	v.y *= s;
	return v;
    }

    /**
     * scalar multiplication
     */
    friend vector2d<T> operator*(T const& s, vector2d<T> v)
    {
	v.x *= s;
	v.y *= s;
	return v;
    }

    /**
     * scalar division
     */
    friend vector2d<T> operator/(vector2d<T> v, T const& s)
    {
	v.x /= s;
	v.y /= s;
	return v;
    }

    /**
     * write vector components to output stream
     */
    friend std::ostream& operator<<(std::ostream& os, vector2d<T> const& v)
    {
	os << v.x << "\t" << v.y;
	return os;
    }

    /**
     * read vector components from input stream
     */
    friend std::istream& operator>>(std::istream& is, vector2d<T>& v)
    {
	is >> v.x >> v.y;
	return is;
    }
};

/**
 * componentwise round to nearest integer
 */
template <typename T>
vector2d<T> rint(vector2d<T> v);

template <>
vector2d<float> rint(vector2d<float> v)
{
    v.x = rintf(v.x);
    v.y = rintf(v.y);
    return v;
}

template <>
vector2d<double> rint(vector2d<double> v)
{
    v.x = rint(v.x);
    v.y = rint(v.y);
    return v;
}

/**
 * componentwise round to nearest integer, away from zero
 */
template <typename T>
vector2d<T> round(vector2d<T> v);

template <>
vector2d<float> round(vector2d<float> v)
{
    v.x = roundf(v.x);
    v.y = roundf(v.y);
    return v;
}

template <>
vector2d<double> round(vector2d<double> v)
{
    v.x = round(v.x);
    v.y = round(v.y);
    return v;
}

/**
 * componentwise round to nearest integer not greater than argument
 */
template <typename T>
vector2d<T> floor(vector2d<T> v)
{
    v.x = std::floor(v.x);
    v.y = std::floor(v.y);
    return v;
}

/**
 * componentwise round to nearest integer not less argument
 */
template <typename T>
vector2d<T> ceil(vector2d<T> v)
{
    v.x = std::ceil(v.x);
    v.y = std::ceil(v.y);
    return v;
}

/**
 * componentwise round to integer towards zero
 */
template <typename T>
vector2d<T> trunc(vector2d<T> v);

template <>
vector2d<float> trunc(vector2d<float> v)
{
    v.x = truncf(v.x);
    v.y = truncf(v.y);
    return v;
}

template <>
vector2d<double> trunc(vector2d<double> v)
{
    v.x = trunc(v.x);
    v.y = trunc(v.y);
    return v;
}

/**
 * componentwise square root function
 */
template <typename T>
vector2d<T> sqrt(vector2d<T> v)
{
    v.x = std::sqrt(v.x);
    v.y = std::sqrt(v.y);
    return v;
}

/**
 * componentwise cosine function
 */
template <typename T>
vector2d<T> cos(vector2d<T> v)
{
    v.x = std::cos(v.x);
    v.y = std::cos(v.y);
    return v;
}

/**
 * componentwise sine function
 */
template <typename T>
vector2d<T> sin(vector2d<T> v)
{
    v.x = std::sin(v.x);
    v.y = std::sin(v.y);
    return v;
}

#endif /* ! MDSIM_VECTOR2D_HPP */
