/* HDF5 C++ extensions
 *
 * Copyright © 2008-2009  Peter Colberg and Felix Höfling
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

#ifndef HALMD_UTIL_H5XX_CTYPE_HPP
#define HALMD_UTIL_H5XX_CTYPE_HPP

#define H5E_auto_t_vers 2
#include <H5Cpp.h>

namespace H5xx
{

/*
 * fundamental type to HDF5 native data type translation
 */
template <typename T>
struct ctype;

#define MAKE_CTYPE(CTYPE, H5TYPE) \
template <> struct ctype<CTYPE> \
{ operator H5::PredType const& () { return H5::PredType::NATIVE_##H5TYPE;} }

MAKE_CTYPE(float, FLOAT);
MAKE_CTYPE(double, DOUBLE);
MAKE_CTYPE(long double, LDOUBLE);
MAKE_CTYPE(int8_t, INT8);
MAKE_CTYPE(uint8_t, UINT8);
MAKE_CTYPE(int16_t, INT16);
MAKE_CTYPE(uint16_t, UINT16);
MAKE_CTYPE(int32_t, INT32);
MAKE_CTYPE(uint32_t, UINT32);
MAKE_CTYPE(int64_t, INT64);
MAKE_CTYPE(uint64_t, UINT64);

#undef MAKE_CTYPE

template <typename T>
struct is_boost_array : public boost::false_type {};

template <typename T, size_t size>
struct is_boost_array<boost::array<T, size> >: public boost::true_type {};

template <typename T>
struct is_boost_multi_array : public boost::false_type {};

template <typename T, size_t dimension>
struct is_boost_multi_array<boost::multi_array<T, dimension> >: public boost::true_type {};

template <typename T>
struct is_vector : public boost::false_type {};

template <typename T>
struct is_vector<std::vector<T> >: public boost::true_type {};

/**
 * check data type of abstract dataset (dataset or attribute)
 */
template <typename T>
typename boost::enable_if<boost::is_fundamental<T>, bool>::type
has_type(H5::AbstractDs const& ds)
{
    return ds.getDataType() == ctype<T>();
}

template <typename T>
typename boost::enable_if<boost::is_same<T, std::string>, bool>::type
has_type(H5::AbstractDs const& ds)
{
    return ds.getTypeClass() == H5T_STRING;
}

template <typename T>
typename boost::enable_if<boost::is_same<T, char const*>, bool>::type
has_type(H5::AbstractDs const& ds)
{
    return has_type<std::string>(ds);
}

template <typename T>
typename boost::enable_if<is_vector<T>, bool>::type
has_type(H5::AbstractDs const& ds)
{
    return has_type<typename T::value_type>(ds);
}

template <typename T>
typename boost::enable_if<is_boost_array<T>, bool>::type
has_type(H5::AbstractDs const& ds)
{
    return has_type<typename T::value_type>(ds);
}

template <typename T>
typename boost::enable_if<is_boost_multi_array<T>, bool>::type
has_type(H5::AbstractDs const& ds)
{
    return has_type<typename T::element>(ds);
}

/**
 * check data space of abstract dataset (dataset or attribute)
 */
inline bool has_scalar_space(H5::AbstractDs const& ds)
{
    return ds.getSpace().getSimpleExtentType() == H5S_SCALAR;
}

inline bool has_simple_space(H5::AbstractDs const& ds)
{
    return ds.getSpace().isSimple();
}

/**
 * check data space extent of abstract dataset (dataset or attribute)
 */
template <typename T>
typename boost::enable_if<is_boost_array<T>, bool>::type
has_extent(H5::AbstractDs const& ds)
{
    H5::DataSpace dataspace = ds.getSpace();
    if (!dataspace.isSimple() || dataspace.getSimpleExtentNdims() != 1)
        return false;

    hsize_t dim[1];
    dataspace.getSimpleExtentDims(dim);
    return dim[0] == T::static_size;
}

template <typename T>
typename boost::enable_if<is_boost_multi_array<T>, bool>::type
has_extent(H5::AbstractDs const& ds, typename T::size_type const* shape)
{
    enum { rank = T::dimensionality };
    H5::DataSpace dataspace = ds.getSpace();
    if (!dataspace.isSimple() || dataspace.getSimpleExtentNdims() != rank)
        return false;

    boost::array<hsize_t, rank> dim;
    dataspace.getSimpleExtentDims(dim.data());

    return std::equal(dim.begin(), dim.end(), shape);
}

} // namespace H5xx

#endif /* ! HALMD_UTIL_H5XX_CTYPE_HPP */
