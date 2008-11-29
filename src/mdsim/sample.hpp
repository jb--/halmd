/* Phase space sample
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

#ifndef MDSIM_SAMPLE_HPP
#define MDSIM_SAMPLE_HPP

#include <boost/function.hpp>
#include <vector>
#include "config.hpp"

namespace mdsim {

/**
 * MD simulation sample
 */
struct trajectory_sample
{
    /** trajectory sample visitor type */
    typedef boost::function<void (std::vector<hvector>&, std::vector<hvector>&)> visitor;

    /** periodically reduced particle positions */
    std::vector<hvector> r;
    /** periodically extended particle positions */
    std::vector<hvector> R;
    /** particle velocities */
    std::vector<hvector> v;
    /** potential energy per particle */
    double en_pot;
    /** virial equation sum per particle */
    double virial;
};

/**
 * Phase space sample for evaluating correlation functions
 */
struct correlation_sample
{
    // swappable host memory vector type
    typedef std::vector<hvector> vector_type;
    typedef hvector::value_type value_type;
    typedef std::vector<std::pair<hvector, hvector> > density_vector_type;

    /**
     * initialise phase space sample
     */
    correlation_sample(vector_type const& r, vector_type const& v, std::vector<value_type> const& q) : r(r), v(v), rho(q.size(), std::pair<hvector, hvector>(0, 0))
    {
	// spatial Fourier transformation
	for (size_t i = 0; i < r.size(); ++i) {
	    for (unsigned int j = 0; j < q.size(); ++j) {
		// compute averages to maintain accuracy with single precision floating-point
		rho[j].first += (cos(r[i] * q[j]) - rho[j].first) / (i + 1);
		rho[j].second += (sin(r[i] * q[j]) - rho[j].second) / (i + 1);
	    }
	}
	// normalize Fourier transformed density with N^(-1/2)
	const value_type n = std::sqrt(r.size());
	for (unsigned int j = 0; j < q.size(); ++j) {
	    // multiply averages with N^(+1/2)
	    rho[j].first *= n;
	    rho[j].second *= n;
	}
    }

    /** particle positions */
    vector_type r;
    /** particle velocities */
    vector_type v;
    /** spatially Fourier transformed density for given q-values */
    density_vector_type rho;
};

} // namespace mdsim

#endif /* ! MDSIM_SAMPLE_HPP */
