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

#include <halmd/algorithm/gpu/reduce_kernel.cuh>
#include <halmd/numeric/mp/dsfloat.hpp>
#include <halmd/observables/gpu/thermodynamics_kernel.hpp>

using namespace halmd::observables::gpu;

namespace halmd {

template class reduction_kernel<kinetic_energy<3, dsfloat> >;
template class reduction_kernel<kinetic_energy<2, dsfloat> >;
template class reduction_kernel<velocity_of_centre_of_mass<3, dsfloat> >;
template class reduction_kernel<velocity_of_centre_of_mass<2, dsfloat> >;
template class reduction_kernel<potential_energy<dsfloat> >;
template class reduction_kernel<virial<3, dsfloat> >;
template class reduction_kernel<virial<2, dsfloat> >;

} // namespace halmd
