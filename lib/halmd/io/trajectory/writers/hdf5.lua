--
-- Copyright © 2010  Peter Colberg
--
-- This file is part of HALMD.
--
-- HALMD is free software: you can redistribute it and/or modify
-- it under the terms of the GNU General Public License as published by
-- the Free Software Foundation, either version 3 of the License, or
-- (at your option) any later version.
--
-- This program is distributed in the hope that it will be useful,
-- but WITHOUT ANY WARRANTY; without even the implied warranty of
-- MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
-- GNU General Public License for more details.
--
-- You should have received a copy of the GNU General Public License
-- along with this program.  If not, see <http://www.gnu.org/licenses/>.
--

require("halmd.modules")

-- grab environment
local mdsim = {
    samples = {
        trajectory = require("halmd.mdsim.samples.trajectory")
    }
}
local hdf5_writer = {
    host = {
        [2] = halmd_wrapper.io.trajectory.writers.hdf5_2_double_
      , [3] = halmd_wrapper.io.trajectory.writers.hdf5_3_double_
    }
    , gpu = {
        [2] = halmd_wrapper.io.trajectory.writers.hdf5_2_float_
      , [3] = halmd_wrapper.io.trajectory.writers.hdf5_3_float_
    }
}
local args = require("halmd.options")
local assert = assert

module("halmd.io.trajectory.writers.hdf5", halmd.modules.register)

--
-- construct HDF5 trajectory writer module
--
function new()
    -- dependency injection
    local sample = mdsim.samples.trajectory()

    -- command line options
    local output = assert(args.output)
    local dimension = assert(args.dimension)
    local backend = assert(args.backend)

    -- parameters
    local file_name = output .. ".trj"

    return hdf5_writer[backend][dimension](sample, file_name)
end
