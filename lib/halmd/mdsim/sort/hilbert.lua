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
local hilbert_wrapper = {
    host = {
        [2] = halmd_wrapper.mdsim.host.sort.hilbert_2_
      , [3] = halmd_wrapper.mdsim.host.sort.hilbert_3_
    }
}
local mdsim = {
  core = require("halmd.mdsim.core")
}
local args = require("halmd.options")
local assert = assert

module("halmd.mdsim.sort.hilbert", halmd.modules.register)

--
-- construct hilbert module
--
function new()
    local dimension = assert(args.dimension)
    local backend = assert(args.backend)

    -- dependency injection
    local core = mdsim.core()
    local particle = assert(core.particle)
    local box = assert(core.box)
    local neighbour = assert(core.neighbour)

    if backend == "gpu" then
        return nil -- FIXME not implemented
    end
    local hilbert = hilbert_wrapper[backend][dimension]
    return hilbert(particle, box, neighbour)
end