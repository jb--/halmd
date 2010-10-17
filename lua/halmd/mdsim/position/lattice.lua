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
local lattice_wrapper = {
    host = {
        [2] = halmd_wrapper.mdsim.host.position.lattice_2_
      , [3] = halmd_wrapper.mdsim.host.position.lattice_3_
    }
}
if halmd_wrapper.mdsim.gpu then
    lattice_wrapper.gpu = {
        [2] = halmd_wrapper.mdsim.gpu.position.lattice_2_
      , [3] = halmd_wrapper.mdsim.gpu.position.lattice_3_
    }
end
local mdsim = {
    core = require("halmd.mdsim.core")
}
local random = {
    gpu = require("halmd.gpu.random")
  , host = require("halmd.host.random")
}
local device = require("halmd.device")
local args = require("halmd.options")
local assert = assert

module("halmd.mdsim.position.lattice", halmd.modules.register)

--
-- construct lattice module
--
function new()
    local dimension = assert(args.dimension)

    -- dependency injection
    local core = mdsim.core()
    local particle = assert(core.particle)
    local box = assert(core.box)

    if not device() then
        return lattice_wrapper.host[dimension](particle, box, random.host())
    end
    return lattice_wrapper.gpu[dimension](particle, box, random.gpu())
end