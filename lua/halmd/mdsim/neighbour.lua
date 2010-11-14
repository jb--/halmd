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
local neighbour_wrapper = {
    host = {
        [2] = halmd_wrapper.mdsim.host.neighbour_2_
      , [3] = halmd_wrapper.mdsim.host.neighbour_3_
    }
}
if halmd_wrapper.mdsim.gpu then
    neighbour_wrapper.gpu = {
        [2] = halmd_wrapper.mdsim.gpu.neighbour_2_
      , [3] = halmd_wrapper.mdsim.gpu.neighbour_3_
    }
end
local mdsim = {
    core = require("halmd.mdsim.core")
}
local device = require("halmd.device")
local po = halmd_wrapper.po
local assert = assert

module("halmd.mdsim.neighbour", halmd.modules.register)

function options(desc)
    desc:add("skin", po.float(), "neighbour list skin")
    if neighbour_wrapper.gpu then
        desc:add("cell-occupancy", po.float(), "desired average cell occupancy")
    end
end

--
-- construct neighbour module
--
function new(args)
    -- dependency injection
    local core = mdsim.core()
    local dimension = assert(core.dimension)
    local particle = assert(core.particle)
    local box = assert(core.box)
    local force = assert(core.force)

    -- command line options
    local skin = args.skin or 0.5 -- default value

    if not device() then
        return neighbour_wrapper.host[dimension](particle, box, force, skin)
    end
    local cell_occupancy = args.cell_occupancy or 0.4 -- default value
    return neighbour_wrapper.gpu[dimension](particle, box, force, skin, cell_occupancy)
end
