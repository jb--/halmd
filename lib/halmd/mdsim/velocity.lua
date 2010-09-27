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

-- grab environment
local modules = require("halmd.modules")
local velocity = {
    [2] = halmd_wrapper.mdsim.velocity_2_
  , [3] = halmd_wrapper.mdsim.velocity_3_
}
local setmetatable = setmetatable

module("halmd.mdsim.velocity", modules.register)

options = velocity[2].options