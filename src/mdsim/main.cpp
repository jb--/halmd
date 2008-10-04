/* Molecular Dynamics Simulation of a Lennard-Jones fluid
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

#include <H5Cpp.h>
#include <boost/algorithm/string/join.hpp>
#include <exception>
#include <iostream>
#include <unistd.h>
#include <vector>
#include "log.hpp"
#include "mdsim.hpp"
#include "options.hpp"
#include "version.h"

int main(int argc, char **argv)
{
#ifdef NDEBUG
    // turns off the automatic error printing from the HDF5 library
    H5::Exception::dontPrint();
#endif

    // parse program options
    mdsim::options opts;
    try {
	opts.parse(argc, argv);
    }
    catch (mdsim::options::exit_exception const& e) {
	return e.status();
    }

    mdsim::log::init(opts);

    LOG(PROGRAM_NAME " " PROGRAM_VERSION);
    LOG("variant: " << PROGRAM_VARIANT);
#ifndef NDEBUG
    LOG_WARNING("built with enabled debugging");
#endif

    // print command line
    std::vector<std::string> cmd(argv, argv + argc);
    LOG("command line: " << boost::algorithm::join(cmd, " "));

    try {
	// initialize molecular dynamics simulation
	mdsim::mdsim sim(opts);

	if (opts.daemon().value()) {
	    // run program in background
	    daemon(0, 0);
	}
	if (!opts.dry_run().value()) {
	    // run MD simulation
	    sim();
	}
    }
    catch (std::exception const& e) {
	LOG_ERROR(e.what());
	LOG_WARNING(PROGRAM_NAME " aborted");
	return EXIT_FAILURE;
    }

    LOG(PROGRAM_NAME " exit");
    return EXIT_SUCCESS;
}
