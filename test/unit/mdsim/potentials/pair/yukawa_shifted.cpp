/*
 * Copyright © 2011-2013 Felix Höfling
 * Copyright © 2011-2012 Michael Kopp
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

#include <halmd/config.hpp>

#define BOOST_TEST_MODULE yukawa_shifted
#include <boost/test/unit_test.hpp>

#include <boost/foreach.hpp>
#include <boost/numeric/ublas/assignment.hpp> // <<=
#include <boost/numeric/ublas/banded.hpp>
#include <cmath> // std::pow
#include <limits>
#include <numeric> // std::accumulate

#include <halmd/mdsim/box.hpp>
#include <halmd/mdsim/host/potentials/pair/yukawa_shifted.hpp>
#include <test/tools/ctest.hpp>

using namespace boost;
using namespace halmd;
using namespace std;

/** test Yukawa-Shifted potential
 *
 *  The host module is a conventional functor which can be tested directly. For
 *  the GPU module, we use the pair_trunc force module in two dimensions to
 *  compute some values of the potential which are compared against the host
 *  module. This requires a special neighbour list module with only one defined
 *  neighbour per particle.
 */

BOOST_AUTO_TEST_CASE( yukawa_shifted_host )
{
    typedef mdsim::host::potentials::pair::yukawa_shifted<double> potential_type;
    typedef potential_type::matrix_type matrix_type;
    typedef potential_type::uint_matrix_type uint_matrix_type;
 /*
    // define interaction parameters
    //
    // choose numbers that are exactly representable as float,
    // otherwise one has to account for rounding errors in the
    // computation of reference values
    unsigned int ntype = 2;  // test a binary mixture
    matrix_type cutoff_array(ntype, ntype);
    cutoff_array <<=
        120.0, 120.0
      , 120.0, 120.0;
    matrix_type gamma_array(ntype, ntype);
    gamma_array <<=
        -0.175,  0.175
       , 0.175, -0.175;
    matrix_type kappa_array(ntype, ntype);
    kappa_array <<=
        2.0, 1.0
      , 0.5, 1.0;
    matrix_type delta_array(ntype, ntype);
    delta_array <<=
        0.35, 1.35
      , 0.65, 0.35;
      
    // construct module
       
    potential_type potential(cutoff_array, gamma_array, kappa_array, delta_array);

    // test paramters
    matrix_type gamma = potential.gamma();
    BOOST_CHECK(gamma(0, 0) == gamma_array(0, 0));
    BOOST_CHECK(gamma(0, 1) == gamma_array(0, 1));
    BOOST_CHECK(gamma(1, 0) == gamma_array(1, 0));
    BOOST_CHECK(gamma(1, 1) == gamma_array(1, 1));
    
    matrix_type kappa = potential.kappa();
    BOOST_CHECK(kappa(0, 0) == kappa_array(0, 0));
    BOOST_CHECK(kappa(0, 1) == kappa_array(0, 1));
    BOOST_CHECK(kappa(1, 0) == kappa_array(1, 0));
    BOOST_CHECK(kappa(1, 1) == kappa_array(1, 1));
    
    matrix_type delta = potential.delta();
    BOOST_CHECK(delta(0, 0) == delta_array(0, 0));
    BOOST_CHECK(delta(0, 1) == delta_array(0, 1));
    BOOST_CHECK(delta(1, 0) == delta_array(1, 0));
    BOOST_CHECK(delta(1, 1) == delta_array(1, 1));
 
    matrix_type delta_sq = potential.delta_sq();
    BOOST_CHECK(delta_sq(0, 0) == delta_array(0, 0) * delta_array(0, 0));
    BOOST_CHECK(delta_sq(0, 1) == delta_array(0, 1) * delta_array(0, 1));
    BOOST_CHECK(delta_sq(1, 0) == delta_array(1, 0) * delta_array(1, 0));
    BOOST_CHECK(delta_sq(1, 1) == delta_array(1, 1) * delta_array(1, 1));
 
    // evaluate some points of potential and force
    typedef boost::array<double, 3> array_type;
    const double tolerance = 3e-06; //5 * numeric_limits<double>::epsilon();

    // expected results  
    boost::array<array_type, 7> results_aa = {{
          {{0.1, -3.02753781249, -0.232144796952}}
        , {{0.5, -0.504319615286, -0.0845962156153}}
        , {{0.8, -0.125887700883, -0.034950663037}}
        , {{1.0, -0.0551461376835, -0.0198468398622}}
        , {{3.0, -0.000106424629207, -0.00013789244318}}
        , {{5.0, -6.78780656132e-07, -1.54680151877e-06}}
        , {{10.0, -7.47317041106e-12, -3.56093534957e-11}}
    }};

    BOOST_FOREACH (array_type const& a, results_aa) {
        double rr = std::pow(a[0], 2);
        double fval, en_pot;
        std::tie(fval, en_pot) = potential(rr, 0, 0);  // interaction AA

        // tolerance due to floating-point rounding depends on difference (r-r_core)
        double r = a[0] / sigma_array(0, 0);        //< r in units of σ
        double tolerance = eps * index_array(0, 0) * (1 + r / (r - core_array(0, 0)));

        BOOST_CHECK_CLOSE_FRACTION(fval, a[1], tolerance);
        BOOST_CHECK_CLOSE_FRACTION(en_pot, a[2], tolerance);

    };
        // expected results  
    boost::array<array_type, 7> results_ab = {{
          {{0.1, 0.0428863495801, 0.0333896763026}}
        , {{0.5, 0.0339156319872, 0.0288119480806}}
        , {{0.8, 0.0242251954841, 0.0232187918478}}
        , {{1.0, 0.0184331299486, 0.0194130341479}}
        , {{3.0, 0.000785698806235, 0.0019822161646}}
        , {{5.0, 4.3851061493e-05, 0.000190352267639}}
        , {{10.0, 7.832324803e-08, 7.19076269054e-07}}
    }};

    BOOST_FOREACH (array_type const& a, results_ab) {
        double rr = std::pow(a[0], 2);
        double fval, en_pot;
        std::tie(fval, en_pot) = potential(rr, 0, 1);  // interaction AB

        // tolerance due to floating-point rounding depends on difference (r-r_core)
        double r = a[0] / sigma_array(0, 1);        //< r in units of σ
        double tolerance = eps * index_array(0, 1) * (1 + r / (r - core_array(0, 1)));

        BOOST_CHECK_CLOSE_FRACTION(fval, a[1], tolerance);
        BOOST_CHECK_CLOSE_FRACTION(en_pot, a[2], tolerance);
    };

        // expected results  
    boost::array<array_type, 7> results_ba = {{
         {{0.1, 0.588464044102, 0.191530830278}}
        , {{0.5, 0.296930042649, 0.14161782711}}
        , {{0.8, 0.14462330766, 0.101401254189}}
        , {{1.0, 0.0906979290016, 0.080820852548}}
        , {{3.0, 0.00330503574767, 0.0122856412575}}
        , {{5.0, 0.000386374942351, 0.00278969473646}}
        , {{10.0, 6.96868780202e-06, 0.00011643075786}}
    }};
    
    BOOST_FOREACH (array_type const& a, results_ba) {
        double rr = std::pow(a[0], 2);
        double fval, en_pot;
        std::tie(fval, en_pot) = potential(rr, 1, 0);  // interaction
        BOOST_CHECK_CLOSE_FRACTION(fval, a[1], tolerance);
        BOOST_CHECK_CLOSE_FRACTION(en_pot, a[2], tolerance);

    };
        // expected results  
    boost::array<array_type, 7> results_bb = {{
          {{0.1, -3.43909975643, -0.334075426805}}
        , {{0.5, -0.673288088186, -0.155744573149}}
        , {{0.8, -0.205605137226, -0.0836925444909}}
        , {{1.0, -0.105048428717, -0.0572556197155}}
        , {{3.0, -0.00124569157775, -0.0028265767971}}
        , {{5.0, -5.56152252257e-05, -0.000232391696044}}
        , {{10.0, -8.67502688344e-08, -7.89165602859e-07}}
    }};
    
    BOOST_FOREACH (array_type const& a, results_bb) {
        double rr = std::pow(a[0], 2);
        double fval, en_pot;
        std::tie(fval, en_pot) = potential(rr, 1, 1);  // interaction BB

        // tolerance due to floating-point rounding depends on difference (r-r_core)
        double r = a[0] / sigma_array(1, 1);        //< r in units of σ
        double tolerance = eps * index_array(1, 1) * (1 + r / (r - core_array(1, 1)));

        BOOST_CHECK_CLOSE_FRACTION(fval, a[1], tolerance);
        BOOST_CHECK_CLOSE_FRACTION(en_pot, a[2], tolerance);
    }; */
}