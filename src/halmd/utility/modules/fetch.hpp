/*
 * Copyright © 2010  Peter Colberg
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

#ifndef HALMD_UTILITY_MODULES_FETCH_HPP
#define HALMD_UTILITY_MODULES_FETCH_HPP

#include <boost/bind.hpp>

#include <halmd/utility/modules/builder.hpp>
#include <halmd/utility/modules/factory.hpp>
#include <halmd/utility/modules/registry.hpp>

namespace halmd
{
namespace modules
{

template <typename T>
struct fetcher
{
    typedef typename modules::registry Registry;
    typedef typename Registry::Graph Graph;
    typedef typename Registry::Vertex Vertex;
    typedef boost::property_map<Graph, tag::builder>::type BuilderPropertyMap;
    typedef boost::property_traits<BuilderPropertyMap>::value_type Builder;
    typedef typename typed_builder_base<T, modules::factory>::BaseT BaseT;
    typedef typed_builder_base<BaseT, modules::factory> BaseBuilder;

    modules::factory& factory;
    po::options const& vm;

    fetcher(modules::factory& factory, po::options const& vm)
      : factory(factory)
      , vm(vm)
    {}

    operator shared_ptr<T>()
    {
        Vertex v = Registry::template vertex<T>();
        std::vector<Builder>& builder = factory.builder[v];
        if (builder.empty()) {
            return shared_ptr<T>(); // unavailable optional module
        }
        if (builder.size() > 1) {
            throw std::logic_error("ambiguous dependency " + demangled_name<T>());
        }
        return fetch(builder.front());
    }

    operator std::vector<shared_ptr<T> >()
    {
        Vertex v = Registry::template vertex<T>();
        std::vector<Builder>& builder = factory.builder[v];
        std::vector<shared_ptr<T> > modules(builder.size());
        std::transform(
            builder.begin()
          , builder.end()
          , modules.begin()
          , boost::bind(&fetcher::fetch, this, _1)
        );
        return modules;
    }

    shared_ptr<T> fetch(Builder& builder)
    {
        shared_ptr<BaseT> p = dynamic_pointer_cast<BaseBuilder>(builder)->fetch(factory, vm);
        return dynamic_pointer_cast<T>(p);
    }
};

template <typename T>
fetcher<T> fetch(modules::factory& factory, po::options const& vm)
{
    return fetcher<T>(factory, vm);
}

}} // namespace halmd::modules

#endif /* ! HALMD_UTILITY_MODULES_FETCH_HPP */
