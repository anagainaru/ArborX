/****************************************************************************
 * Copyright (c) 2012-2019 by the ArborX authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef ARBORX_DETAILS_BRUTE_FORCE_IMPL_HPP
#define ARBORX_DETAILS_BRUTE_FORCE_IMPL_HPP

#include <ArborX_DetailsBatchedQueries.hpp>
#include <ArborX_DetailsBoundingVolumeHierarchyImpl.hpp> // FIXME
#include <ArborX_DetailsKokkosExt.hpp>                   // ArithmeticTraits
#include <ArborX_DetailsPriorityQueue.hpp>
#include <ArborX_Predicates.hpp>
#include <ArborX_Traits.hpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>

namespace ArborX
{
namespace Details
{
template <typename Primitives>
struct WrappedBF
{
  Primitives primitives_;

  template <typename ExecutionSpace, typename Predicates, typename Callbacks>
  void operator()(ExecutionSpace const &space, Predicates const predicates,
                  Callbacks const &callbacks) const
  {
    using TeamPolicy = Kokkos::TeamPolicy<ExecutionSpace>;
    int const n_primitives = primitives_.extent(0);
    using Access =
        ArborX::Traits::Access<Predicates, ArborX::Traits::PredicatesTag>;
    int const n_queries = Access::size(predicates);
    using PredicateType = decay_result_of_get_t<Access>;
    using PrimitiveType = typename Primitives::value_type;
    int const predicates_per_team = 600;
    int const primitives_per_team = 600;

    int const n_primitive_tiles =
        ceil((float)n_primitives / primitives_per_team);
    int const n_predicate_tiles = ceil((float)n_queries / predicates_per_team);
    int const n_teams = n_primitive_tiles * n_predicate_tiles;

    auto &pbf = primitives_;

    Kokkos::parallel_for(
        TeamPolicy((long)n_teams, Kokkos::AUTO),
        KOKKOS_LAMBDA(const typename TeamPolicy::member_type &teamMember) {
          int predicate_start = predicates_per_team *
				teamMember.league_rank() / n_primitive_tiles;
          int primitive_start = primitives_per_team *
				(teamMember.league_rank() % n_primitive_tiles);

          int predicates_in_this_team = KokkosExt::min(
	      predicates_per_team, n_queries - predicate_start);
          int primitives_in_this_team = KokkosExt::min(
              primitives_per_team, n_primitives - primitive_start);

          Kokkos::parallel_for(
              Kokkos::TeamThreadRange(teamMember,
                                      (long)primitives_in_this_team),
              [&](const int j) {
                Kokkos::parallel_for(
                    Kokkos::ThreadVectorRange(teamMember,
                                              predicates_in_this_team),
                    [&](const int q) {
                      auto const predicate = Access::get(
                                       predicates, predicate_start + q);
                      auto const &primitive = pbf(primitive_start + j);
                      if (predicate(primitive))
                      {
                        auto const callback = callbacks(q + predicate_start);
                        callback(j + primitive_start);
                      }
                    });
              });
        });
  }
};

namespace BruteForceImpl
{
template <typename ExecutionSpace, typename Primitives, typename Predicates,
          typename OutputView, typename OffsetView, typename Callback>
void queryDispatch(Details::SpatialPredicateTag, ExecutionSpace const &space,
                   Primitives const &primitives, Predicates const &predicates,
                   Callback const &callback, OutputView &out,
                   OffsetView &offset,
                   Experimental::TraversalPolicy const &policy =
                       Experimental::TraversalPolicy())
{
  static_assert(is_detected<SpatialPredicateInlineCallbackArchetypeExpression,
                            Callback, PredicatesHelper<Predicates>,
                            OutputFunctorHelper<OutputView>>::value,
                "Callback function does not have the correct signature");

  dood(space, WrappedBF<Primitives>{primitives}, predicates, callback, out,
       offset, policy._buffer_size);
}

template <typename ExecutionSpace, typename Primitives, typename Predicates,
          typename Indices, typename Offset>
inline void queryDispatch(SpatialPredicateTag, Primitives const &primitives,
                          ExecutionSpace const &space,
                          Predicates const &predicates, Indices &indices,
                          Offset &offset,
                          Experimental::TraversalPolicy const &policy =
                              Experimental::TraversalPolicy())
{
  queryDispatch(SpatialPredicateTag{}, primitives, space, predicates,
                CallbackDefaultSpatialPredicate{}, indices, offset, policy);
}
} // namespace BruteForceImpl
} // namespace Details
} // namespace ArborX

#endif
