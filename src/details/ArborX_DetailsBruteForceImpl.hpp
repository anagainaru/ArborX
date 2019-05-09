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
    int const n_primitives = primitives_.extent(0);

    using Access =
        ArborX::Traits::Access<Predicates, ArborX::Traits::PredicatesTag>;
    int const n_queries = Access::size(predicates);

    Kokkos::parallel_for(
        ARBORX_MARK_REGION(""),
        Kokkos::MDRangePolicy<ExecutionSpace, Kokkos::Rank<2>>(
            space, {{0, 0}}, {{(long)n_queries, (long)n_primitives}}),
        KOKKOS_LAMBDA(int i, int j) {
          auto const predicate = Access::get(predicates, i);
          auto const &primitive = primitives_(j);
          auto const callback = callbacks(i);
          if (predicate(primitive))
          {
            callback(j);
          }
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
