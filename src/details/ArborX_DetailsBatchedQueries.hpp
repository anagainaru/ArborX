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

#ifndef ARBORX_DETAILS_BATCHED_QUERIES_HPP
#define ARBORX_DETAILS_BATCHED_QUERIES_HPP

#include <ArborX_Box.hpp>
#include <ArborX_DetailsAlgorithms.hpp> // returnCentroid, translateAndScale
#include <ArborX_DetailsMortonCode.hpp> // morton3D
#include <ArborX_DetailsSortUtils.hpp>  // sortObjects
#include <ArborX_DetailsUtils.hpp>      // iota, exclusivePrefixSum, lastElement
#include <ArborX_Macros.hpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_Parallel.hpp>
#include <Kokkos_View.hpp>

#include <tuple>

namespace ArborX
{

namespace Details
{
template <typename DeviceType>
struct BatchedQueries
{
public:
  using ExecutionSpace = typename DeviceType::execution_space;

  // BatchedQueries defines functions for sorting queries along the Z-order
  // space-filling curve in order to minimize data divergence.  The goal is
  // to increase correlation between traversal decisions made by nearby
  // threads and thereby increase performance.
  //
  // NOTE: sortQueriesAlongZOrderCurve() does not actually apply the sorting
  // order, it returns the permutation indices.  applyPermutation() was added
  // in that purpose.  reversePermutation() is able to restore the initial
  // order on the results that are in "compressed row storage" format.  You
  // may notice it is not used any more in the code that performs the batched
  // queries.  We found that it was slighly more performant to add a level of
  // indirection when recording results rather than using that function at
  // the end.  We decided to keep reversePermutation around for now.

  template <typename Query>
  static Kokkos::View<size_t *, DeviceType>
  sortQueriesAlongZOrderCurve(Box const &scene_bounding_box,
                              Kokkos::View<Query *, DeviceType> queries)
  {
    Kokkos::View<Box, DeviceType> bounds("bounds");
    Kokkos::deep_copy(bounds, scene_bounding_box);
    return sortQueriesAlongZOrderCurve(bounds, queries);
  }

  template <typename Query>
  static Kokkos::View<size_t *, DeviceType>
  sortQueriesAlongZOrderCurve(Kokkos::View<Box const, DeviceType> bounds,
                              Kokkos::View<Query *, DeviceType> queries)
  {
    auto const n_queries = queries.extent(0);

    Kokkos::View<unsigned int *, DeviceType> morton_codes(
        Kokkos::ViewAllocateWithoutInitializing("morton"), n_queries);
    Kokkos::parallel_for(ARBORX_MARK_REGION("assign_morton_codes_to_queries"),
                         Kokkos::RangePolicy<ExecutionSpace>(0, n_queries),
                         KOKKOS_LAMBDA(int i) {
                           Point xyz =
                               Details::returnCentroid(queries(i)._geometry);
                           translateAndScale(xyz, xyz, bounds());
                           morton_codes(i) = morton3D(xyz[0], xyz[1], xyz[2]);
                         });
    Kokkos::fence();

    return sortObjects(morton_codes);
  }

  template <typename T>
  static Kokkos::View<T *, DeviceType>
  applyPermutation(Kokkos::View<size_t const *, DeviceType> permute,
                   Kokkos::View<T *, DeviceType> v)
  {
    auto const n = permute.extent(0);
    ARBORX_ASSERT(v.extent(0) == n);

    auto w = cloneWithoutInitializingNorCopying(v);
    Kokkos::parallel_for(ARBORX_MARK_REGION("permute_entries"),
                         Kokkos::RangePolicy<ExecutionSpace>(0, n),
                         KOKKOS_LAMBDA(int i) { w(i) = v(permute(i)); });
    Kokkos::fence();

    return w;
  }

  static Kokkos::View<int *, DeviceType>
  permuteOffset(Kokkos::View<size_t const *, DeviceType> permute,
                Kokkos::View<int const *, DeviceType> offset)
  {
    auto const n = permute.extent(0);
    ARBORX_ASSERT(offset.extent(0) == n + 1);

    auto tmp_offset = cloneWithoutInitializingNorCopying(offset);
    Kokkos::parallel_for(
        ARBORX_MARK_REGION("adjacent_difference_and_permutation"),
        Kokkos::RangePolicy<ExecutionSpace>(0, n), KOKKOS_LAMBDA(int i) {
          tmp_offset(permute(i)) = offset(i + 1) - offset(i);
        });
    Kokkos::fence();

    exclusivePrefixSum(tmp_offset);

    return tmp_offset;
  }

  template <typename T>
  static Kokkos::View<T *, DeviceType>
  permuteIndices(Kokkos::View<size_t const *, DeviceType> permute,
                 Kokkos::View<T const *, DeviceType> indices,
                 Kokkos::View<int const *, DeviceType> offset,
                 Kokkos::View<int const *, DeviceType> tmp_offset)
  {
    auto const n = permute.extent(0);

    ARBORX_ASSERT(offset.extent(0) == n + 1);
    ARBORX_ASSERT(tmp_offset.extent(0) == n + 1);
    ARBORX_ASSERT(lastElement(offset) == indices.extent_int(0));
    ARBORX_ASSERT(lastElement(tmp_offset) == indices.extent_int(0));

    auto tmp_indices = cloneWithoutInitializingNorCopying(indices);
    Kokkos::parallel_for(
        ARBORX_MARK_REGION("permute_indices"),
        Kokkos::RangePolicy<ExecutionSpace>(0, n), KOKKOS_LAMBDA(int q) {
          for (int i = 0; i < offset(q + 1) - offset(q); ++i)
          {
            tmp_indices(tmp_offset(permute(q)) + i) = indices(offset(q) + i);
          }
        });
    Kokkos::fence();
    return tmp_indices;
  }

  static std::tuple<Kokkos::View<int *, DeviceType>,
                    Kokkos::View<int *, DeviceType>>
  reversePermutation(Kokkos::View<size_t const *, DeviceType> permute,
                     Kokkos::View<int const *, DeviceType> offset,
                     Kokkos::View<int const *, DeviceType> indices)
  {
    auto const tmp_offset = permuteOffset(permute, offset);

    auto const tmp_indices =
        permuteIndices(permute, indices, offset, tmp_offset);
    return std::make_tuple(tmp_offset, tmp_indices);
  }

  static std::tuple<Kokkos::View<int *, DeviceType>,
                    Kokkos::View<int *, DeviceType>,
                    Kokkos::View<double *, DeviceType>>
  reversePermutation(Kokkos::View<size_t const *, DeviceType> permute,
                     Kokkos::View<int const *, DeviceType> offset,
                     Kokkos::View<int const *, DeviceType> indices,
                     Kokkos::View<double const *, DeviceType> distances)
  {
    auto const tmp_offset = permuteOffset(permute, offset);

    auto const tmp_indices =
        permuteIndices(permute, indices, offset, tmp_offset);

    auto const tmp_distances =
        permuteIndices(permute, distances, offset, tmp_offset);

    return std::make_tuple(tmp_offset, tmp_indices, tmp_distances);
  }
};

} // namespace Details
} // namespace ArborX

#endif
