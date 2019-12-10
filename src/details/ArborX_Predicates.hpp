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
#ifndef ARBORX_PREDICATE_HPP
#define ARBORX_PREDICATE_HPP

#include <ArborX_DetailsAlgorithms.hpp>
#include <ArborX_DetailsNode.hpp>

namespace ArborX
{
namespace Details
{
struct NearestPredicateTag
{
};
struct SpatialPredicateTag
{
};
} // namespace Details

template <typename Geometry>
struct Nearest
{
  using Tag = Details::NearestPredicateTag;

  KOKKOS_INLINE_FUNCTION
  Nearest() = default;

  KOKKOS_INLINE_FUNCTION
  Nearest(Geometry const &geometry, int k)
      : _geometry(geometry)
      , _k(k)
  {
  }

  Geometry _geometry;
  int _k = 0;
};

template <typename Geometry>
struct Intersects
{
  using Tag = Details::SpatialPredicateTag;

  KOKKOS_INLINE_FUNCTION Intersects() = default;

  KOKKOS_INLINE_FUNCTION Intersects(Geometry const &geometry)
      : _geometry(geometry)
  {
  }

  template <typename Other>
  KOKKOS_INLINE_FUNCTION bool operator()(Other const &other) const
  {
    return Details::intersects(_geometry, other);
  }

  Geometry _geometry;
};

template <typename Geometry>
KOKKOS_INLINE_FUNCTION Nearest<Geometry> nearest(Geometry const &geometry,
                                                 int k = 1)
{
  return Nearest<Geometry>(geometry, k);
}

template <typename Geometry>
KOKKOS_INLINE_FUNCTION Intersects<Geometry> intersects(Geometry const &geometry)
{
  return Intersects<Geometry>(geometry);
}

} // namespace ArborX

#endif
