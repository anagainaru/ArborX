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

#ifndef ARBORX_DETAILS_TAGS_HPP
#define ARBORX_DETAILS_TAGS_HPP

#include <ArborX_Box.hpp>
#include <ArborX_Point.hpp>

namespace ArborX
{
namespace Details
{

struct PointTag
{
};

struct BoxTag
{
};

template <typename Geometry>
struct Tag
{
};

template <>
struct Tag<Point>
{
  using type = PointTag;
};

template <>
struct Tag<Box>
{
  using type = BoxTag;
};

} // namespace Details
} // namespace ArborX

#endif
