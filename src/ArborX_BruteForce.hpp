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
#ifndef ARBORX_BRUTE_FORCE_HPP
#define ARBORX_BRUTE_FORCE_HPP

#include <ArborX_Box.hpp>
#include <ArborX_DetailsBruteForceImpl.hpp>
#include <ArborX_DetailsConcepts.hpp>
#include <ArborX_DetailsTreeConstruction.hpp>

#include <Kokkos_Core.hpp>

namespace ArborX
{

template <typename MemorySpace, typename Enable = void>
class BruteForce
{
public:
  using memory_space = MemorySpace;
  static_assert(Kokkos::is_memory_space<MemorySpace>::value, "");
  using bounding_volume_type = Box;
  using size_type = typename MemorySpace::size_type;

  BruteForce() = default;

  template <typename ExecutionSpace, typename Primitives>
  BruteForce(ExecutionSpace const &space, Primitives const &primitives);

  KOKKOS_FUNCTION
  size_type size() const noexcept { return _size; }

  KOKKOS_FUNCTION
  bool empty() const noexcept { return size() == 0; }

  bounding_volume_type bounds() const noexcept { return _bounds; }

  template <typename ExecutionSpace, typename Predicates, typename... Args>
  void query(ExecutionSpace const &space, Predicates const &predicates,
             Args &&... args) const
  {
    static_assert(Kokkos::is_execution_space<ExecutionSpace>::value, "");

    using Access = Traits::Access<Predicates, Traits::PredicatesTag>;
    using Tag =
        typename Details::Tag<Details::decay_result_of_get_t<Access>>::type;
    static_assert(std::is_same<Tag, Details::NearestPredicateTag>::value ||
                      std::is_same<Tag, Details::SpatialPredicateTag>::value,
                  "Invalid tag for the predicates");

    Details::BruteForceImpl::queryDispatch(Tag{}, space, _bounding_volumes,
                                           predicates,
                                           std::forward<Args>(args)...);
  }

private:
  bounding_volume_type const &getBoundingVolume(size_type i) const
  {
    return _bounding_volumes(i);
  }

  size_type _size;
  bounding_volume_type _bounds;
  Kokkos::View<bounding_volume_type *, memory_space> _bounding_volumes;
};

template <typename DeviceType>
class BruteForce<
    DeviceType, std::enable_if_t<Kokkos::is_device<DeviceType>::value>>
    : public BruteForce<typename DeviceType::memory_space>
{
public:
  using device_type = DeviceType;
  BruteForce() = default;
  template <typename Primitives>
  BruteForce(Primitives const &primitives)
      : BruteForce<typename DeviceType::memory_space>(
            typename DeviceType::execution_space{}, primitives)
  {
  }
  template <typename... Args>
  void query(Args &&... args) const
  {
    BruteForce<typename DeviceType::memory_space>::query(
        typename DeviceType::execution_space{}, std::forward<Args>(args)...);
  }
};

namespace Details
{
// NOTE  Working around the fact that CUDA does not allow the use of lambdas in
// a constructor:
// Error: cannot take address of enclosing function
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/#extended-lambda-restrictions
// Possible improvement would be tag dispatching with copy for boxes and
// assigment of Box{p(i), p(i)} for points.
template <typename ExecutionSpace, typename Primitives,
          typename BoundingVolumes>
void initializeBoundingVolumes(ExecutionSpace const &space,
                               Primitives const &primitives,
                               BoundingVolumes &bounding_volumes)
{
  using Access = typename Traits::Access<Primitives, Traits::PrimitivesTag>;
  auto size = bounding_volumes.extent(0);
  Kokkos::parallel_for(ARBORX_MARK_REGION("not_a_very_useful_name"),
                       Kokkos::RangePolicy<ExecutionSpace>(space, 0, size),
                       KOKKOS_LAMBDA(int i) {
                         Details::expand(bounding_volumes(i),
                                         Access::get(primitives, i));
                       });
}
} // namespace Details

template <typename MemorySpace, typename Enable>
template <typename ExecutionSpace, typename Primitives>
BruteForce<MemorySpace, Enable>::BruteForce(ExecutionSpace const &space,
                                    Primitives const &primitives)
    : _size{Traits::Access<Primitives, Traits::PrimitivesTag>::size(primitives)}
    , _bounding_volumes{
          Kokkos::ViewAllocateWithoutInitializing("bounding_volumes"), _size}
{
  // placeholder for concept check

  // NOTE  In principle could assign the bounding volumes within the reduction
  // determine the bounding box of the scene
  Details::TreeConstruction::calculateBoundingBoxOfTheScene(space, primitives,
                                                            _bounds);

  // compute bounding volumes of all objects
  Kokkos::deep_copy(/*space,*/ _bounding_volumes, Box{});
  Details::initializeBoundingVolumes(space, primitives, _bounding_volumes);
}

} // namespace ArborX


#endif
