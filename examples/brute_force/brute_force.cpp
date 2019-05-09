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

#include <ArborX_BruteForce.hpp>
#include <ArborX_DetailsBufferOptimization.hpp>
#include <ArborX_LinearBVH.hpp>

#include <Kokkos_Core.hpp>

struct Dummy
{
};

using MemorySpace = Kokkos::HostSpace;
using ExecutionSpace = Kokkos::DefaultHostExecutionSpace;

namespace ArborX
{
namespace Traits
{
template <>
struct Access<Dummy, PrimitivesTag>
{
  using memory_space = MemorySpace;
  using size_type = typename MemorySpace::size_type;
  inline static size_type size(Dummy const &) { return 5; }
  KOKKOS_FUNCTION static Point get(Dummy const &, size_type i)
  {
    return {{(float)i, (float)i, (float)i}};
  }
};

template <>
struct Access<Dummy, PredicatesTag>
{
  using memory_space = MemorySpace;
  using size_type = typename MemorySpace::size_type;
  inline static size_type size(Dummy const &) { return 5; }
  KOKKOS_FUNCTION static auto get(Dummy const &, size_type i)
  {
    return attach(
        intersects(Sphere{{{(float)i, (float)i, (float)i}}, (float)i}), i);
  }
};
} // namespace Traits
} // namespace ArborX

struct PrintfCallback
{
  using tag = ArborX::Details::InlineCallbackTag;
  Kokkos::View<int, ExecutionSpace, Kokkos::MemoryTraits<Kokkos::Atomic>> c_;
  PrintfCallback()
      : c_{"counter"}
  {
  }
  template <typename Predicate, typename OutputFunctor>
  KOKKOS_FUNCTION void operator()(Predicate const &predicate, int i,
                                  OutputFunctor const &out) const
  {
    int const j = getData(predicate);
    printf("%d callback (%d,%d)\n", ++c_(), i, j);
    out(i);
  }
};

template <typename T, typename... P>
std::vector<T> view2vec(Kokkos::View<T *, P...> view)
{
  std::vector<T> vec(view.size());
  Kokkos::deep_copy(Kokkos::View<T *, Kokkos::HostSpace,
                                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
                        vec.data(), vec.size()),
                    view);
  return vec;
}

template <typename OutputView, typename OffsetView>
void print(OutputView const out, OffsetView const offset)
{
  int const n_queries = offset.extent(0) - 1;

  auto const h_out = view2vec(out);
  auto const h_offset = view2vec(offset);
  int count = 0;

  for (int j = 0; j < n_queries; ++j)
    for (int k = h_offset[j]; k < h_offset[j + 1]; ++k)
      printf("%d result (%d, %d)\n", ++count, h_out[k], j);
}

int main(int argc, char *argv[])
{
  Kokkos::ScopeGuard guard(argc, argv);

  ExecutionSpace space{};
  Dummy primitives{};
  Dummy predicates{};

  printf("Bounding Volume Hierarchy\n");
  {
    ArborX::BVH<MemorySpace> bvh{space, primitives};

    Kokkos::View<int *, ExecutionSpace> indices("indices_ref", 0);
    Kokkos::View<int *, ExecutionSpace> offset("offset_ref", 0);
    bvh.query(space, predicates, PrintfCallback{}, indices, offset);

    printf("print\n");
    print(indices, offset);
  }

  printf("Brute Force\n");
  {
    ArborX::BruteForce<MemorySpace> brute{space, primitives};

    Kokkos::View<int *, ExecutionSpace> indices("indices", 0);
    Kokkos::View<int *, ExecutionSpace> offset("offset", 0);
    brute.query(space, predicates, PrintfCallback{}, indices, offset);

    printf("print\n");
    print(indices, offset);
  }

  return 0;
}
