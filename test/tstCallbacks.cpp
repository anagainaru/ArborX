#include <ArborX_AccessTraits.hpp>
#include <ArborX_Callbacks.hpp>
#include <ArborX_Predicates.hpp>

struct NearestPredicates
{
};

struct SpatialPredicates
{
};

namespace ArborX
{
template <>
struct AccessTraits<NearestPredicates, PredicatesTag>
{
  using memory_space = Kokkos::HostSpace;
  static int size(NearestPredicates const &) { return 1; }
  static auto get(NearestPredicates const &, int) { return nearest(Point{}); }
};
template <>
struct AccessTraits<SpatialPredicates, PredicatesTag>
{
  using memory_space = Kokkos::HostSpace;
  static int size(SpatialPredicates const &) { return 1; }
  static auto get(SpatialPredicates const &, int)
  {
    return intersects(Point{});
  }
};
} // namespace ArborX

// Custom callbacks
struct PredicateCallbackMissingTag
{
  template <typename Predicate, typename OutputFunctor>
  void operator()(Predicate const &, int, OutputFunctor const &) const
  {
  }
};

struct Wrong
{
};

struct PredicateCallbackDoesNotTakeCorrectArgument
{
  template <typename OutputFunctor>
  void operator()(Wrong, int, OutputFunctor const &) const
  {
  }
};

struct CustomCallbackPredicate
{
  template <class Predicate>
  KOKKOS_FUNCTION void operator()(Predicate const &, int) const
  {
  }
};

struct CustomCallbackPredicateMissingConstQualifier
{
  template <class Predicate>
  KOKKOS_FUNCTION void operator()(Predicate const &, int)
  {
  }
};

struct CustomCallbackPredicateNonVoidReturnType
{
  template <class Predicate>
  KOKKOS_FUNCTION auto operator()(Predicate const &, int) const
  {
    return Wrong{};
  }
};

int main()
{
  using ArborX::Details::check_valid_callback;

  // view type does not matter as long as we do not call the output functor
  Kokkos::View<float *> v;

  check_valid_callback(ArborX::Details::CallbackDefaultPredicate{},
                       SpatialPredicates{}, v);
  check_valid_callback(ArborX::Details::CallbackDefaultPredicate{},
                       NearestPredicates{}, v);

  // not required to tag inline callbacks any more
  check_valid_callback(PredicateCallbackMissingTag{}, SpatialPredicates{}, v);
  check_valid_callback(PredicateCallbackMissingTag{}, NearestPredicates{}, v);

  check_valid_callback(CustomCallbackPredicate{}, SpatialPredicates{});
  check_valid_callback(CustomCallbackPredicate{}, NearestPredicates{});

  // generic lambdas are supported if not using NVCC
#ifndef __NVCC__
  check_valid_callback([](auto const & /*predicate*/, int /*primitive*/,
                          auto const & /*out*/) {},
                       SpatialPredicates{}, v);

  check_valid_callback([](auto const & /*predicate*/, int /*primitive*/,
                          auto const & /*out*/) {},
                       NearestPredicates{}, v);

  check_valid_callback([](auto const & /*predicate*/, int /*primitive*/) {},
                       SpatialPredicates{});

  check_valid_callback([](auto const & /*predicate*/, int /*primitive*/) {},
                       NearestPredicates{});
#endif

  // Uncomment to see error messages

  // check_valid_callback(PredicateCallbackDoesNotTakeCorrectArgument{},
  //                     SpatialPredicates{}, v);

  // check_valid_callback(CustomCallbackPredicateNonVoidReturnType{},
  //                     SpatialPredicates{});

  // check_valid_callback(CustomCallbackPredicateMissingConstQualifier{},
  //                     SpatialPredicates{});

#ifndef __NVCC__
  // check_valid_callback([](Wrong, int /*primitive*/) {}, SpatialPredicates{});
#endif
}
