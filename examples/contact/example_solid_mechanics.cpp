#include <ArborX.hpp>
#include <ArborX_BruteForce.hpp>

#include <Kokkos_Core.hpp>

#include <fstream>
#include <iostream>
#include <numeric>
#include <vector>

std::istream &operator>>(std::istream &is, ArborX::Point &p)
{
  char ignore;
  is >> ignore >> p[0] >> ignore >> p[1] >> ignore >> p[2] >> ignore;
  return is;
}

std::istream &operator>>(std::istream &is, ArborX::Box &b)
{
  char ignore;
  is >> ignore >> b.minCorner() >> ignore >> ignore >> b.maxCorner() >> ignore;
  return is;
}

std::vector<ArborX::Box> parse_boxes(std::string const &filename)
{
  std::vector<ArborX::Box> v;
  std::ifstream input(filename);
  input.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  ArborX::Box b;
  while (input >> b)
  {
    v.push_back(b);
  }
  return v;
}

template <typename... P, typename T>
auto vec2view(std::vector<T> const &in, std::string const &label = "")
{
  Kokkos::View<T *, P...> out(
      Kokkos::view_alloc(label, Kokkos::WithoutInitializing), in.size());
  Kokkos::deep_copy(out, Kokkos::View<T *, Kokkos::HostSpace,
                                      Kokkos::MemoryTraits<Kokkos::Unmanaged>>{
                             const_cast<T *>(in.data()), in.size()});
  return out;
}

template <typename View>
struct Wrapped
{
  View _M_view;
};

template <typename View>
auto wrap(View v)
{
  return Wrapped<View>{v};
}

namespace ArborX
{
namespace Traits
{
template <typename View>
struct Access<Wrapped<View>, PredicatesTag>
{
  using memory_space = typename View::memory_space;
  static size_t size(Wrapped<View> const &w) { return w._M_view.extent(0); }
  static KOKKOS_FUNCTION auto get(Wrapped<View> const &w, size_t i)
  {
    return attach(intersects(w._M_view(i)), (int)i);
  }
};
} // namespace Traits
} // namespace ArborX

struct Csr2Coo
{
  using tag = ArborX::Details::InlineCallbackTag;
  template <typename Predicate, typename OutputFunctor>
  KOKKOS_FUNCTION void operator()(Predicate const &pred, int index,
                                  OutputFunctor const &out) const
  {
    out(Kokkos::pair<int, int>{index, getData(pred)});
  }
};

void test_parse_boxes()
{
  auto v = parse_boxes("/scratch/dice");

  for (auto const &x : v)
  {
    std::cout << '{' << x.minCorner()[0] << ',' << x.minCorner()[1] << ','
              << x.minCorner()[2] << '}';
    std::cout << "->";
    std::cout << '{' << x.maxCorner()[0] << ',' << x.maxCorner()[1] << ','
              << x.maxCorner()[2] << '}';
    std::cout << '\n';
  }
  std::cout << v.size() << '\n';
}

template <typename DeviceType>
void run(std::string const &primitives_filename,
         std::string const &predicates_filename, int repetitions = 10)
{
  auto const primitives_boxes = parse_boxes(primitives_filename);
  auto const predicates_boxes = parse_boxes(predicates_filename);

  Kokkos::Timer timer;

  std::vector<double> durations;
  durations.reserve(repetitions);

  int drop = 1;

  for (int r = 0; r < repetitions + drop; ++r)
  {
    timer.reset();

    // copy boxes to the device
    auto const primitives =
        vec2view<DeviceType>(primitives_boxes, "primitives");
    auto const predicates =
        wrap(vec2view<DeviceType>(predicates_boxes, "predicates"));

    // build tree
    //ArborX::BVH<DeviceType> bvh(primitives);
    ArborX::BruteForce<DeviceType> bvh(primitives);

    // perform queries
    Kokkos::View<Kokkos::pair<int, int> *, DeviceType> results(
        Kokkos::view_alloc("results", Kokkos::WithoutInitializing), 0);
    Kokkos::View<int *, DeviceType> offset(
        Kokkos::view_alloc("offset", Kokkos::WithoutInitializing), 0);
    bvh.query(predicates, Csr2Coo{}, results, offset);

    // copy results back to the host
    auto const n_results = results.extent(0);
    auto const n_queries = offset.extent(0) - 1;
    std::vector<std::pair<int, int>> interactions(n_results);
    Kokkos::deep_copy(
        Kokkos::View<Kokkos::pair<int, int> *, Kokkos::HostSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
            reinterpret_cast<Kokkos::pair<int, int> *>(interactions.data()),
            interactions.size()),
        results);

    durations.emplace_back(timer.seconds());

    std::cout << bvh.size() << " (tree size)  ";
    std::cout << n_queries << "  (query size)  ";
    std::cout << n_results << " (num interactions)  ";
    std::cout << durations.back() << " seconds\n";
    if (r < drop)
    {
      durations.pop_back();
    }
  }

  double const total =
      std::accumulate(std::begin(durations), std::end(durations), 0.);
  double const average = total / repetitions;

  double const stddev =
      std::sqrt(std::inner_product(std::begin(durations), std::end(durations),
                                   std::begin(durations), 0.,
                                   [](double x, double y) { return x + y; },
                                   [average](double x, double y) {
                                     return (x - average) * (y - average);
                                   }) /
                (repetitions - 1));

  std::cout << "total time " << total << " (" << repetitions
            << " repetitions)\n";
  std::cout << "average " << average << '\n';
  std::cout << "standard deviation " << stddev << '\n';
}

int main(int argc, char *argv[])
{
  Kokkos::initialize(argc, argv);
  {
    // test_parse_boxes();

    using device_type = Kokkos::Device<Kokkos::Cuda, Kokkos::CudaSpace>;

    std::string const prefix = "/home/gainara/arborx/benchmark/bbox_data/";

    run<device_type>(prefix + "gears_assembly.txt_dice_1.0",
                     prefix + "gears_assembly.txt_tool_1.0", 20);

    run<device_type>(prefix + "jenga.txt_dice_1.0",
                     prefix + "jenga.txt_tool_1.0", 20);

    run<device_type>(prefix + "newtoncradle_final_3balldrop.txt_dice_1.0",
                     prefix + "newtoncradle_final_3balldrop.txt_tool_1.0", 20);
  }
  Kokkos::finalize();
  return 0;
}
