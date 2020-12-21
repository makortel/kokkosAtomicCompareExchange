#include <Kokkos_Core.hpp>

#include <iostream>

using KokkosExecSpace = Kokkos::Threads;
//using KokkosExecSpace = Kokkos::Serial;
//using KokkosExecSpace = Kokkos::Cuda;

namespace {
  constexpr int ELEMENTS = 20;
  constexpr int VALUE = 10;

  KOKKOS_INLINE_FUNCTION void kernel(Kokkos::View<int*, KokkosExecSpace> data, size_t i) {
    data[i] += VALUE;
  }
}  // namespace

void test() {
  Kokkos::View<int*, KokkosExecSpace> data_d("data_d", ELEMENTS);
  auto data_h = Kokkos::create_mirror_view(data_d);
  for (int i = 0; i < ELEMENTS; ++i) {
    data_h[i] = i;
  }
  Kokkos::deep_copy(KokkosExecSpace(), data_d, data_h);

  Kokkos::parallel_for(
                       Kokkos::RangePolicy<KokkosExecSpace>(KokkosExecSpace(), 0, ELEMENTS), KOKKOS_LAMBDA(size_t i) { kernel(data_d, i); });

  Kokkos::deep_copy(KokkosExecSpace(), data_h, data_d);

  KokkosExecSpace().fence();
  for (int i = 0; i < ELEMENTS; ++i) {
    std::cout << i << " " << data_h(i) << std::endl;
  }
}

int main() {
  Kokkos::initialize();
  test();
  Kokkos::finalize();
  return 0;
}
