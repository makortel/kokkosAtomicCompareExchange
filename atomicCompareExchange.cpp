#include <Kokkos_Core.hpp>

#include <iostream>

using KokkosExecSpace = Kokkos::Serial;
//using KokkosExecSpace = Kokkos::Threads;
//using KokkosExecSpace = Kokkos::Cuda;

struct Ptr {
  double *ptr = nullptr;
};

void test() {
  auto data1 = Kokkos::View<Ptr*, KokkosExecSpace>("data1", 10);
  auto data2 = Kokkos::View<double*, KokkosExecSpace>("data2", 10);
  Kokkos::parallel_for(Kokkos::RangePolicy<KokkosExecSpace>(KokkosExecSpace(), 0, 10),
                       KOKKOS_LAMBDA(size_t i) {
                         using ptrAsInt = unsigned long long;
                         auto zero = (ptrAsInt)nullptr;
                         bool shouldChange = (data1[i].ptr == nullptr);
                         auto oldVal = data1[i].ptr;
                         Kokkos::atomic_compare_exchange((ptrAsInt*)(&data1[i].ptr),
                                                         zero,
                                                         (ptrAsInt)&data2[i]);
                         bool wasChanged = (data1[i].ptr != nullptr);
                         if (shouldChange != wasChanged) {
                           printf("shouldChange %d wasChanged %d old %llx new %llx zero %llx new %llx\n",
                                  shouldChange, wasChanged,
                                  (ptrAsInt)oldVal, (ptrAsInt)data1[i].ptr, zero, (ptrAsInt)&data2[i]);
                         }
                       });
}


int main() {
  Kokkos::initialize();
  test();
  Kokkos::finalize();
  return 0;
}
