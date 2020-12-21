cmake_minimum_required (VERSION 3.10)
project (reproducer)
set(Kokkos_DIR "/home/mkortela/pixeltrack-standalone/external/kokkos/install/lib/cmake/Kokkos")
find_package(Kokkos REQUIRED)

#set(CMAKE_CXX_COMPILER "/home/mkortela/pixeltrack-standalone/external/kokkos/install/bin/nvcc_wrapper")

add_executable(RangePolicy rangePolicy.cpp)
target_link_libraries(RangePolicy Kokkos::kokkos)

add_executable(TeamPolicy teamPolicy.cpp)
target_link_libraries(TeamPolicy Kokkos::kokkos)

add_executable(AtomicCompareExchange atomicCompareExchange.cpp)
target_link_libraries(AtomicCompareExchange Kokkos::kokkos)
