#include <Kokkos_Core.hpp>

#include <iostream>

using KokkosExecSpace = Kokkos::Threads;
//using KokkosExecSpace = Kokkos::Serial;
//using KokkosExecSpace = Kokkos::Cuda;

namespace {
  constexpr int TEAMS = 10;
  constexpr int THREADS_PER_TEAM = 128;
  constexpr int ELEMENTS_PER_TEAM = 1000;
  constexpr int ELEMENTS = TEAMS * ELEMENTS_PER_TEAM;

  KOKKOS_INLINE_FUNCTION void kernel(Kokkos::View<int*, KokkosExecSpace> data,
                                     Kokkos::TeamPolicy<KokkosExecSpace>::member_type const& teamMember) {
    constexpr int tmp = ELEMENTS_PER_TEAM;
    Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, tmp), [=](int i) {
      data[teamMember.league_rank() * ELEMENTS_PER_TEAM + i] *= 10;
    });
  }
}  // namespace

void test() {
  Kokkos::View<int*, KokkosExecSpace> data_d("data_d", ELEMENTS);
  auto data_h = Kokkos::create_mirror_view(data_d);
  for (int i = 0; i < ELEMENTS; ++i) {
    data_h[i] = i;
  }
  Kokkos::deep_copy(KokkosExecSpace(), data_d, data_h);

  using TeamPolicy = Kokkos::TeamPolicy<KokkosExecSpace>;
  using MemberType = TeamPolicy::member_type;
  //TeamPolicy policy(KokkosExecSpace(), TEAMS, THREADS_PER_TEAM);
  TeamPolicy policy(KokkosExecSpace(), TEAMS, Kokkos::AUTO());
  Kokkos::parallel_for(
      policy, KOKKOS_LAMBDA(MemberType const& teamMember) { kernel(data_d, teamMember); });

  Kokkos::deep_copy(KokkosExecSpace(), data_h, data_d);

  KokkosExecSpace().fence();
  for (int iTeam = 0; iTeam != TEAMS; ++iTeam) {
    for (int i = 0; i < 10; ++i) {
      std::cout << "Team " << iTeam << " element " << i << " " << data_h[iTeam * ELEMENTS_PER_TEAM + i] << std::endl;
    }
  }
}

int main() {
  Kokkos::initialize();
  test();
  Kokkos::finalize();
  return 0;
}
