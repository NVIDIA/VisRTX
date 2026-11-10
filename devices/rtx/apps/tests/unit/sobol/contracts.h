// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "gpu/owenSobol.h"
#include "gpu/shadingState.h"

// Shared behavioral checks call the production overlay and PCG seams on both
// host and device. A cloned real PCG stream is the oracle for exactly one draw.
VISRTX_DEVICE bool validSamplerContracts()
{
  using namespace visrtx;
  for (int n = 0; n <= 4; ++n) {
    const NextRayQmc qmc{{0.125f, 0.25f, 0.5f, 0.75f}, n};
    for (int start = 0; start <= n; ++start) {
      RandState actual{0x123456789abcdef0ull, 55ull};
      RandState expected = actual;
      int cursor = start;
      for (int draw = 0; draw < 8; ++draw) {
        const bool supplied = start + draw < n;
        const float want =
            supplied ? qmc.u[start + draw] : pcg_uniform(&expected);
        const float got = nextRayUniform(&actual, &qmc, cursor);
        const int wantCursor = supplied ? start + draw + 1 : n;
        if (got != want || cursor != wantCursor
            || actual.state != expected.state || actual.inc != expected.inc)
          return false;
      }
      // A PCG pointer is not required on the supplied-value path.
      for (int i = start; i < n;) {
        const float want = qmc.u[i];
        const int old = i;
        if (nextRayUniform(nullptr, &qmc, i) != want || i != old + 1)
          return false;
      }
    }
  }
  RandState actual{0x123456789abcdef0ull, 55ull};
  RandState expected = actual;
  int cursor = 0;
  for (int i = 0; i < 8; ++i) {
    const float want = pcg_uniform(&expected);
    if (nextRayUniform(&actual, nullptr, cursor) != want || cursor != 0
        || actual.state != expected.state || actual.inc != expected.inc)
      return false;
  }
  // Ordinary dimension execution, including both table boundaries.
  return sobolUInt(1u, 0u) == 0x80000000u && sobolUInt(1u, 15u) == 0x80000000u;
}
