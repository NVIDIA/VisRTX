// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include "sobol/contractProcess.h"
#include "sobol/contracts.h"

int main(int argc, char **argv)
{
  if (argc == 2) {
    const auto in = contract_test::parse(argv[1]);
    int cursor = in.cursor;
    visrtx::NextRayQmc qmc{{0.125f, 0.25f, 0.5f, 0.75f}, in.n};
    visrtx::RandState rs{42u, 55u};
    volatile float sample = in.kind >= 2
        ? visrtx::nextRayUniform(&rs, in.kind == 2 ? &qmc : nullptr, cursor)
        : in.kind == 1 ? visrtx::owenSobol(in.index, in.dimension, 123u)
                       : float(visrtx::sobolUInt(in.index, in.dimension));
    (void)sample;
    return 0;
  }
  if (!validSamplerContracts()) {
    std::fprintf(stderr, "valid sampler/PCG contract failed\n");
    return 1;
  }
  // Only SIGABRT is an expected host failure, never arbitrary nonzero exit.
  return contract_test::eachMisuse([&](const char *mode) {
    return contract_test::run(argv[0], mode, 10, [](int status) {
      return WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT;
    });
  })
      ? 0
      : 1;
}
