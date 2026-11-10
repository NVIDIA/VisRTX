// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include "sobol/contractProcess.h"
#include "sobol/contracts.h"
#include "sobol/cudaResources.h"

#include <cuda_runtime.h>

namespace {
constexpr int expectedErrorExit = 42;

VISRTX_GLOBAL void baseline(int *result)
{
  *result = validSamplerContracts() ? 1 : 0;
}

VISRTX_GLOBAL void misuse(contract_test::Misuse in, float *result)
{
  using namespace visrtx;
  if (in.kind == 0)
    *result = float(sobolUInt(in.index, in.dimension));
  else if (in.kind == 1)
    *result = owenSobol(in.index, in.dimension, 123u);
  else {
    NextRayQmc qmc{{0.125f, 0.25f, 0.5f, 0.75f}, in.n};
    RandState rs{42u, 55u};
    *result = nextRayUniform(&rs, in.kind == 2 ? &qmc : nullptr, in.cursor);
  }
}

bool check(cudaError_t error, const char *operation)
{
  if (error == cudaSuccess)
    return true;
  std::fprintf(
      stderr, "%s: %s (%d)\n", operation, cudaGetErrorName(error), int(error));
  return false;
}

int child(const char *mode)
{
  int count = 0;
  if (!check(cudaGetDeviceCount(&count), "availability") || count == 0)
    return 1;
  CudaBuffer<int> result;
  if (!check(result.allocate(1), "allocation"))
    return 1;
  baseline<<<1, 1>>>(result.get());
  if (!check(cudaGetLastError(), "baseline launch")
      || !check(cudaDeviceSynchronize(), "baseline sync"))
    return 1;
  int value = 0;
  if (!check(
          cudaMemcpy(&value, result.get(), sizeof(int), cudaMemcpyDeviceToHost),
          "baseline copy")
      || value != 1)
    return 1;
  if (std::strcmp(mode, "baseline") == 0) {
    int runtime = 0, driver = 0;
    cudaRuntimeGetVersion(&runtime);
    cudaDriverGetVersion(&driver);
    std::printf(
        "CUDA baseline passed; runtime=%d driver=%d\n", runtime, driver);
    return 0;
  }

  const auto in = contract_test::parse(mode);
  auto *output = reinterpret_cast<float *>(result.get());
  // Relinquish cleanup before the trap: even failure returns below must not
  // make destructor CUDA calls into the disposable, poisoned context.
  result.abandon();
  misuse<<<1, 1>>>(in, output);
  if (!check(cudaGetLastError(), "negative launch"))
    return 1; // Prelaunch errors are never expected failures.
  const cudaError_t error = cudaDeviceSynchronize();
  // Reviewed Linux/CUDA allowlist: only execution assert/trap outcomes.
  // Memory-access, availability, initialization and unknown errors fail closed.
  if (error != cudaErrorAssert && error != cudaErrorIllegalInstruction
      && error != cudaErrorLaunchFailure) {
    std::fprintf(stderr,
        "%s: unexpected synchronization %s (%d)\n",
        mode,
        cudaGetErrorName(error),
        int(error));
    return 1;
  }
  std::printf("%s: expected synchronization %s (%d); child completes\n",
      mode,
      cudaGetErrorName(error),
      int(error));
  // Do not make any further CUDA calls or reuse the poisoned context.
  return expectedErrorExit;
}

bool run(const char *executable, const char *mode, int expected)
{
  return contract_test::run(executable, mode, 20, [=](int status) {
    return WIFEXITED(status) && WEXITSTATUS(status) == expected;
  });
}
} // namespace

int main(int argc, char **argv)
{
  if (argc == 2)
    return child(argv[1]);
  // Parent never initializes CUDA. Every case execs a fresh process/context,
  // establishes successful execution first, and must explicitly exit normally.
  if (!run(argv[0], "baseline", 0))
    return 1;
  return contract_test::eachMisuse([&](const char *mode) {
    return run(argv[0], mode, expectedErrorExit);
  })
      ? 0
      : 1;
}
