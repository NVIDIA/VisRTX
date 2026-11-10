// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "../haltonReference.h"
#include "gpu/halton.h"
#include "gpu/owenSobol.h"

#ifdef SOBOL_CUDA
#include "cudaResources.h"

#include <cuda_runtime.h>
#endif

#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace sobol_test {

// Compile the unmodified pinned FastOwen excerpt as an independent host oracle.
// Reference bit reversal deliberately uses a simple bit-at-a-time construction.
namespace pinned {
static uint32_t ReverseBits32(uint32_t v)
{
  uint32_t result = 0;
  for (unsigned bit = 0; bit < 32; ++bit) {
    result = (result << 1) | (v & 1u);
    v >>= 1;
  }
  return result;
}
#define PBRT_CPU_GPU
#include "FastOwenScrambler.reference.txt"
#undef PBRT_CPU_GPU
} // namespace pinned

using Columns = std::array<std::array<uint32_t, 32>, 16>;

static void require(bool condition, const std::string &message)
{
  if (!condition)
    throw std::runtime_error(message);
}

static bool sameFloat(float a, float b)
{
  return std::memcmp(&a, &b, sizeof(float)) == 0;
}

static Columns referenceColumns()
{
  Columns result{};
  for (int i = 0; i < 32; ++i)
    result[0][i] = uint32_t(1) << (31 - i);
  std::ifstream input(SOBOL_PARAMETER_FILE);
  require(bool(input), "cannot open Joe-Kuo parameters");
  std::string line;
  // Attribution comments followed by the original column header.
  for (int i = 0; i < 4; ++i)
    std::getline(input, line);
  for (unsigned dimension = 1; dimension < 16; ++dimension) {
    unsigned d, s, a;
    require(bool(input >> d >> s >> a) && d == dimension + 1 && s <= 6,
        "invalid parameter row");
    auto &v = result[dimension];
    for (unsigned i = 0; i < s; ++i) {
      uint32_t numerator;
      require(bool(input >> numerator), "missing numerator");
      v[i] = numerator << (31 - i);
    }
    for (unsigned i = s; i < 32; ++i) {
      v[i] = v[i - s] ^ (v[i - s] >> s);
      for (unsigned k = 1; k < s; ++k)
        if (a & (1u << (s - 1 - k)))
          v[i] ^= v[i - k];
    }
  }
  return result;
}

struct Input
{
  uint32_t index, dimension, seed;
  uint32_t x = 0, y = 0;
  int bounce = 0, transparency = 0;
  bool pixelSeedInput = false;
};

struct Output
{
  uint32_t pixelKey, vertexKey, raw, reversed, scrambled;
  float sample, converted;
  ::float2 hemi;
  uint32_t haltonKey;
  ::float4 camera, rawCamera;
};

// Both executables call exactly these production seams. The independent
// expectations below come from pinned parameters and offline reference vectors.
VISRTX_HOST_DEVICE Output evaluate(Input in)
{
  const auto pixelKey = in.pixelSeedInput
      ? visrtx::owenPixelSeed(in.x, in.y)
      : in.seed;
  const auto vertexKey =
      visrtx::owenBounceSeed(pixelKey, in.bounce, in.transparency);
  const auto raw = visrtx::sobolUInt(in.index, in.dimension);
  const auto dimSeed = vertexKey ^ (0x9e3779b9u * (in.dimension + 1u));
  return {pixelKey,
      vertexKey,
      raw,
      visrtx::reverseBits32(raw),
      visrtx::owenScramble(raw, dimSeed),
      visrtx::owenSobol(in.index, in.dimension, vertexKey),
      visrtx::sobolFloat(in.index),
      visrtx::owenSobolHemi(in.index, vertexKey),
      visrtx::haltonPixelHash(in.x, in.y),
      visrtx::halton4D(in.index + visrtx::haltonPixelHash(in.x, in.y)),
      visrtx::halton4D(in.index)};
}

// Halton is float arithmetic, not an integer Sobol grid: allow 4 float ulps
// at unit scale for accumulated reciprocal rounding (including CUDA FMA).
static void checkCamera(Input in, const Output &out)
{
  const ::float4 points[] = {out.rawCamera, out.camera};
  const uint32_t indices[] = {in.index, in.index + out.haltonKey};
  for (unsigned p = 0; p < 2; ++p) {
    const auto expected = halton_reference::point(indices[p]);
    const float actual[] = {points[p].x, points[p].y, points[p].z, points[p].w};
    for (unsigned c = 0; c < 4; ++c)
      require(std::isfinite(actual[c]) && actual[c] >= 0.f && actual[c] < 1.f
              && std::abs(actual[c] - expected[c]) <= 0x1p-21,
          "Halton rational oracle component " + std::to_string(c) + " index "
              + std::to_string(indices[p]));
  }
}

static void checkHalton(std::vector<Input> &inputs)
{
  // Hand-worked base fractions: index 1 and index 8 (1000b, 22_3, 13_5, 11_7).
  const double answers[][4] = {{0, 0, 0, 0},
      {1. / 2, 1. / 3, 1. / 5, 1. / 7},
      {1. / 16, 8. / 9, 16. / 25, 8. / 49}};
  unsigned row = 0;
  for (uint32_t index : {0u, 1u, 8u}) {
    const auto p = visrtx::halton4D(index);
    const float a[] = {p.x, p.y, p.z, p.w};
    for (unsigned c = 0; c < 4; ++c)
      require(std::abs(a[c] - answers[row][c]) < 0x1p-21,
          "Halton known-answer component " + std::to_string(c));
    ++row;
  }
  struct Pixel
  {
    uint32_t x, y, hash;
  };
  const Pixel pixels[] = {{0, 0, 0xf1a85082u},
      {1, 2, 0x339b28e2u},
      {1920, 1080, 0xeebd993au},
      {0xffffffffu, 0xffffffffu, 0x4ab06c79u}};
  for (const auto &p : pixels) {
    require(
        visrtx::haltonPixelHash(p.x, p.y) == p.hash, "Halton fixed pixel hash");
    for (uint32_t index : {0u,
             1u,
             8u,
             0x7fffffffu,
             0x80000000u,
             0xfffffffeu,
             0xffffffffu,
             uint32_t(-p.hash - 1u),
             uint32_t(-p.hash),
             uint32_t(1u - p.hash)}) {
      Input in{index, 0, 0};
      in.x = p.x;
      in.y = p.y;
      const auto out = evaluate(in);
      checkCamera(in, out);
      // Boundary vectors force the effective index to UINT_MAX, zero, one.
      if (index == uint32_t(-p.hash))
        require(out.camera.x == 0.f && out.camera.y == 0.f
                && out.camera.z == 0.f && out.camera.w == 0.f,
            "Halton additive offset wraps to zero");
      inputs.push_back(in);
    }
  }
  std::cout
      << "Halton 4D known answers, large indices and hash wrapping pass\n";
}

#ifdef SOBOL_CUDA
static void cudaCheck(cudaError_t error)
{
  require(error == cudaSuccess, cudaGetErrorString(error));
}

VISRTX_GLOBAL void sampleKernel(const Input *inputs, Output *outputs, size_t n)
{
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    outputs[i] = evaluate(inputs[i]);
}

static void checkDevice(const std::vector<Input> &inputs)
{
  int count = 0;
  cudaCheck(cudaGetDeviceCount(&count));
  require(count > 0, "CUDA device required; missing device is not a pass");
  cudaDeviceProp properties{};
  cudaCheck(cudaGetDeviceProperties(&properties, 0));
  int driver, runtime;
  cudaCheck(cudaDriverGetVersion(&driver));
  cudaCheck(cudaRuntimeGetVersion(&runtime));
  std::cout << "CUDA " << properties.name << ", driver " << driver
            << ", runtime " << runtime << '\n';
  CudaBuffer<Input> deviceInputs;
  CudaBuffer<Output> deviceOutputs;
  cudaCheck(deviceInputs.allocate(inputs.size()));
  cudaCheck(deviceOutputs.allocate(inputs.size()));
  cudaCheck(cudaMemcpy(deviceInputs.get(),
      inputs.data(),
      inputs.size() * sizeof(Input),
      cudaMemcpyHostToDevice));
  sampleKernel<<<(inputs.size() + 127) / 128, 128>>>(
      deviceInputs.get(), deviceOutputs.get(), inputs.size());
  cudaCheck(cudaGetLastError());
  cudaCheck(cudaDeviceSynchronize());
  std::vector<Output> outputs(inputs.size());
  cudaCheck(cudaMemcpy(outputs.data(),
      deviceOutputs.get(),
      outputs.size() * sizeof(Output),
      cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < inputs.size(); ++i) {
    const auto a = evaluate(inputs[i]);
    const auto b = outputs[i];
    require(a.haltonKey == b.haltonKey, "Halton host/CUDA hash mismatch");
    checkCamera(inputs[i], b);
    require(a.pixelKey == b.pixelKey && a.vertexKey == b.vertexKey
            && a.raw == b.raw && a.reversed == b.reversed
            && a.scrambled == b.scrambled && sameFloat(a.sample, b.sample)
            && sameFloat(a.converted, b.converted)
            && sameFloat(a.hemi.x, b.hemi.x) && sameFloat(a.hemi.y, b.hemi.y),
        "host/device mismatch at vector " + std::to_string(i));
  }
  std::cout << inputs.size() << " exact host/CUDA vector comparisons\n";
}
#endif

inline int run()
try {
  // Direct endpoints exercise production conversion, not accidental samples.
  require(visrtx::sobolFloat(0) == 0.f && visrtx::sobolFloat(0xffu) == 0.f
          && visrtx::sobolFloat(0x100u) == 0x1p-24f
          && visrtx::sobolFloat(0xffffffffu) == 0x1.fffffep-1f,
      "24-bit conversion endpoints");
  require(visrtx::reverseBits32(0) == 0
          && visrtx::reverseBits32(1) == 0x80000000u
          && visrtx::reverseBits32(0x01234567) == 0xe6a2c480u
          && visrtx::reverseBits32(0xffffffff) == 0xffffffff,
      "known-answer bit reversal");
  const auto reference = referenceColumns();
  std::vector<Input> inputs;
  checkHalton(inputs);
  unsigned failures = 0;
  for (unsigned d = 0; d < 16; ++d) {
    for (unsigned bit = 0; bit < 32; ++bit) {
      const uint32_t index = uint32_t(1) << bit;
      const auto actual = visrtx::sobolUInt(index, d);
      if (actual != reference[d][bit]) {
        std::cerr << "dimension " << d << ", column " << bit << ": actual "
                  << std::hex << actual << ", expected " << reference[d][bit]
                  << std::dec << '\n';
        ++failures;
      }
      inputs.push_back({index, d, 0});
    }
  }
  require(failures == 0, "direction columns disagree with independent oracle");
  require(visrtx::sobolUInt(16, 7) == 0x88000000u,
      "dimension 7 column 4 regression");

  std::ifstream vectors(SOBOL_VECTOR_FILE);
  require(bool(vectors), "cannot open known-answer vectors");
  std::string line;
  std::getline(vectors, line);
  std::getline(vectors, line);
  Input in{};
  uint32_t raw, reversed, scrambled, grid;
  unsigned vectorCount = 0;
  while (vectors >> std::hex >> in.index >> in.dimension >> in.seed >> raw
      >> reversed >> scrambled >> grid) {
    const auto actual = evaluate(in);
    require(actual.raw == raw, "binary-index integer vector");
    require(actual.reversed == reversed, "bit reversal vector");
    require(pinned::FastOwenScrambler(
                in.seed ^ (0x9e3779b9u * (in.dimension + 1u)))(raw)
            == scrambled,
        "offline fixture disagrees with pinned FastOwen operations");
    require(actual.scrambled == scrambled, "FastOwen integer vector");
    // All integers <= 2^24-1 are exactly representable as float; scaling by
    // a power of two is exact. Float expectations are separate from scramble.
    require(sameFloat(actual.sample, std::ldexp(float(grid), -24)),
        "float grid vector");
    require(actual.sample >= 0 && actual.sample < 1
            && std::ldexp(actual.sample, 24)
                == std::floor(std::ldexp(actual.sample, 24)),
        "sample outside exact half-open grid");
    require(actual.hemi.x == visrtx::owenSobol(in.index, 0, in.seed)
            && actual.hemi.y == visrtx::owenSobol(in.index, 1, in.seed),
        "transport hemisphere component mapping");
    inputs.push_back(in);
    ++vectorCount;
  }
  require(
      vectors.eof() && vectorCount == 480, "incomplete known-answer fixture");
  for (uint32_t seed : {0u, 0x12345678u, 0xffffffffu}) {
    for (unsigned d = 0; d < 16; ++d) {
      for (unsigned m = 1; m <= 10; ++m) {
        const unsigned n = 1u << m;
        std::vector<bool> occupied(n, false);
        for (unsigned i = 0; i < n; ++i) {
          const auto sample = visrtx::owenSobol(i, d, seed);
          const unsigned bin = unsigned(sample * n);
          require(bin < n && !occupied[bin], "dyadic 1D stratum repeated");
          occupied[bin] = true;
        }
      }
    }
  }
  for (uint32_t endpoint : {0u, 0xffu, 0x100u, 0xffffffffu})
    inputs.push_back({endpoint, 0, 0});
#ifdef SOBOL_CUDA
  checkDevice(inputs);
#endif
  std::cout << "512 columns, 480 vectors, endpoints and all 1D strata pass\n";
  return 0;
} catch (const std::exception &e) {
  std::cerr << e.what() << '\n';
  return 1;
}

} // namespace sobol_test
