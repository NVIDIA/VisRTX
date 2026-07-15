// Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include "TextureStats.h"

#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform_reduce.h>

#include <cfloat>
#include <cmath>
#include <cstdint>

namespace visrtx {

namespace {

__device__ inline float srgbToLinear(float v)
{
  return v <= 0.04045f ? v / 12.92f : powf((v + 0.055f) / 1.055f, 2.4f);
}

// Device-friendly POD mirror of TexelAccum (plain arrays so thrust can hold it
// in registers during the fold). Identity element: zero sums, zero maxAbs,
// +inf minValue, finite.
struct Accum4
{
  double posSum[4];
  float maxAbs[4];
  float minValue[4];
  int finite;
};

__host__ __device__ inline Accum4 identityAccum()
{
  Accum4 a;
  for (int c = 0; c < 4; ++c) {
    a.posSum[c] = 0.0;
    a.maxAbs[c] = 0.0f;
    a.minValue[c] = FLT_MAX;
  }
  a.finite = 1;
  return a;
}

// Decode one texel's nc channels to linear and fold them into an Accum4. Only
// the touched channels [0,nc) are set; the rest keep the identity so unused
// channels never pollute the reduction.
struct DecodeTexel
{
  const void *data;
  TexelFormat fmt;
  int nc;
  int colorChannels;

  __device__ Accum4 operator()(std::size_t i) const
  {
    Accum4 a = identityAccum();
    for (int c = 0; c < nc; ++c) {
      float v;
      if (fmt == TexelFormat::Float32) {
        v = static_cast<const float *>(data)[i * nc + c];
      } else {
        const float u = static_cast<const uint8_t *>(data)[i * nc + c] / 255.0f;
        v = (fmt == TexelFormat::Srgb8 && c < colorChannels) ? srgbToLinear(u)
                                                             : u;
      }
      a.posSum[c] = fmaxf(v, 0.0f);
      a.maxAbs[c] = fabsf(v);
      a.minValue[c] = v;
      if (!isfinite(v))
        a.finite = 0;
    }
    return a;
  }
};

struct MergeAccum
{
  __host__ __device__ Accum4 operator()(const Accum4 &x, const Accum4 &y) const
  {
    Accum4 r;
    for (int c = 0; c < 4; ++c) {
      r.posSum[c] = x.posSum[c] + y.posSum[c];
      r.maxAbs[c] = fmaxf(x.maxAbs[c], y.maxAbs[c]);
      r.minValue[c] = fminf(x.minValue[c], y.minValue[c]);
    }
    r.finite = x.finite & y.finite;
    return r;
  }
};

} // namespace

TexelAccum reduceTexelsDevice(const void *deviceData,
    TexelFormat fmt,
    int nc,
    int colorChannels,
    std::size_t count)
{
  const Accum4 folded = thrust::transform_reduce(thrust::device,
      thrust::counting_iterator<std::size_t>(0),
      thrust::counting_iterator<std::size_t>(count),
      DecodeTexel{deviceData, fmt, nc, colorChannels},
      identityAccum(),
      MergeAccum{});

  TexelAccum out;
  for (int c = 0; c < 4; ++c) {
    out.posSum[c] = folded.posSum[c];
    out.maxAbs[c] = folded.maxAbs[c];
    out.minValue[c] = folded.minValue[c];
  }
  out.finite = folded.finite != 0;
  return out;
}

} // namespace visrtx
