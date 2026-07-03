// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/algorithms/cuda/boxOutline.hpp"
// thrust
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
// std
#include <algorithm>
#include <cmath>

namespace tsd::algorithms::cuda {

namespace {

// Fraction of the fragment's eye distance subtracted before the depth
// compare; the box typically touches the geometry it bounds, so it would
// z-fight without a relative bias toward the camera.
constexpr float DEPTH_BIAS = 1e-3f;

struct ProjectedEdge
{
  tsd::math::float2 s0, s1; // pixel-space endpoints
  float invW0, invW1; // 1/clip.w for perspective-correct interpolation
  tsd::math::float3 world0, world1; // world-space endpoints
};

struct ProjectedEdges
{
  int count{0};
  ProjectedEdge e[12];
};

// Project the 12 box edges to pixel space, clipping the portion of each edge
// behind the eye (clip.w <= 0, perspective only; w == 1 for orthographic).
// Runs on the host; the result is captured by value into the device lambda.
ProjectedEdges projectBoxEdges(const tsd::math::box3 &box,
    const tsd::math::mat4 &projView,
    uint32_t width,
    uint32_t height)
{
  using math_float2 = tsd::math::float2;
  using math_float3 = tsd::math::float3;
  using math_float4 = tsd::math::float4;

  math_float3 corners[8];
  math_float4 clip[8];
  for (int i = 0; i < 8; i++) {
    corners[i] = math_float3(i & 1 ? box.upper.x : box.lower.x,
        i & 2 ? box.upper.y : box.lower.y,
        i & 4 ? box.upper.z : box.lower.z);
    clip[i] = tsd::math::mul(projView, math_float4(corners[i], 1.f));
  }

  constexpr float W_EPS = 1e-6f;

  ProjectedEdges result;
  for (int i = 0; i < 8; i++) {
    for (int bit = 1; bit <= 4; bit <<= 1) {
      if (i & bit)
        continue;
      const int j = i | bit;

      // Clip t-range of the segment to clip.w >= W_EPS //

      const float w0 = clip[i].w;
      const float w1 = clip[j].w;
      if (w0 < W_EPS && w1 < W_EPS)
        continue;

      float t0 = 0.f;
      float t1 = 1.f;
      if (w0 < W_EPS)
        t0 = (W_EPS - w0) / (w1 - w0);
      else if (w1 < W_EPS)
        t1 = (W_EPS - w0) / (w1 - w0);

      const math_float4 c0 = tsd::math::lerp(clip[i], clip[j], t0);
      const math_float4 c1 = tsd::math::lerp(clip[i], clip[j], t1);

      auto toScreen = [&](const math_float4 &c) {
        const math_float2 ndc(c.x / c.w, c.y / c.w);
        return math_float2((ndc.x * 0.5f + 0.5f) * width,
            (ndc.y * 0.5f + 0.5f) * height);
      };

      auto &edge = result.e[result.count++];
      edge.s0 = toScreen(c0);
      edge.s1 = toScreen(c1);
      edge.invW0 = 1.f / c0.w;
      edge.invW1 = 1.f / c1.w;
      edge.world0 = tsd::math::lerp(corners[i], corners[j], t0);
      edge.world1 = tsd::math::lerp(corners[i], corners[j], t1);
    }
  }

  return result;
}

} // namespace

void boxOutline(cudaStream_t stream,
    const tsd::math::box3 &box,
    const tsd::math::mat4 &projView,
    const tsd::math::float3 &eye,
    const tsd::math::float3 &dir,
    bool orthographicDepth,
    const float *depth,
    uint32_t *color,
    uint32_t outlineColor,
    uint32_t lineWidth,
    uint32_t width,
    uint32_t height)
{
  const ProjectedEdges edges = projectBoxEdges(box, projView, width, height);
  const float halfWidth = std::max(1u, lineWidth) * 0.5f;

  thrust::for_each(thrust::cuda::par.on(stream),
      thrust::make_counting_iterator(0u),
      thrust::make_counting_iterator(width * height),
      [=] __device__(uint32_t i) {
        using tsd::math::dot;
        // linalg::length is host-only; use dot + sqrtf in device code
        auto norm2 = [](const tsd::math::float2 &v) {
          return ::sqrtf(dot(v, v));
        };
        auto norm3 = [](const tsd::math::float3 &v) {
          return ::sqrtf(dot(v, v));
        };
        const tsd::math::float2 p(i % width + 0.5f, i / width + 0.5f);

        for (int e = 0; e < edges.count; e++) {
          const auto &edge = edges.e[e];
          const tsd::math::float2 d = edge.s1 - edge.s0;
          const float len2 = dot(d, d);
          const float t = len2 > 0.f
              ? ::fminf(::fmaxf(dot(p - edge.s0, d) / len2, 0.f), 1.f)
              : 0.f;
          const tsd::math::float2 closest = edge.s0 + t * d;
          if (norm2(p - closest) > halfWidth)
            continue;

          if (depth || orthographicDepth) {
            // Perspective-correct world position at t //
            const float iw0 = (1.f - t) * edge.invW0;
            const float iw1 = t * edge.invW1;
            const tsd::math::float3 world =
                (edge.world0 * iw0 + edge.world1 * iw1) / (iw0 + iw1);

            const float z = orthographicDepth ? dot(world - eye, dir)
                                              : norm3(world - eye);
            if (orthographicDepth && z < 0.f)
              continue; // behind the orthographic camera plane
            if (depth && z * (1.f - DEPTH_BIAS) > depth[i])
              continue;
          }

          color[i] = outlineColor;
          break;
        }
      });
}

void boxOutline(const tsd::math::box3 &box,
    const tsd::math::mat4 &projView,
    const tsd::math::float3 &eye,
    const tsd::math::float3 &dir,
    bool orthographicDepth,
    const float *depth,
    uint32_t *color,
    uint32_t outlineColor,
    uint32_t lineWidth,
    uint32_t width,
    uint32_t height)
{
  boxOutline(cudaStream_t{0},
      box,
      projView,
      eye,
      dir,
      orthographicDepth,
      depth,
      color,
      outlineColor,
      lineWidth,
      width,
      height);
}

} // namespace tsd::algorithms::cuda
