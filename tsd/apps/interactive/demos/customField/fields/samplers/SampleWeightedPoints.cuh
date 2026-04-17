// Copyright 2025-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../WeightedPointsFieldData.h"
#include "gpu/gpu_math.h"

// Traverses a flattened BFS octree with Gaussian kernel and
// cutoff-based LOD.  Each node stores (x, y, z, value) where
// value = sum of subtree weights, position = weighted centroid.
//
// LOD: at each internal node, if distance >= curCutoff the node
// is used directly; otherwise we recurse into its children with
// cutoff halved.  Leaf nodes (childBegin == 0 && childEnd == 0)
// are always evaluated.
inline __device__ float sampleWeightedPoints(
    const WeightedPointsFieldData &field, const visrtx::vec3 &P)
{
  if (!field.values || !field.indices || field.numNodes <= 0)
    return 0.f;

  if (P.x < field.domainMin.x || P.x > field.domainMax.x ||
      P.y < field.domainMin.y || P.y > field.domainMax.y ||
      P.z < field.domainMin.z || P.z > field.domainMax.z)
    return 0.f;

  const int N = field.numNodes;

  struct Entry { int idx; float cut; };
  Entry stack[32];
  int sp = 0;

  stack[sp++] = {0, field.cutoff};

  float result = 0.f;
  int iterations = 0;
  constexpr int MAX_ITERATIONS = 8192;

  while (sp > 0 && iterations < MAX_ITERATIONS) {
    ++iterations;
    Entry e = stack[--sp];
    int ni = e.idx;

    if (ni < 0 || ni >= N)
      continue;

    float curCutoff = e.cut;

    int childBegin = field.indices[ni * 2];
    int childEnd   = field.indices[ni * 2 + 1];

    if (childBegin == 0 && childEnd == 0) {
      float nv = field.values[ni * 4 + 3];
      if (nv <= 0.f) continue;
      float dx = P.x - field.values[ni * 4 + 0];
      float dy = P.y - field.values[ni * 4 + 1];
      float dz = P.z - field.values[ni * 4 + 2];
      result += nv * expf(-(dx*dx + dy*dy + dz*dz) * field.inv2SigmaSq);
      continue;
    }

    if (childBegin < 0 || childEnd < 0 || childBegin >= N || childEnd >= N)
      continue;

    float halfCutoff = curCutoff * 0.5f;

    for (int ci = childBegin; ci <= childEnd; ci++) {
      float cv = field.values[ci * 4 + 3];
      if (cv <= 0.f) continue;

      float dx = P.x - field.values[ci * 4 + 0];
      float dy = P.y - field.values[ci * 4 + 1];
      float dz = P.z - field.values[ci * 4 + 2];
      float d2 = dx*dx + dy*dy + dz*dz;

      if (d2 >= curCutoff * curCutoff) {
        result += cv * expf(-d2 * field.inv2SigmaSq);
      } else {
        if (sp < 31)
          stack[sp++] = {ci, halfCutoff};
      }
    }
  }

  return result;
}
