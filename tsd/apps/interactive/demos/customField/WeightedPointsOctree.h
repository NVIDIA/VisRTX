// Copyright 2025-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace tsd::demo {

struct WeightedPoint
{
  float x, y, z, w;
};

// Octree over weighted points for GPU volume rendering.
//
// Build from 3D points with values. Produces two flat GPU arrays:
//   - values[]: 4 floats per node (x, y, z, value). Root at index 0.
//   - indices[]: 2 int32s per node (childBegin, childEnd). Leaf = (0,0).
//
// Leaf node: position = actual point position (centroid if >1 point),
//            value = sum of point weights.
// Internal node: position = weighted centroid of children,
//                value = sum of children values.
class WeightedPointsOctree
{
 public:
  void build(const std::vector<WeightedPoint> &points,
             int maxPointsPerLeaf = 8,
             int maxDepth = 20);

  const std::vector<float> &flatValues() const { return m_flatValues; }
  const std::vector<int32_t> &flatIndices() const { return m_flatIndices; }

  int numNodes() const { return (int)(m_flatValues.size() / 4); }
  const float *boundsMin() const { return m_boundsMin; }
  const float *boundsMax() const { return m_boundsMax; }

 private:
  struct Node
  {
    float center[3];
    float halfSize[3];
    float value = 0.f;
    int children[8] = {-1,-1,-1,-1,-1,-1,-1,-1};
    bool isLeaf = true;
    int pointCount = 0;
    std::vector<size_t> pointIndices;
  };

  void buildRecursive(std::vector<Node> &tree,
                      const std::vector<WeightedPoint> &points,
                      const std::vector<size_t> &indices,
                      float cx, float cy, float cz,
                      float hx, float hy, float hz,
                      int depth, int maxPointsPerLeaf, int maxDepth);

  void flattenBFS(const std::vector<Node> &tree);

  std::vector<float> m_flatValues;
  std::vector<int32_t> m_flatIndices;
  float m_boundsMin[3] = {0, 0, 0};
  float m_boundsMax[3] = {0, 0, 0};
};

} // namespace tsd::demo
