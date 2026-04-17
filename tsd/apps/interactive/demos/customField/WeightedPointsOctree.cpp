// Copyright 2025-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "WeightedPointsOctree.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <queue>

namespace tsd::demo {

void WeightedPointsOctree::build(const std::vector<WeightedPoint> &points,
                                 int maxPointsPerLeaf,
                                 int maxDepth)
{
  m_flatValues.clear();
  m_flatIndices.clear();
  if (points.empty())
    return;

  float bmin[3] = {points[0].x, points[0].y, points[0].z};
  float bmax[3] = {points[0].x, points[0].y, points[0].z};
  for (const auto &p : points) {
    if (p.x < bmin[0]) bmin[0] = p.x;
    if (p.y < bmin[1]) bmin[1] = p.y;
    if (p.z < bmin[2]) bmin[2] = p.z;
    if (p.x > bmax[0]) bmax[0] = p.x;
    if (p.y > bmax[1]) bmax[1] = p.y;
    if (p.z > bmax[2]) bmax[2] = p.z;
  }
  float pad = 1e-4f;
  for (int i = 0; i < 3; i++) {
    bmin[i] -= pad;
    bmax[i] += pad;
  }
  std::memcpy(m_boundsMin, bmin, sizeof(m_boundsMin));
  std::memcpy(m_boundsMax, bmax, sizeof(m_boundsMax));

  float cx = 0.5f * (bmin[0] + bmax[0]);
  float cy = 0.5f * (bmin[1] + bmax[1]);
  float cz = 0.5f * (bmin[2] + bmax[2]);
  float hx = 0.5f * (bmax[0] - bmin[0]);
  float hy = 0.5f * (bmax[1] - bmin[1]);
  float hz = 0.5f * (bmax[2] - bmin[2]);

  std::vector<size_t> allIndices(points.size());
  for (size_t i = 0; i < points.size(); i++)
    allIndices[i] = i;

  std::vector<Node> tree;
  buildRecursive(tree, points, allIndices,
                 cx, cy, cz, hx, hy, hz,
                 0, maxPointsPerLeaf, maxDepth);

  // Propagate values bottom-up: parent value = SUM of children,
  // parent position = weighted centroid
  for (int i = (int)tree.size() - 1; i >= 0; i--) {
    Node &n = tree[i];
    if (!n.isLeaf) {
      float sumVal = 0.f;
      float wx = 0.f, wy = 0.f, wz = 0.f;
      for (int c = 0; c < 8; c++) {
        int ci = n.children[c];
        if (ci >= 0 && ci < (int)tree.size() && tree[ci].value > 0.f) {
          sumVal += tree[ci].value;
          wx += tree[ci].value * tree[ci].center[0];
          wy += tree[ci].value * tree[ci].center[1];
          wz += tree[ci].value * tree[ci].center[2];
        }
      }
      n.value = sumVal;
      if (sumVal > 0.f) {
        n.center[0] = wx / sumVal;
        n.center[1] = wy / sumVal;
        n.center[2] = wz / sumVal;
      }
    }
  }

  flattenBFS(tree);
}

void WeightedPointsOctree::buildRecursive(std::vector<Node> &tree,
                                          const std::vector<WeightedPoint> &points,
                                          const std::vector<size_t> &indices,
                                          float cx, float cy, float cz,
                                          float hx, float hy, float hz,
                                          int depth,
                                          int maxPointsPerLeaf,
                                          int maxDepth)
{
  size_t myIdx = tree.size();
  tree.emplace_back();
  tree[myIdx].center[0] = cx; tree[myIdx].center[1] = cy; tree[myIdx].center[2] = cz;
  tree[myIdx].halfSize[0] = hx; tree[myIdx].halfSize[1] = hy; tree[myIdx].halfSize[2] = hz;

  if (indices.size() <= (size_t)maxPointsPerLeaf || depth >= maxDepth) {
    tree[myIdx].isLeaf = true;
    tree[myIdx].pointCount = (int)indices.size();
    tree[myIdx].pointIndices = indices;
    float sumW = 0.f;
    float sx = 0.f, sy = 0.f, sz = 0.f;
    for (size_t i : indices) {
      sumW += points[i].w;
      sx += points[i].x;
      sy += points[i].y;
      sz += points[i].z;
    }
    tree[myIdx].value = sumW;
    if (!indices.empty()) {
      float invN = 1.f / (float)indices.size();
      tree[myIdx].center[0] = sx * invN;
      tree[myIdx].center[1] = sy * invN;
      tree[myIdx].center[2] = sz * invN;
    }
    return;
  }

  tree[myIdx].isLeaf = false;
  float chx = 0.5f * hx, chy = 0.5f * hy, chz = 0.5f * hz;

  int childSlot = 0;
  for (int iz = 0; iz < 2; iz++)
    for (int iy = 0; iy < 2; iy++)
      for (int ix = 0; ix < 2; ix++) {
        float ncx = cx + (ix ? chx : -chx);
        float ncy = cy + (iy ? chy : -chy);
        float ncz = cz + (iz ? chz : -chz);

        std::vector<size_t> childIndices;
        for (size_t i : indices) {
          const auto &p = points[i];
          if (p.x >= ncx - chx && p.x <= ncx + chx &&
              p.y >= ncy - chy && p.y <= ncy + chy &&
              p.z >= ncz - chz && p.z <= ncz + chz)
            childIndices.push_back(i);
        }

        tree[myIdx].children[childSlot] = (int)tree.size();

        if (!childIndices.empty()) {
          buildRecursive(tree, points, childIndices,
                         ncx, ncy, ncz, chx, chy, chz,
                         depth + 1, maxPointsPerLeaf, maxDepth);
        } else {
          size_t emptyIdx = tree.size();
          tree.emplace_back();
          tree[emptyIdx].center[0] = ncx; tree[emptyIdx].center[1] = ncy; tree[emptyIdx].center[2] = ncz;
          tree[emptyIdx].halfSize[0] = chx; tree[emptyIdx].halfSize[1] = chy; tree[emptyIdx].halfSize[2] = chz;
          tree[emptyIdx].value = 0.f;
          tree[emptyIdx].isLeaf = true;
          tree[emptyIdx].pointCount = 0;
        }
        childSlot++;
      }
}

void WeightedPointsOctree::flattenBFS(const std::vector<Node> &tree)
{
  if (tree.empty()) return;

  int N = (int)tree.size();
  m_flatValues.resize(N * 4, 0.f);
  m_flatIndices.resize(N * 2, 0);

  std::vector<int> flatIdx(N, -1);
  std::queue<int> q;
  int nextFlat = 0;

  q.push(0);
  flatIdx[0] = nextFlat++;

  while (!q.empty()) {
    int treeIdx = q.front();
    q.pop();
    const Node &n = tree[treeIdx];
    int fi = flatIdx[treeIdx];

    m_flatValues[fi * 4 + 0] = n.center[0];
    m_flatValues[fi * 4 + 1] = n.center[1];
    m_flatValues[fi * 4 + 2] = n.center[2];
    m_flatValues[fi * 4 + 3] = n.value;

    if (n.isLeaf) {
      m_flatIndices[fi * 2 + 0] = 0;
      m_flatIndices[fi * 2 + 1] = 0;
    } else {
      int childBegin = nextFlat;
      for (int c = 0; c < 8; c++) {
        int ci = n.children[c];
        if (ci >= 0 && ci < N) {
          flatIdx[ci] = nextFlat++;
          q.push(ci);
        }
      }
      int childEnd = nextFlat - 1;
      m_flatIndices[fi * 2 + 0] = childBegin;
      m_flatIndices[fi * 2 + 1] = childEnd;
    }
  }
}

} // namespace tsd::demo
