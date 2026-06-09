/*
 * Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include "gpu_math.h"
#include "uniformGrid.h"

namespace visrtx {

// Amanatides & Woo 3D-DDA traversal.
// Construct with a ray and grid description, then loop while valid(),
// reading cellIndex / tEntry / tExit each iteration and calling next().
struct GridTraversal
{
  size_t cellIndex; // size_t: the linear index overflows int32 above ~2^31 cells
  float tEntry;
  float tExit;
  int crossedAxis; // axis of the face the ray entered the current cell through

  VISRTX_DEVICE GridTraversal(const Ray &ray, ivec3 gridDims, box3 bounds)
      : m_gridDims(gridDims), m_tEnd(ray.t.upper), m_valid(false)
  {
    if (gridDims.x <= 0 || gridDims.y <= 0 || gridDims.z <= 0)
      return;

    float t = max(ray.t.lower, 0.f);
    if (!(t < m_tEnd))
      return;

    const vec3 gridSize = (bounds.upper - bounds.lower) / vec3(gridDims);

    constexpr float inf = std::numeric_limits<float>::infinity();
    float tIn = -inf;
    float tOut = inf;
    crossedAxis = 0;
    vec3 rcpDir;
    for (int axis = 0; axis < 3; ++axis) {
      if (ray.dir[axis] != 0.f) {
        rcpDir[axis] = 1.f / ray.dir[axis];
        const float t1 = (bounds.lower[axis] - ray.org[axis]) * rcpDir[axis];
        const float t2 = (bounds.upper[axis] - ray.org[axis]) * rcpDir[axis];
        const float tNear = min(t1, t2);
        if (tNear > tIn) {
          tIn = tNear;
          crossedAxis = axis; // grid is entered through this axis' face
        }
        tOut = min(tOut, max(t1, t2));
      } else {
        rcpDir[axis] = 0.f;
        if (ray.org[axis] < bounds.lower[axis]
            || ray.org[axis] > bounds.upper[axis])
          return;
      }
    }
    if (tIn > tOut || tOut < t)
      return;
    t = max(t, tIn);

    const vec3 orgAtT = ray.org + ray.dir * t;
    m_cell = projectOnGrid(orgAtT, gridDims, bounds);

    for (int axis = 0; axis < 3; ++axis) {
      const float dir = ray.dir[axis];
      const float org = orgAtT[axis];
      const float axisCellMin =
          bounds.lower[axis] + float(m_cell[axis]) * gridSize[axis];
      const float axisCellMax = axisCellMin + gridSize[axis];

      if (dir > 0.f) {
        m_step[axis] = 1;
        m_tDelta[axis] = gridSize[axis] * rcpDir[axis];
        m_tMax[axis] = t + (axisCellMax - org) * rcpDir[axis];
      } else if (dir < 0.f) {
        m_step[axis] = -1;
        m_tDelta[axis] = gridSize[axis] * -rcpDir[axis];
        m_tMax[axis] = t + (axisCellMin - org) * rcpDir[axis];
      } else {
        m_step[axis] = 0;
        m_tDelta[axis] = inf;
        m_tMax[axis] = inf;
      }
    }

    m_t = t;
    computeCurrentVoxel();
  }

  VISRTX_DEVICE bool valid() const
  {
    return m_valid;
  }

  VISRTX_DEVICE void next()
  {
    m_t = tExit;
    if (!(m_t < m_tEnd)) {
      m_valid = false;
      return;
    }

    step(min(m_tMax.x, min(m_tMax.y, m_tMax.z)));
    computeCurrentVoxel();
  }

 private:
  VISRTX_DEVICE void step(float tNext)
  {
    // The axis with the smallest tMax is the boundary plane being crossed —
    // i.e. the face through which the ray enters the next cell.
    if (m_tMax.x <= m_tMax.y && m_tMax.x <= m_tMax.z)
      crossedAxis = 0;
    else if (m_tMax.y <= m_tMax.z)
      crossedAxis = 1;
    else
      crossedAxis = 2;

    if (m_tMax.x <= tNext) {
      m_cell.x += m_step.x;
      m_tMax.x += m_tDelta.x;
    }
    if (m_tMax.y <= tNext) {
      m_cell.y += m_step.y;
      m_tMax.y += m_tDelta.y;
    }
    if (m_tMax.z <= tNext) {
      m_cell.z += m_step.z;
      m_tMax.z += m_tDelta.z;
    }
  }

  VISRTX_DEVICE void computeCurrentVoxel()
  {
    constexpr float inf = std::numeric_limits<float>::infinity();
    while (unsigned(m_cell.x) < unsigned(m_gridDims.x)
        && unsigned(m_cell.y) < unsigned(m_gridDims.y)
        && unsigned(m_cell.z) < unsigned(m_gridDims.z) && m_t < m_tEnd) {
      const float tNext = min(m_tMax.x, min(m_tMax.y, m_tMax.z));
      if (!(tNext < inf))
        break;

      const float t1 = min(tNext, m_tEnd);
      if (t1 > m_t) {
        cellIndex = size_t(m_cell.x)
            + size_t(m_gridDims.x)
                * (size_t(m_cell.y) + size_t(m_gridDims.y) * size_t(m_cell.z));
        tEntry = m_t;
        tExit = t1;
        m_valid = true;
        return;
      }

      m_t = t1;
      step(tNext);
    }
    m_valid = false;
  }

  ivec3 m_gridDims;
  float m_tEnd;
  ivec3 m_cell;
  vec3 m_tMax;
  vec3 m_tDelta;
  ivec3 m_step;
  float m_t;
  bool m_valid;
};

} // namespace visrtx