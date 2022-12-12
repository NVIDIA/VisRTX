/*
 * Copyright (c) 2019-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "gpu_debug.h"
#include "gpu_math.h"
#include "gpu_objects.h"
#include "uniformGrid.h"

namespace visrtx {

  typedef ivec3 GridIterationState;
  
  template <typename Func>
  RT_FUNCTION
  void dda3(Ray ray, ivec3 gridDims, box3 modelBounds, Func func)
  {
    const vec3 rcp_dir(ray.dir.x != 0.f ? 1.f/ray.dir.x : 0.f,
                       ray.dir.y != 0.f ? 1.f/ray.dir.y : 0.f,
                       ray.dir.z != 0.f ? 1.f/ray.dir.z : 0.f);


    const vec3 lo = (modelBounds.lower - ray.org) * rcp_dir;
    const vec3 hi = (modelBounds.upper - ray.org) * rcp_dir;

    const vec3 tnear = min(lo,hi);
    const vec3 tfar  = max(lo,hi);

    ivec3 cellID = projectOnGrid(ray.org,gridDims,modelBounds);

    // Distance in world space to get from cell to cell
    const vec3 dist((tfar-tnear)/vec3(gridDims));

    // Cell increment
    const ivec3 step = {
      ray.dir.x > 0.f ? 1 : -1,
      ray.dir.y > 0.f ? 1 : -1,
      ray.dir.z > 0.f ? 1 : -1
    };

    // Stop when we reach grid borders
    const ivec3 stop = {
      ray.dir.x > 0.f ? gridDims.x : -1,
      ray.dir.y > 0.f ? gridDims.y : -1,
      ray.dir.z > 0.f ? gridDims.z : -1
    };

    // Increment in world space
    vec3 tnext = {
      ray.dir.x > 0.f ? tnear.x + float(cellID.x+1) * dist.x
                            : tnear.x + float(gridDims.x-cellID.x) * dist.x,
      ray.dir.y > 0.f ? tnear.y + float(cellID.y+1) * dist.y
                            : tnear.y + float(gridDims.y-cellID.y) * dist.y,
      ray.dir.z > 0.f ? tnear.z + float(cellID.z+1) * dist.z
                            : tnear.z + float(gridDims.z-cellID.z) * dist.z
    };


    float t0 = max(ray.t.lower,0.f);

    while (1) { // loop over grid cells
      const float t1 = min(compMin(tnext),ray.t.upper);
      // if (debug()) printf("DDA cell: (%i,%i,%i), ival: [%f,%f]\n",
      //                     cellID.x,cellID.y,cellID.z,t0,t1);
      if (!func(linearIndex(cellID,gridDims),t0,t1))
        return;

#if 0
      int axis = arg_min(tnext);
      tnext[axis] += dist[axis];
      cellID[axis] += step[axis];
      if (cellID[axis]==stop[axis]) {
        break;
      }
#else
      const float t_closest = compMin(tnext);
      if (tnext.x == t_closest) {
        tnext.x += dist.x;
        cellID.x += step.x;
        if (cellID.x==stop.x) {
          break;
        }
      }
      if (tnext.y == t_closest) {
        tnext.y += dist.y;
        cellID.y += step.y;
        if (cellID.y==stop.y) {
          break;
        }
      }
      if (tnext.z == t_closest) {
        tnext.z += dist.z;
        cellID.z += step.z;
        if (cellID.z==stop.z) {
          break;
        }
      }
#endif
      t0 = t1;
    }
  }
} // namespace vistrx
