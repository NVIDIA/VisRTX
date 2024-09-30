/*
 * Copyright (c) 2019-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "DebugMethod.h"
#include "gpu/shading_api.h"

namespace visrtx {

enum class RayType
{
  DEBUG
};

struct SurfaceRayData : public SurfaceHit
{
  vec3 outColor{0.f};
};

struct VolumeRayData : public VolumeHit
{
  vec3 outColor{0.f};
};

DECLARE_FRAME_DATA(frameData)

VISRTX_DEVICE void handleSurfaceHit()
{
  auto &rd = ray::rayData<SurfaceRayData>();
  ray::populateSurfaceHit(rd);

  const auto method =
      static_cast<DebugMethod>(frameData.renderer.params.debug.method);

  switch (method) {
  case DebugMethod::PRIM_ID:
    rd.outColor = makeRandomColor(ray::primID());
    break;
  case DebugMethod::GEOM_ID:
    rd.outColor = makeRandomColor(ray::objID());
    break;
  case DebugMethod::INST_ID:
    rd.outColor = makeRandomColor(ray::instID());
    break;
  case DebugMethod::RAY_UVW:
    rd.outColor = ray::uvw(rd.geometry->type);
    break;
  case DebugMethod::IS_TRIANGLE:
    rd.outColor = boolColor(rd.geometry->type == GeometryType::TRIANGLE);
    break;
  case DebugMethod::IS_VOLUME:
    rd.outColor = boolColor(false);
    break;
  case DebugMethod::BACKFACE:
    rd.outColor = boolColor(optixIsFrontFaceHit());
    break;
  case DebugMethod::NG:
    rd.outColor = rd.Ng;
    break;
  case DebugMethod::NG_ABS:
    rd.outColor = abs(rd.Ng);
    break;
  case DebugMethod::NS:
    rd.outColor = rd.Ns;
    break;
  case DebugMethod::NS_ABS:
    rd.outColor = abs(rd.Ns);
    break;
  case DebugMethod::HAS_MATERIAL:
    rd.outColor = boolColor(rd.material);
    break;
  case DebugMethod::GEOMETRY_ATTRIBUTE_0:
    rd.outColor = readAttributeValue(0, rd);
    break;
  case DebugMethod::GEOMETRY_ATTRIBUTE_1:
    rd.outColor = readAttributeValue(1, rd);
    break;
  case DebugMethod::GEOMETRY_ATTRIBUTE_2:
    rd.outColor = readAttributeValue(2, rd);
    break;
  case DebugMethod::GEOMETRY_ATTRIBUTE_3:
    rd.outColor = readAttributeValue(3, rd);
    break;
  case DebugMethod::GEOMETRY_ATTRIBUTE_COLOR:
    rd.outColor = readAttributeValue(4, rd);
    break;
  default:
    rd.outColor = vec3(1.f);
    break;
  }

  const auto c = rd.outColor * glm::abs(glm::dot(ray::direction(), rd.Ns));
  rd.outColor = glm::mix(rd.outColor, c, 0.5f);
}

VISRTX_DEVICE void handleVolumeHit()
{
  auto &rd = ray::rayData<VolumeRayData>();
  ray::populateVolumeHit(rd);

  const auto method =
      static_cast<DebugMethod>(frameData.renderer.params.debug.method);

  switch (method) {
  case DebugMethod::PRIM_ID:
    rd.outColor = makeRandomColor(ray::primID());
    break;
  case DebugMethod::GEOM_ID:
    rd.outColor = makeRandomColor(ray::objID());
    break;
  case DebugMethod::INST_ID:
    rd.outColor = makeRandomColor(ray::instID());
    break;
  case DebugMethod::IS_TRIANGLE:
    rd.outColor = boolColor(false);
    break;
  case DebugMethod::IS_VOLUME:
    rd.outColor = boolColor(true);
    break;
  case DebugMethod::BACKFACE:
    rd.outColor = boolColor(optixIsFrontFaceHit());
    break;
  default:
    rd.outColor = vec3(1.f);
    break;
  }
}

VISRTX_GLOBAL void __closesthit__()
{
  if (ray::isIntersectingSurfaces())
    handleSurfaceHit();
  else
    handleVolumeHit();
}

VISRTX_GLOBAL void __miss__()
{
  // no-op
}

VISRTX_GLOBAL void __raygen__()
{
  auto ss = createScreenSample(frameData);
  if (pixelOutOfFrame(ss.pixel, frameData.fb))
    return;
  auto ray = makePrimaryRay(ss, true /*pixel centered*/);

  auto color = vec3(getBackground(frameData.renderer, ss.screen));
  auto depth = ray.t.upper;
  auto normal = ray.dir;
  uint32_t primID = ~0u;
  uint32_t objID = ~0u;
  uint32_t instID = ~0u;

  SurfaceRayData srd{};
  intersectSurface(ss, ray, RayType::DEBUG, &srd);

  VolumeRayData vrd{};
  intersectVolume(ss, ray, RayType::DEBUG, &vrd);

  if (srd.foundHit && vrd.foundHit) {
    const bool volumeFirst = vrd.localRay.t.lower < srd.t;
    if (volumeFirst) {
      color = vrd.outColor;
      depth = vrd.localRay.t.lower;
      normal = -ray.dir;
      primID = 0;
      objID = vrd.volumeData->id;
      instID = vrd.instID;
    } else {
      color = srd.outColor;
      depth = srd.t;
      normal = srd.Ng;
      primID = srd.primID;
      objID = srd.objID;
      instID = srd.instID;
    }
  } else if (srd.foundHit) {
    color = srd.outColor;
    depth = srd.t;
    normal = srd.Ng;
    primID = srd.primID;
    objID = srd.objID;
    instID = srd.instID;
  } else if (vrd.foundHit) {
    color = vrd.outColor;
    depth = vrd.localRay.t.lower;
    normal = -ray.dir;
    primID = 0;
    objID = vrd.volumeData->id;
    instID = vrd.instID;
  }

  accumResults(frameData.fb,
      ss.pixel,
      vec4(color, 1.f),
      depth,
      color,
      normal,
      primID,
      objID,
      instID);
}

} // namespace visrtx
