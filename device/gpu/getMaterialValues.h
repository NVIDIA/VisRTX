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

#include "gpu/gpu_util.h"

namespace visrtx {

RT_FUNCTION bool isPopulated(const AttributePtr &ap)
{
  return ap.data != nullptr && ap.numChannels > 0;
}

RT_FUNCTION const SamplerGPUData &getSamplerData(
    const FrameGPUData &frameData, DeviceObjectIndex idx)
{
  return frameData.registry.samplers[idx];
}

template <typename T>
RT_FUNCTION const T *typedOffset(const void *mem, uint32_t offset)
{
  return ((const T *)mem) + offset;
}

RT_FUNCTION vec4 getAttributeValue(const AttributePtr &ap, uint32_t offset)
{
  if (offset == 0xFFFFFFFF)
    return vec4(0.f, 0.f, 0.f, 1.f);

  switch (ap.numChannels) {
  case 1:
    return vec4(*typedOffset<float>(ap.data, offset), 0.f, 0.f, 1.f);
  case 2:
    return vec4(*typedOffset<vec2>(ap.data, offset), 0.f, 1.f);
  case 3:
    return vec4(*typedOffset<vec3>(ap.data, offset), 1.f);
  case 4:
    return *typedOffset<vec4>(ap.data, offset);
  default:
    break;
  }

  return vec4(0.f, 0.f, 0.f, 1.f);
}

RT_FUNCTION uvec3 decodeTriangleAttributeIndices(
    const GeometryGPUData &ggd, uint32_t attributeID, const SurfaceHit &hit)
{
  if (ggd.tri.vertexAttrIndices[attributeID] != nullptr)
    return ggd.tri.vertexAttrIndices[attributeID][hit.primID];
  else if (ggd.tri.indices != nullptr)
    return ggd.tri.indices[hit.primID];
  else
    return 3 * hit.primID + uvec3(0, 1, 2);
}

RT_FUNCTION uvec4 decodeQuadAttributeIndices(
    const GeometryGPUData &ggd, uint32_t attributeID, uint32_t _primID)
{
  auto primID = (_primID / 2) * 2;
  auto i0 = ggd.quad.indices[primID + 0];
  auto i1 = ggd.quad.indices[primID + 1];
  return uvec4(i0.x, i0.y, i0.z, i1.y);
}

RT_FUNCTION uvec2 decodeCylinderAttributeIndices(
    const GeometryGPUData &ggd, uint32_t attributeID, const SurfaceHit &hit)
{
  if (ggd.tri.indices != nullptr)
    return ggd.tri.indices[hit.primID];
  else
    return 2 * hit.primID + uvec2(0, 1);
}

RT_FUNCTION vec4 readAttributeValue(uint32_t attributeID, const SurfaceHit &hit)
{
  const auto &ggd = *hit.geometry;

  // First check per-vertex attributes
  if (ggd.type == GeometryType::TRIANGLE) {
    const auto &ap = ggd.tri.vertexAttr[attributeID];
    if (isPopulated(ap)) {
      const uvec3 idx = decodeTriangleAttributeIndices(ggd, attributeID, hit);
      const vec3 b = hit.uvw;
      return b.x * getAttributeValue(ap, idx.x)
          + b.y * getAttributeValue(ap, idx.y)
          + b.z * getAttributeValue(ap, idx.z);
    }
  } else if (ggd.type == GeometryType::QUAD) {
    const auto &ap = ggd.quad.vertexAttr[attributeID];
    if (isPopulated(ap)) {
      const uvec4 idx =
          decodeQuadAttributeIndices(ggd, attributeID, hit.primID);
      const vec3 b = hit.uvw;
      auto v0 = getAttributeValue(ap, idx.x);
      auto v1 = getAttributeValue(ap, idx.y);
      auto v2 = getAttributeValue(ap, idx.z);
      auto v3 = getAttributeValue(ap, idx.w);
      auto l0 = mix(v3, v0, b.z);
      auto l1 = mix(v2, v1, b.z);
      return mix(l0, l1, b.x);
    }
  } else if (ggd.type == GeometryType::CYLINDER) {
    const auto &ap = ggd.cylinder.vertexAttr[attributeID];
    if (isPopulated(ap)) {
      const uvec2 idx = decodeCylinderAttributeIndices(ggd, attributeID, hit);
      const vec3 b = hit.uvw;
      return b.y * getAttributeValue(ap, idx.x)
          + b.z * getAttributeValue(ap, idx.y);
    }
  } else if (ggd.type == GeometryType::SPHERE) {
    const auto &ap = ggd.sphere.vertexAttr[attributeID];
    if (isPopulated(ap))
      return getAttributeValue(ap, hit.primID);
  }

  // Else fall through to per-primitive attributes
  const auto &ap = ggd.attr[attributeID];
  if (ggd.type == GeometryType::QUAD)
    return getAttributeValue(ap, hit.primID / 2);
  else if (ggd.type == GeometryType::CONE)
    return getAttributeValue(ap, hit.primID / ggd.cone.trianglesPerCone);
  else
    return getAttributeValue(ap, hit.primID);
}

template <typename T>
RT_FUNCTION T evaluateSampler(
    const FrameGPUData &fd, const DeviceObjectIndex _s, const SurfaceHit &hit)
{
  vec4 retval{0.f};
  const auto &sampler = getSamplerData(fd, _s);
  const vec4 tc = readAttributeValue(sampler.attribute, hit);
  switch (sampler.type) {
  case SamplerType::TEXTURE2D: {
    retval = make_vec4(tex2D<::float4>(sampler.image2D.texobj, tc.x, tc.y));
    break;
  }
  case SamplerType::PRIMITIVE: {
    retval = getAttributeValue(sampler.primitive.attr, hit.primID);
    break;
  }
  case SamplerType::COLOR_MAP: {
    float coord = position(tc.x, sampler.colormap.valueRange);
    retval = make_vec4(tex1D<::float4>(sampler.colormap.tfTex, coord));
    break;
  }
  default:
    break;
  }
  return bit_cast<T>(retval);
}

template <typename T>
RT_FUNCTION T getMaterialParameter(const FrameGPUData &fd,
    const MaterialParameter<T> &mp,
    const SurfaceHit &hit)
{
  switch (mp.type) {
  case MaterialParameterType::VALUE:
    return mp.value;
  case MaterialParameterType::SAMPLER:
    return evaluateSampler<T>(fd, mp.sampler, hit);
  case MaterialParameterType::ATTRIB_0:
    return bit_cast<T>(readAttributeValue(0, hit));
  case MaterialParameterType::ATTRIB_1:
    return bit_cast<T>(readAttributeValue(1, hit));
  case MaterialParameterType::ATTRIB_2:
    return bit_cast<T>(readAttributeValue(2, hit));
  case MaterialParameterType::ATTRIB_3:
    return bit_cast<T>(readAttributeValue(3, hit));
  case MaterialParameterType::ATTRIB_COLOR:
    return bit_cast<T>(readAttributeValue(4, hit));
  case MaterialParameterType::WORLD_POSITION:
    return bit_cast<T>(vec4(hit.hitpoint, 1.f));
  case MaterialParameterType::WORLD_NORMAL:
    return bit_cast<T>(vec4(hit.normal, 1.f));
  /////////////////////////////////////////////////////////////////////////////
  // NOTE: these are in world space - need to quantify best performing option
  case MaterialParameterType::OBJECT_POSITION:
    return bit_cast<T>(vec4(hit.hitpoint, 1.f));
  case MaterialParameterType::OBJECT_NORMAL:
    return bit_cast<T>(vec4(hit.normal, 1.f));
  /////////////////////////////////////////////////////////////////////////////
  default:
    break;
  }

  return T{};
}

RT_FUNCTION MaterialValues getMaterialValues(
    const FrameGPUData &fd, const MaterialGPUData &md, const SurfaceHit &hit)
{
  MaterialValues retval;
  retval.baseColor = getMaterialParameter(fd, md.baseColor, hit);
  retval.metalness = getMaterialParameter(fd, md.metalness, hit);
  retval.emissive = getMaterialParameter(fd, md.emissive, hit);
  retval.roughness = getMaterialParameter(fd, md.roughness, hit);
  retval.transmissiveness = getMaterialParameter(fd, md.transmissiveness, hit);
  retval.opacity = getMaterialParameter(fd, md.opacity, hit);
  return retval;
}

} // namespace visrtx
