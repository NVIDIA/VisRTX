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

#include "Group.h"

namespace visrtx {

// Helper functions ///////////////////////////////////////////////////////////

template <typename T>
static std::vector<OptixBuildInput> createOBI(Span<T> objs)
{
  std::vector<OptixBuildInput> buildInput(objs.size());
  std::transform(objs.begin(), objs.end(), buildInput.begin(), [](auto o) {
    return o->buildInput();
  });
  return buildInput;
}

template <typename T>
static std::vector<OptixBuildInput> createOBI(const std::vector<T *> &objs)
{
  return createOBI(make_Span(objs.data(), objs.size()));
}

// Group definitions //////////////////////////////////////////////////////////

Group::Group(DeviceGlobalState *d)
    : Object(ANARI_GROUP, d),
      m_surfaceData(this),
      m_volumeData(this),
      m_lightData(this)
{}

Group::~Group() = default;

bool Group::getProperty(const std::string_view &name,
    ANARIDataType type,
    void *ptr,
    uint64_t size,
    uint32_t flags)
{
  if (name == "bounds" && type == ANARI_FLOAT32_BOX3) {
    if (flags & ANARI_WAIT) {
      deviceState()->commitBuffer.flush();
      rebuildSurfaceBVHs();
      rebuildVolumeBVH();
    }
    auto bounds = m_triangleBounds;
    bounds.extend(m_curveBounds);
    bounds.extend(m_userBounds);
    bounds.extend(m_volumeBounds);
    std::memcpy(ptr, &bounds, sizeof(bounds));
    return true;
  }

  return Object::getProperty(name, type, ptr, size, flags);
}

void Group::commitParameters()
{
  m_surfaceData = getParamObject<ObjectArray>("surface");
  m_volumeData = getParamObject<ObjectArray>("volume");
  m_lightData = getParamObject<ObjectArray>("light");
}

void Group::finalize()
{
  m_objectUpdates.lastSurfaceBVHBuilt = 0;
  m_objectUpdates.lastVolumeBVHBuilt = 0;
  m_objectUpdates.lastLightRebuild = 0;
}

void Group::markFinalized()
{
  Object::markFinalized();
  deviceState()->objectUpdates.lastBLASChange = helium::newTimeStamp();
}

OptixTraversableHandle Group::optixTraversableTriangle() const
{
  return m_traversableTriangle;
}

OptixTraversableHandle Group::optixTraversableCurve() const
{
  return m_traversableCurve;
}

OptixTraversableHandle Group::optixTraversableUser() const
{
  return m_traversableUser;
}

OptixTraversableHandle Group::optixTraversableVolume() const
{
  return m_traversableVolume;
}

Span<DeviceObjectIndex> Group::surfaceTriangleGPUIndices() const
{
  return make_Span(
      (const DeviceObjectIndex *)m_surfaceTriangleObjectIndices.ptr(),
      m_surfacesTriangle.size());
}

Span<DeviceObjectIndex> Group::surfaceCurveGPUIndices() const
{
  return make_Span((const DeviceObjectIndex *)m_surfaceCurveObjectIndices.ptr(),
      m_surfacesCurve.size());
}

Span<DeviceObjectIndex> Group::surfaceUserGPUIndices() const
{
  return make_Span((const DeviceObjectIndex *)m_surfaceUserObjectIndices.ptr(),
      m_surfacesUser.size());
}

Span<DeviceObjectIndex> Group::volumeGPUIndices() const
{
  return make_Span(
      (const DeviceObjectIndex *)m_volumeObjectIndices.ptr(), m_volumes.size());
}

bool Group::containsTriangleGeometry() const
{
  return !m_surfacesTriangle.empty();
}

bool Group::containsCurveGeometry() const
{
  return !m_surfacesCurve.empty();
}

bool Group::containsUserGeometry() const
{
  return !m_surfacesUser.empty();
}

bool Group::containsVolumes() const
{
  return m_volumes.size() > 0;
}

bool Group::containsLights() const
{
  return m_lights.size() > 0;
}

Span<DeviceObjectIndex> Group::lightGPUIndices() const
{
  return make_Span(
      (const DeviceObjectIndex *)m_lightObjectIndices.ptr(), m_lights.size());
}

DeviceObjectIndex Group::firstHDRI() const
{
  return m_firstHDRI;
}

const std::vector<Light *> &Group::lights() const
{
  return m_lights;
}

void Group::rebuildSurfaceBVHs()
{
  const auto &state = *deviceState();
  if (state.objectUpdates.lastBLASChange < m_objectUpdates.lastSurfaceBVHBuilt)
    return;

  partitionValidGeometriesByType();

  m_triangleBounds = box3();
  m_curveBounds = box3();
  m_userBounds = box3();
  m_traversableTriangle = {};
  m_traversableCurve = {};
  m_traversableUser = {};

  if (!m_surfacesTriangle.empty()) {
    reportMessage(ANARI_SEVERITY_DEBUG, "visrtx::Group building triangle BVH");
    buildOptixBVH(createOBI(m_surfacesTriangle),
        m_bvhTriangle,
        m_traversableTriangle,
        m_triangleBounds,
        this);
  } else {
    reportMessage(
        ANARI_SEVERITY_DEBUG, "visrtx::Group skipping triangle BVH build");
  }

  if (!m_surfacesCurve.empty()) {
    reportMessage(ANARI_SEVERITY_DEBUG, "visrtx::Group building curve BVH");
    buildOptixBVH(createOBI(m_surfacesCurve),
        m_bvhCurve,
        m_traversableCurve,
        m_curveBounds,
        this);
  } else {
    reportMessage(
        ANARI_SEVERITY_DEBUG, "visrtx::Group skipping curve BVH build");
  }

  if (!m_surfacesUser.empty()) {
    reportMessage(ANARI_SEVERITY_DEBUG, "visrtx::Group building user BVH");
    buildOptixBVH(createOBI(m_surfacesUser),
        m_bvhUser,
        m_traversableUser,
        m_userBounds,
        this);
  } else {
    reportMessage(
        ANARI_SEVERITY_DEBUG, "visrtx::Group skipping user BVH build");
  }

  buildSurfaceGPUData();

  m_objectUpdates.lastSurfaceBVHBuilt = helium::newTimeStamp();
}

void Group::rebuildVolumeBVH()
{
  partitionValidVolumes();
  if (m_volumes.empty()) {
    m_volumeBounds = box3();
    m_traversableVolume = {};
    reportMessage(
        ANARI_SEVERITY_DEBUG, "visrtx::Group skipping volume BVH build");
    return;
  }

  reportMessage(ANARI_SEVERITY_DEBUG, "visrtx::Group building volume BVH");
  buildOptixBVH(createOBI(m_volumes),
      m_bvhVolume,
      m_traversableVolume,
      m_volumeBounds,
      this);

  buildVolumeGPUData();

  m_objectUpdates.lastVolumeBVHBuilt = helium::newTimeStamp();
}

void Group::rebuildLights()
{
  partitionValidLights();
  buildLightGPUData();
  m_objectUpdates.lastLightRebuild = helium::newTimeStamp();
}

void Group::partitionValidGeometriesByType()
{
  m_surfacesTriangle.clear();
  m_surfacesCurve.clear();
  m_surfacesUser.clear();

  if (!m_surfaceData)
    return;

  auto surfaces = make_Span(
      (Surface **)m_surfaceData->handlesBegin(), m_surfaceData->totalSize());

  for (auto s : surfaces) {
    if (!(s && s->isValid() && s->isVisible())) {
      if (s && !s->isValid()) {
        reportMessage(ANARI_SEVERITY_WARNING,
            "visrtx::Group encountered invalid surface %p",
            s);
      }
      continue;
    }
    auto g = s->geometry();
    if (g->optixGeometryType() == OPTIX_BUILD_INPUT_TYPE_TRIANGLES)
      m_surfacesTriangle.push_back(s);
    else if (g->optixGeometryType() == OPTIX_BUILD_INPUT_TYPE_CURVES)
      m_surfacesCurve.push_back(s);
    else
      m_surfacesUser.push_back(s);
  }
}

void Group::partitionValidVolumes()
{
  m_volumes.clear();
  if (!m_volumeData)
    return;

  auto volumes = make_Span(
      (Volume **)m_volumeData->handlesBegin(), m_volumeData->totalSize());
  for (auto v : volumes) {
    if (!(v && v->isValid() && v->isVisible())) {
      if (v && !v->isValid()) {
        reportMessage(ANARI_SEVERITY_WARNING,
            "visrtx::Group encountered invalid volume %p",
            v);
      }
      continue;
    }
    m_volumes.push_back(v);
  }
}

void Group::partitionValidLights()
{
  m_lights.clear();
  m_firstHDRI = -1;
  if (!m_lightData)
    return;

  auto lights = make_Span(
      (Light **)m_lightData->handlesBegin(), m_lightData->totalSize());
  for (auto l : lights) {
    if (!l->isValid()) {
      reportMessage(ANARI_SEVERITY_WARNING,
          "visrtx::Group encountered invalid light %p",
          l);
      continue;
    }
    m_lights.push_back(l);
    if (m_firstHDRI == -1 && l->isHDRI())
      m_firstHDRI = l->index();
  }
}

void Group::buildSurfaceGPUData()
{
  auto populateData = [](auto g) { return g->gpuData(); };

  if (!m_surfacesTriangle.empty()) {
    std::vector<DeviceObjectIndex> tmp(m_surfacesTriangle.size());
    std::transform(m_surfacesTriangle.begin(),
        m_surfacesTriangle.end(),
        tmp.begin(),
        [](auto v) { return v->index(); });
    m_surfaceTriangleObjectIndices.upload(tmp);
  } else
    m_surfaceTriangleObjectIndices.reset();

  if (!m_surfacesCurve.empty()) {
    std::vector<DeviceObjectIndex> tmp(m_surfacesCurve.size());
    std::transform(m_surfacesCurve.begin(),
        m_surfacesCurve.end(),
        tmp.begin(),
        [](auto v) { return v->index(); });
    m_surfaceCurveObjectIndices.upload(tmp);
  } else
    m_surfaceCurveObjectIndices.reset();

  if (!m_surfacesUser.empty()) {
    std::vector<DeviceObjectIndex> tmp(m_surfacesUser.size());
    std::transform(
        m_surfacesUser.begin(), m_surfacesUser.end(), tmp.begin(), [](auto v) {
          return v->index();
        });
    m_surfaceUserObjectIndices.upload(tmp);
  } else
    m_surfaceUserObjectIndices.reset();
}

void Group::buildVolumeGPUData()
{
  std::vector<DeviceObjectIndex> tmp(m_volumes.size());
  std::transform(m_volumes.begin(), m_volumes.end(), tmp.begin(), [](auto v) {
    return v->index();
  });
  m_volumeObjectIndices.upload(tmp);
}

void Group::buildLightGPUData()
{
  if (m_lights.empty())
    return;
  std::vector<DeviceObjectIndex> tmp(m_lights.size());
  std::transform(m_lights.begin(), m_lights.end(), tmp.begin(), [](auto l) {
    return l->index();
  });
  m_lightObjectIndices.upload(tmp);
}

} // namespace visrtx

VISRTX_ANARI_TYPEFOR_DEFINITION(visrtx::Group *);
