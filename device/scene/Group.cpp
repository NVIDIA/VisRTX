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

#include "Group.h"

namespace visrtx {

// Helper functions ///////////////////////////////////////////////////////////

template <typename T>
static std::vector<OptixBuildInput> createOBI(anari::Span<T> objs)
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
  return createOBI(anari::make_Span(objs.data(), objs.size()));
}

// Group definitions //////////////////////////////////////////////////////////

static size_t s_numGroups = 0;

size_t Group::objectCount()
{
  return s_numGroups;
}

Group::Group(DeviceGlobalState *d) : Object(ANARI_GROUP, d)
{
  s_numGroups++;
}

Group::~Group()
{
  cleanup();
  s_numGroups--;
}

bool Group::getProperty(
    const std::string_view &name, ANARIDataType type, void *ptr, uint32_t flags)
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

  return Object::getProperty(name, type, ptr, flags);
}

void Group::commit()
{
  cleanup();

  m_surfaceData = getParamObject<ObjectArray>("surface");
  m_volumeData = getParamObject<ObjectArray>("volume");
  m_lightData = getParamObject<ObjectArray>("light");

  partitionValidGeometriesByType();
  partitionValidVolumes();
  partitionValidLights();

  m_objectUpdates.lastSurfaceBVHBuilt = 0;
  m_objectUpdates.lastVolumeBVHBuilt = 0;
  m_objectUpdates.lastLightRebuild = 0;

  if (m_surfaceData)
    m_surfaceData->addCommitObserver(this);
  if (m_volumeData)
    m_volumeData->addCommitObserver(this);
  if (m_lightData)
    m_lightData->addCommitObserver(this);
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

anari::Span<const DeviceObjectIndex> Group::surfaceTriangleGPUIndices() const
{
  return anari::make_Span(
      (const DeviceObjectIndex *)m_surfaceTriangleObjectIndices.ptr(),
      m_surfacesTriangle.size());
}

anari::Span<const DeviceObjectIndex> Group::surfaceCurveGPUIndices() const
{
  return anari::make_Span(
      (const DeviceObjectIndex *)m_surfaceCurveObjectIndices.ptr(),
      m_surfacesCurve.size());
}

anari::Span<const DeviceObjectIndex> Group::surfaceUserGPUIndices() const
{
  return anari::make_Span(
      (const DeviceObjectIndex *)m_surfaceUserObjectIndices.ptr(),
      m_surfacesUser.size());
}

anari::Span<const DeviceObjectIndex> Group::volumeGPUIndices() const
{
  return anari::make_Span(
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

anari::Span<const DeviceObjectIndex> Group::lightGPUIndices() const
{
  return anari::make_Span(
      (const DeviceObjectIndex *)m_lightObjectIndices.ptr(), m_lights.size());
}

void Group::rebuildSurfaceBVHs()
{
  if (!m_surfaces) {
    m_triangleBounds = box3();
    m_curveBounds = box3();
    m_userBounds = box3();
    m_traversableTriangle = {};
    m_traversableCurve = {};
    m_traversableUser = {};
    reportMessage(
        ANARI_SEVERITY_DEBUG, "visrtx::Group skipping surface BVH build");
    return;
  }

  reportMessage(ANARI_SEVERITY_DEBUG, "visrtx::Group building triangle BVH");
  buildOptixBVH(createOBI(m_surfacesTriangle),
      m_bvhTriangle,
      m_traversableTriangle,
      m_triangleBounds,
      this);

  reportMessage(ANARI_SEVERITY_DEBUG, "visrtx::Group building curve BVH");
  buildOptixBVH(createOBI(m_surfacesCurve),
      m_bvhCurve,
      m_traversableCurve,
      m_curveBounds,
      this);

  reportMessage(ANARI_SEVERITY_DEBUG, "visrtx::Group building user BVH");
  buildOptixBVH(createOBI(m_surfacesUser),
      m_bvhUser,
      m_traversableUser,
      m_userBounds,
      this);

  buildSurfaceGPUData();

  m_objectUpdates.lastSurfaceBVHBuilt = helium::newTimeStamp();
}

void Group::rebuildVolumeBVH()
{
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

void Group::markCommitted()
{
  Object::markCommitted();
  deviceState()->objectUpdates.lastBLASChange = helium::newTimeStamp();
}

void Group::partitionValidGeometriesByType()
{
  m_surfaces.reset();
  if (!m_surfaceData)
    return;

  m_surfaces = anari::make_Span(
      (Surface **)m_surfaceData->handlesBegin(), m_surfaceData->totalSize());
  m_surfacesTriangle.clear();
  m_surfacesCurve.clear();
  m_surfacesUser.clear();
  for (auto s : m_surfaces) {
    if (!s->isValid())
      continue;
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

  auto volumes = anari::make_Span(
      (Volume **)m_volumeData->handlesBegin(), m_volumeData->totalSize());
  for (auto v : volumes) {
    if (!v->isValid())
      continue;
    m_volumes.push_back(v);
  }
}

void Group::partitionValidLights()
{
  m_lights.clear();
  if (!m_lightData)
    return;

  auto lights = anari::make_Span(
      (Light **)m_lightData->handlesBegin(), m_lightData->totalSize());
  for (auto l : lights) {
    if (!l->isValid())
      continue;
    m_lights.push_back(l);
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

void Group::cleanup()
{
  if (m_surfaceData)
    m_surfaceData->removeCommitObserver(this);
  if (m_volumeData)
    m_volumeData->removeCommitObserver(this);
  if (m_lightData)
    m_lightData->removeCommitObserver(this);
}

} // namespace visrtx

VISRTX_ANARI_TYPEFOR_DEFINITION(visrtx::Group *);
