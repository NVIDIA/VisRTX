/*
 * Copyright (c) 2019-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "World.h"
// ptx
#include "Intersectors_ptx.h"

namespace visrtx {

ptx_ptr intersection_ptx()
{
  return Intersectors_ptx;
}

// Helper functions ///////////////////////////////////////////////////////////

static std::vector<OptixBuildInput> createOBI(
    HostDeviceArray<OptixInstance> &optixInstances)
{
  auto optixInstancesDevice = optixInstances.deviceSpan();
  auto numInstances = optixInstancesDevice.size();

  if (numInstances == 0)
    return {};

  OptixBuildInput buildInput{};

  buildInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
  buildInput.instanceArray.instances =
      numInstances > 0 ? (CUdeviceptr)optixInstancesDevice.data() : 0;
  buildInput.instanceArray.numInstances = numInstances;

  return {buildInput};
}

// World definitions //////////////////////////////////////////////////////////

static size_t s_numWorlds = 0;

size_t World::objectCount()
{
  return s_numWorlds;
}

World::World(DeviceGlobalState *d) : Object(ANARI_WORLD, d)
{
  s_numWorlds++;

  m_zeroGroup = new Group(d);
  m_zeroInstance = new Instance(d);
  m_zeroInstance->setParamDirect("group", m_zeroGroup.ptr);

  // never any public ref to these objects
  m_zeroGroup->refDec(helium::RefType::PUBLIC);
  m_zeroInstance->refDec(helium::RefType::PUBLIC);
}

World::~World()
{
  cleanup();
  s_numWorlds--;
}

bool World::getProperty(
    const std::string_view &name, ANARIDataType type, void *ptr, uint32_t flags)
{
  if (name == "bounds" && type == ANARI_FLOAT32_BOX3) {
    if (flags & ANARI_WAIT) {
      deviceState()->commitBuffer.flush();
      rebuildBVHs();
    }
    auto bounds = m_surfaceBounds;
    bounds.extend(m_volumeBounds);
    std::memcpy(ptr, &bounds, sizeof(bounds));
    return true;
  }

  return Object::getProperty(name, type, ptr, flags);
}

void World::commit()
{
  cleanup();

  m_zeroSurfaceData = getParamObject<ObjectArray>("surface");
  m_zeroVolumeData = getParamObject<ObjectArray>("volume");
  m_zeroLightData = getParamObject<ObjectArray>("light");

  m_addZeroInstance = m_zeroSurfaceData || m_zeroVolumeData || m_zeroLightData;
  if (m_addZeroInstance)
    reportMessage(ANARI_SEVERITY_DEBUG, "visrtx::World will add zero instance");

  if (m_zeroSurfaceData) {
    reportMessage(
        ANARI_SEVERITY_DEBUG, "visrtx::World found surfaces in zero instance");
    m_zeroGroup->setParamDirect("surface", getParamDirect("surface"));
  } else
    m_zeroGroup->removeParam("surface");

  if (m_zeroVolumeData) {
    reportMessage(
        ANARI_SEVERITY_DEBUG, "visrtx::World found volumes in zero instance");
    m_zeroGroup->setParamDirect("volume", getParamDirect("volume"));
  } else
    m_zeroGroup->removeParam("volume");

  if (m_zeroLightData) {
    reportMessage(
        ANARI_SEVERITY_DEBUG, "visrtx::World found lights in zero instance");
    m_zeroGroup->setParamDirect("light", getParamDirect("light"));
  } else
    m_zeroGroup->removeParam("light");

  m_zeroInstance->setParam("id", getParam<uint32_t>("id", ~0u));

  m_zeroGroup->commit();
  m_zeroInstance->commit();

  m_instanceData = getParamObject<ObjectArray>("instance");

  m_instances.reset();

  if (m_instanceData) {
    m_instanceData->removeAppendedHandles();
    if (m_addZeroInstance)
      m_instanceData->appendHandle(m_zeroInstance.ptr);
    m_instances = make_Span((Instance **)m_instanceData->handlesBegin(),
        m_instanceData->totalSize());
  } else if (m_addZeroInstance)
    m_instances = make_Span(&m_zeroInstance.ptr, 1);

  m_objectUpdates.lastTLASBuild = 0;
  m_objectUpdates.lastBLASCheck = 0;

  if (m_instanceData)
    m_instanceData->addCommitObserver(this);
  if (m_zeroSurfaceData)
    m_zeroSurfaceData->addCommitObserver(this);
  if (m_zeroVolumeData)
    m_zeroVolumeData->addCommitObserver(this);
  if (m_zeroLightData)
    m_zeroLightData->addCommitObserver(this);
}

OptixTraversableHandle World::optixTraversableHandleSurfaces() const
{
  return m_traversableSurfaces;
}

OptixTraversableHandle World::optixTraversableHandleVolumes() const
{
  return m_traversableVolumes;
}

Span<const InstanceSurfaceGPUData> World::instanceSurfaceGPUData() const
{
  return m_instanceSurfaceGPUData.deviceSpan();
}

Span<const InstanceVolumeGPUData> World::instanceVolumeGPUData() const
{
  return m_instanceVolumeGPUData.deviceSpan();
}

Span<const InstanceLightGPUData> World::instanceLightGPUData() const
{
  return m_instanceLightGPUData.deviceSpan();
}

void World::rebuildBVHs()
{
  const auto &state = *deviceState();

  if (state.objectUpdates.lastBLASChange >= m_objectUpdates.lastBLASCheck) {
    m_objectUpdates.lastTLASBuild = 0; // BLAS changed, so need to build TLAS
    rebuildBLASs();
  }

  if (state.objectUpdates.lastTLASChange < m_objectUpdates.lastTLASBuild)
    return;

  m_surfaceBounds = box3();
  m_volumeBounds = box3();
  m_traversableSurfaces = {};
  m_traversableVolumes = {};

  populateOptixInstances();
  reportMessage(ANARI_SEVERITY_DEBUG,
      "visrtx::World building surface BVH over %zu instances",
      m_optixSurfaceInstances.size());
  buildOptixBVH(createOBI(m_optixSurfaceInstances),
      m_bvhSurfaces,
      m_traversableSurfaces,
      m_surfaceBounds,
      this);
  reportMessage(
      ANARI_SEVERITY_DEBUG, "visrtx::World building surface gpu data");
  buildInstanceSurfaceGPUData();

  reportMessage(ANARI_SEVERITY_DEBUG,
      "visrtx::World building volume BVH over %zu instances",
      m_optixVolumeInstances.size());
  buildOptixBVH(createOBI(m_optixVolumeInstances),
      m_bvhVolumes,
      m_traversableVolumes,
      m_volumeBounds,
      this);
  reportMessage(ANARI_SEVERITY_DEBUG, "visrtx::World building volume gpu data");
  buildInstanceVolumeGPUData();

  buildInstanceLightGPUData();

  m_objectUpdates.lastTLASBuild = helium::newTimeStamp();
}

void World::populateOptixInstances()
{
  m_numTriangleInstances = 0;
  m_numCurveInstances = 0;
  m_numUserInstances = 0;
  m_numVolumeInstances = 0;
  m_numLightInstances = 0;

  std::for_each(m_instances.begin(), m_instances.end(), [&](auto *inst) {
    auto *group = inst->group();
    if (group->containsTriangleGeometry())
      m_numTriangleInstances++;
    if (group->containsCurveGeometry())
      m_numCurveInstances++;
    if (group->containsUserGeometry())
      m_numUserInstances++;
    if (group->containsVolumes())
      m_numVolumeInstances++;
    if (group->containsLights())
      m_numLightInstances++;
  });

  m_optixSurfaceInstances.resize(
      m_numTriangleInstances + m_numCurveInstances + m_numUserInstances);
  m_optixVolumeInstances.resize(m_numVolumeInstances);

  auto prepInstance =
      [](auto &i, int instID, auto handle, int sbtOffset) -> OptixInstance {
    OptixInstance inst{};

    mat3x4 xfm = glm::transpose(i->xfm());
    std::memcpy(inst.transform, &xfm, sizeof(xfm));

    auto *group = i->group();
    inst.traversableHandle = handle;
    inst.flags = OPTIX_INSTANCE_FLAG_NONE;
    inst.instanceId = instID;
    inst.sbtOffset = sbtOffset;
    inst.visibilityMask = 1;

    return inst;
  };

  int instID = 0;
  int instVolID = 0;
  std::for_each(m_instances.begin(), m_instances.end(), [&](auto *inst) {
    auto *group = inst->group();
    auto *osi = m_optixSurfaceInstances.dataHost();
    auto *ovi = m_optixVolumeInstances.dataHost();
    if (group->containsTriangleGeometry()) {
      osi[instID] = prepInstance(
          inst, instID, group->optixTraversableTriangle(), SBT_TRIANGLE_OFFSET);
      instID++;
    }
    if (group->containsCurveGeometry()) {
      osi[instID] = prepInstance(
          inst, instID, group->optixTraversableCurve(), SBT_CURVE_OFFSET);
      instID++;
    }
    if (group->containsUserGeometry()) {
      osi[instID] = prepInstance(
          inst, instID, group->optixTraversableUser(), SBT_CUSTOM_OFFSET);
      instID++;
    }
    if (group->containsVolumes()) {
      ovi[instVolID] = prepInstance(
          inst, instVolID, group->optixTraversableVolume(), SBT_CUSTOM_OFFSET);
      instVolID++;
    }
  });

  m_optixSurfaceInstances.upload();
  m_optixVolumeInstances.upload();
}

void World::rebuildBLASs()
{
  reportMessage(ANARI_SEVERITY_DEBUG, "visrtx::World rebuilding BLASs");

  std::for_each(m_instances.begin(), m_instances.end(), [&](auto *inst) {
    auto *group = inst->group();
    group->rebuildSurfaceBVHs();
    group->rebuildVolumeBVH();
    group->rebuildLights();
  });

  m_objectUpdates.lastBLASCheck = helium::newTimeStamp();
}

void World::buildInstanceSurfaceGPUData()
{
  m_instanceSurfaceGPUData.resize(
      m_numTriangleInstances + m_numCurveInstances + m_numUserInstances);

  int instID = 0;
  std::for_each(m_instances.begin(), m_instances.end(), [&](auto *inst) {
    auto *group = inst->group();
    auto *sd = m_instanceSurfaceGPUData.dataHost();
    auto id = inst->userID();
    if (group->containsTriangleGeometry())
      sd[instID++] = {group->surfaceTriangleGPUIndices().data(), id};
    if (group->containsCurveGeometry())
      sd[instID++] = {group->surfaceCurveGPUIndices().data(), id};
    if (group->containsUserGeometry())
      sd[instID++] = {group->surfaceUserGPUIndices().data(), id};
  });

  m_instanceSurfaceGPUData.upload();
}

void World::buildInstanceVolumeGPUData()
{
  m_instanceVolumeGPUData.resize(m_numVolumeInstances);

  int instID = 0;
  std::for_each(m_instances.begin(), m_instances.end(), [&](auto *inst) {
    auto *group = inst->group();
    auto *vd = m_instanceVolumeGPUData.dataHost();
    auto id = inst->userID();
    if (group->containsVolumes())
      vd[instID++] = {group->volumeGPUIndices().data(), id};
  });

  m_instanceVolumeGPUData.upload();
}

void World::buildInstanceLightGPUData()
{
  m_instanceLightGPUData.resize(m_numLightInstances);

  int instID = 0;
  std::for_each(m_instances.begin(), m_instances.end(), [&](auto *inst) {
    auto *group = inst->group();
    auto *li = m_instanceLightGPUData.dataHost();
    if (group->containsLights()) {
      group->rebuildLights();
      const auto lgi = group->lightGPUIndices();
      if (!inst->xfmIsIdentity() && lgi.size() != 0) {
        inst->reportMessage(
            ANARI_SEVERITY_WARNING, "light transformations not implemented");
      }
      li[instID++] = {lgi.data(), lgi.size()};
    }
  });

  m_instanceLightGPUData.upload();
}

void World::cleanup()
{
  if (m_instanceData)
    m_instanceData->removeCommitObserver(this);
  if (m_zeroSurfaceData)
    m_zeroSurfaceData->removeCommitObserver(this);
  if (m_zeroVolumeData)
    m_zeroVolumeData->removeCommitObserver(this);
  if (m_zeroLightData)
    m_zeroLightData->removeCommitObserver(this);
}

} // namespace visrtx

VISRTX_ANARI_TYPEFOR_DEFINITION(visrtx::World *);
