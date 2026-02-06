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

#include "World.h"

#include <helium/utility/IntrusivePtr.h>
#include <helium/utility/TimeStamp.h>
// ptx
#include "Intersectors_ptx.h"
#include "array/ObjectArray.h"
#include "gpu/gpu_objects.h"
#include "optix_visrtx.h"
#include "utility/AnariTypeHelpers.h"

#ifdef USE_MDL
#include "geometry/ComputeTangent.h"
#include "material/MDL.h"
#endif // defined(USE_MDL)

namespace visrtx {

ptx_blob intersection_ptx()
{
  return {Intersectors_ptx, sizeof(Intersectors_ptx)};
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

World::World(DeviceGlobalState *d)
    : Object(ANARI_WORLD, d),
      m_zeroSurfaceData(this),
      m_zeroVolumeData(this),
      m_zeroLightData(this),
      m_instanceData(this)
{
  m_zeroGroup = new Group(d);
  m_zeroInstance = new Instance(d);

  m_zeroInstance->setParamDirect("group", m_zeroGroup.ptr);
  m_zeroInstance->commitParameters();
  m_zeroInstance->finalize();

  // never any public ref to these objects
  m_zeroGroup->refDec(helium::RefType::PUBLIC);
  m_zeroInstance->refDec(helium::RefType::PUBLIC);
}

World::~World() = default;

bool World::getProperty(const std::string_view &name,
    ANARIDataType type,
    void *ptr,
    uint64_t size,
    uint32_t flags)
{
  if (name == "bounds" && type == ANARI_FLOAT32_BOX3) {
    if (flags & ANARI_WAIT) {
      deviceState()->commitBuffer.flush();
      rebuildWorld();
    }
    auto bounds = m_surfaceBounds;
    bounds.extend(m_volumeBounds);
    std::memcpy(ptr, &bounds, sizeof(bounds));
    return true;
  }

  return Object::getProperty(name, type, ptr, size, flags);
}

void World::commitParameters()
{
  m_zeroSurfaceData = getParamObject<ObjectArray>("surface");
  m_zeroVolumeData = getParamObject<ObjectArray>("volume");
  m_zeroLightData = getParamObject<ObjectArray>("light");
  m_instanceData = getParamObject<ObjectArray>("instance");
}

void World::finalize()
{
  const bool addZeroInstance =
      m_zeroSurfaceData || m_zeroVolumeData || m_zeroLightData;
  if (addZeroInstance)
    reportMessage(ANARI_SEVERITY_DEBUG, "visrtx::World will add zero instance");

  if (m_zeroSurfaceData) {
    reportMessage(ANARI_SEVERITY_DEBUG,
        "visrtx::World found %zu surfaces in zero instance",
        m_zeroSurfaceData->totalSize());
    m_zeroGroup->setParamDirect("surface", getParamDirect("surface"));
  } else
    m_zeroGroup->removeParam("surface");

  if (m_zeroVolumeData) {
    reportMessage(ANARI_SEVERITY_DEBUG,
        "visrtx::World found %zu volumes in zero instance",
        m_zeroVolumeData->totalSize());
    m_zeroGroup->setParamDirect("volume", getParamDirect("volume"));
  } else
    m_zeroGroup->removeParam("volume");

  if (m_zeroLightData) {
    reportMessage(ANARI_SEVERITY_DEBUG,
        "visrtx::World found %zu lights in zero instance",
        m_zeroLightData->totalSize());
    m_zeroGroup->setParamDirect("light", getParamDirect("light"));
  } else
    m_zeroGroup->removeParam("light");

  m_zeroInstance->setParam("id", getParam<uint32_t>("id", ~0u));

  m_zeroGroup->commitParameters();
  m_zeroGroup->finalize();

  m_instances.clear();

  if (m_instanceData) {
    std::for_each(m_instanceData->handlesBegin(),
        m_instanceData->handlesEnd(),
        [&](auto *o) {
          if (o && o->isValid())
            m_instances.push_back((Instance *)o);
        });
  }

  if (addZeroInstance)
    m_instances.push_back(m_zeroInstance.ptr);

  m_objectUpdates.lastTLASBuild = 0;
  m_objectUpdates.lastBLASCheck = 0;
}

WorldGPUData World::gpuData() const
{
  WorldGPUData retval;

  retval.surfaceInstances = m_instanceSurfaceGPUData.dataDevice();
  retval.numSurfaceInstances = m_instanceSurfaceGPUData.size();
  retval.surfacesTraversable = m_traversableSurfaces;

  retval.volumeInstances = m_instanceVolumeGPUData.dataDevice();
  retval.numVolumeInstances = m_instanceVolumeGPUData.size();
  retval.volumesTraversable = m_traversableVolumes;

  retval.lightInstances = m_instanceLightGPUData.dataDevice();
  retval.numLightInstances = m_instanceLightGPUData.size();

  retval.hdri = m_hdri;

  return retval;
}

void World::rebuildWorld()
{
  const auto &state = *deviceState();

  if (state.objectUpdates.lastBLASChange >= m_objectUpdates.lastBLASCheck) {
    m_objectUpdates.lastTLASBuild = 0; // BLAS changed, so need to build TLAS
    rebuildBLASs();
  }

  if (m_objectUpdates.lastTLASBuild <= state.objectUpdates.lastTLASChange) {
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
    reportMessage(
        ANARI_SEVERITY_DEBUG, "visrtx::World building volume gpu data");
    buildInstanceVolumeGPUData();

    buildInstanceLightGPUData();

    reportMessage(ANARI_SEVERITY_DEBUG,
        "visrtx::World finished building world over %zu instances",
        m_instances.size());

    m_objectUpdates.lastTLASBuild = helium::newTimeStamp();
  }
}

void World::populateOptixInstances()
{
  m_numTriangleInstances = 0;
  m_numCurveInstances = 0;
  m_numUserInstances = 0;
  m_numVolumeInstances = 0;
  m_numLightInstances = 0;

  std::for_each(m_instances.begin(), m_instances.end(), [&](auto *inst) {
    const auto *group = inst->group();
    const size_t numTransforms = inst->numTransforms();
    if (group->containsTriangleGeometry())
      m_numTriangleInstances += numTransforms;
    if (group->containsCurveGeometry())
      m_numCurveInstances += numTransforms;
    if (group->containsUserGeometry())
      m_numUserInstances += numTransforms;
    if (group->containsVolumes())
      m_numVolumeInstances += numTransforms;
    if (group->containsLights())
      m_numLightInstances += numTransforms;
  });

  m_optixSurfaceInstances.resize(
      m_numTriangleInstances + m_numCurveInstances + m_numUserInstances);
  m_optixVolumeInstances.resize(m_numVolumeInstances);

  auto prepInstance = [](auto &i,
                          int instID,
                          size_t t,
                          auto handle,
                          int sbtOffset) -> OptixInstance {
    OptixInstance inst{};

    mat3x4 xfm = glm::transpose(i->xfm(t));
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
    const auto *group = inst->group();
    auto *osi = m_optixSurfaceInstances.dataHost();
    auto *ovi = m_optixVolumeInstances.dataHost();
    for (size_t t = 0; t < inst->numTransforms(); t++) {
      if (group->containsTriangleGeometry()) {
        osi[instID] = prepInstance(inst,
            instID,
            t,
            group->optixTraversableTriangle(),
            SBT_TRIANGLE_OFFSET);
        instID++;
      }
      if (group->containsCurveGeometry()) {
        osi[instID] = prepInstance(
            inst, instID, t, group->optixTraversableCurve(), SBT_CURVE_OFFSET);
        instID++;
      }
      if (group->containsUserGeometry()) {
        osi[instID] = prepInstance(
            inst, instID, t, group->optixTraversableUser(), SBT_CUSTOM_OFFSET);
        instID++;
      }
      if (group->containsVolumes()) {
        ovi[instVolID] = prepInstance(inst,
            instVolID,
            t,
            group->optixTraversableVolume(),
            SBT_CUSTOM_OFFSET);
        instVolID++;
      }
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

  auto makeInstanceGPUData = [](const DeviceObjectIndex *s,
                                 const UniformAttributes &ua,
                                 uint32_t id,
                                 uint32_t arrayOffset = 0) {
    InstanceSurfaceGPUData retval;

    retval.surfaces = s;
    retval.attrUniform[0] = ua.attribute0.value_or(vec4(0, 0, 0, 1));
    retval.attrUniformPresent[0] = ua.attribute0.has_value();
    retval.attrUniform[1] = ua.attribute1.value_or(vec4(0, 0, 0, 1));
    retval.attrUniformPresent[1] = ua.attribute1.has_value();
    retval.attrUniform[2] = ua.attribute2.value_or(vec4(0, 0, 0, 1));
    retval.attrUniformPresent[2] = ua.attribute2.has_value();
    retval.attrUniform[3] = ua.attribute3.value_or(vec4(0, 0, 0, 1));
    retval.attrUniformPresent[3] = ua.attribute3.has_value();
    retval.attrUniform[4] = ua.color.value_or(vec4(0, 0, 0, 1));
    retval.attrUniformPresent[4] = ua.color.has_value();

    // FIXME: Fill up retval.attrUniformArray and
    // retval.attrUniformArrayPresent from ua
    constexpr const auto setupUniformArray =
        [](const helium::IntrusivePtr<Array1D> &array) -> AttributeData {
      AttributeData ad = {};
      if (array.ptr) {
        ad.type = array->elementType();
        ad.data = array->dataGPU();
        ad.numChannels = numANARIChannels(array->elementType());
      }
      return ad;
    };

    retval.attrUniformArrayPresent[0] = ua.attribute0Array.ptr != nullptr;
    retval.attrUniformArray[0] = setupUniformArray(ua.attribute0Array);
    retval.attrUniformArrayPresent[1] = ua.attribute1Array.ptr != nullptr;
    retval.attrUniformArray[1] = setupUniformArray(ua.attribute1Array);
    retval.attrUniformArrayPresent[2] = ua.attribute2Array.ptr != nullptr;
    retval.attrUniformArray[2] = setupUniformArray(ua.attribute2Array);
    retval.attrUniformArrayPresent[3] = ua.attribute3Array.ptr != nullptr;
    retval.attrUniformArray[3] = setupUniformArray(ua.attribute3Array);
    retval.attrUniformArrayPresent[4] = ua.colorArray.ptr != nullptr;
    retval.attrUniformArray[4] = setupUniformArray(ua.colorArray);

    retval.id = id;
    retval.localArrayId = arrayOffset;

    return retval;
  };

  int instID = 0;
  std::for_each(m_instances.begin(), m_instances.end(), [&](auto *inst) {
    auto *group = inst->group();
    auto *sd = m_instanceSurfaceGPUData.dataHost();

    for (size_t t = 0; t < inst->numTransforms(); t++) {
      auto id = inst->userID(t);
      if (group->containsTriangleGeometry()) {
        sd[instID++] =
            makeInstanceGPUData(group->surfaceTriangleGPUIndices().data(),
                inst->uniformAttributes(),
                id,
                t);
      }
      if (group->containsCurveGeometry()) {
        sd[instID++] =
            makeInstanceGPUData(group->surfaceCurveGPUIndices().data(),
                inst->uniformAttributes(),
                id,
                t);
      }
      if (group->containsUserGeometry()) {
        sd[instID++] =
            makeInstanceGPUData(group->surfaceUserGPUIndices().data(),
                inst->uniformAttributes(),
                id,
                t);
      }
    }
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
    for (size_t t = 0; t < inst->numTransforms(); t++) {
      auto id = inst->userID(t);
      if (group->containsVolumes())
        vd[instID++] = {group->volumeGPUIndices().data(), id};
    }
  });

  m_instanceVolumeGPUData.upload();
}

void World::buildInstanceLightGPUData()
{
  m_instanceLightGPUData.resize(m_numLightInstances);

  m_hdri = -1;

  int instID = 0;
  std::for_each(m_instances.begin(), m_instances.end(), [&](auto *inst) {
    auto *group = inst->group();
    auto *li = m_instanceLightGPUData.dataHost();
    if (!group->containsLights())
      return;

    group->rebuildLights();
    const auto lgi = group->lightGPUIndices();

    if (m_hdri == -1)
      m_hdri = group->firstHDRI();

    for (size_t t = 0; t < inst->numTransforms(); t++)
      li[instID++] = {lgi.data(), lgi.size(), mat4(inst->xfm(t))};
  });

  m_instanceLightGPUData.upload();
}

} // namespace visrtx

VISRTX_ANARI_TYPEFOR_DEFINITION(visrtx::World *);
