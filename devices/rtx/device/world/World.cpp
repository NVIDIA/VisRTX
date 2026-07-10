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
#include "surface/Surface.h"
#include "utility/AnariTypeHelpers.h"
#include "world/Group.h"

#include <glm/gtc/matrix_inverse.hpp>
#include <cassert>
#include <cmath>
#include <set>

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
  buildInput.instanceArray.instances = (CUdeviceptr)optixInstancesDevice.data();
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

  retval.hdriLightInstances = m_instanceHdriLightGPUData.dataDevice();
  retval.numHdriLightInstances = m_instanceHdriLightGPUData.size();

  retval.lightPickCdf = m_lightPickCdf.dataDevice();
  retval.totalLightPower = m_totalLightPower;
  retval.hdriPower = m_hdriPower;
  retval.sceneRadius = m_sceneRadius;

  return retval;
}

void World::rebuildWorld()
{
  const auto &state = *deviceState();

  const auto &updates = state.objectUpdates;
  const auto lastCheck = m_objectUpdates.lastBLASCheck;
  if (updates.lastSurfaceBLASChange >= lastCheck
      || updates.lastVolumeBLASChange >= lastCheck
      || updates.lastLightSetChange >= lastCheck) {
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

      const mat4 o2wMat4 = mat4(inst->xfm(t));
      const mat3x4 o2w = glm::transpose(mat4x3(o2wMat4));
      const mat3x4 w2o =
          glm::transpose(mat4x3(glm::affineInverse(o2wMat4)));

      auto assignXfm = [&](InstanceSurfaceGPUData &g) {
        g.objectToWorld = o2w;
        g.worldToObject = w2o;
      };

      if (group->containsTriangleGeometry()) {
        sd[instID] =
            makeInstanceGPUData(group->surfaceTriangleGPUIndices().data(),
                inst->uniformAttributes(),
                id,
                t);
        assignXfm(sd[instID]);
        ++instID;
      }
      if (group->containsCurveGeometry()) {
        sd[instID] =
            makeInstanceGPUData(group->surfaceCurveGPUIndices().data(),
                inst->uniformAttributes(),
                id,
                t);
        assignXfm(sd[instID]);
        ++instID;
      }
      if (group->containsUserGeometry()) {
        sd[instID] =
            makeInstanceGPUData(group->surfaceUserGPUIndices().data(),
                inst->uniformAttributes(),
                id,
                t);
        assignXfm(sd[instID]);
        ++instID;
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
      if (!group->containsVolumes())
        continue;

      const mat4 o2wMat4 = mat4(inst->xfm(t));
      const mat3x4 o2w = glm::transpose(mat4x3(o2wMat4));
      const mat3x4 w2o =
          glm::transpose(mat4x3(glm::affineInverse(o2wMat4)));

      InstanceVolumeGPUData g;
      g.objectToWorld = o2w;
      g.worldToObject = w2o;
      g.volumes = group->volumeGPUIndices().data();
      g.id = id;
      vd[instID++] = g;
    }
  });

  m_instanceVolumeGPUData.upload();
}

size_t World::countGeometryLights(Group *group) const
{
  size_t n = 0;
  for (auto *surface : group->surfacesTriangle())
    if (surface->geometryLight())
      ++n;
  for (auto *surface : group->surfacesUser())
    if (surface->geometryLight())
      ++n;
  return n;
}

void World::synthesizeGeometryLights()
{
  // Configure or drop each candidate surface's Geometry Light from current
  // material + geometry state. Runs over triangle and user (custom-primitive,
  // e.g. sphere) surfaces; isSampleableEmitter gates out non-area-samplable ones.
  // Object-space, so done once per group; the fill pass instances it per
  // transform like an authored light. The light carries the material's mean
  // radiance (Pick Power); the sampler evaluates the material at the sampled
  // point when its emission is not constant.
  auto configure = [](Surface *surface) {
    if (surface->isSampleableEmitter()) {
      auto *geometry = surface->geometry();
      geometry->ensureAreaData();
      const float area = geometry->totalArea();
      if (area > 0.0f) {
        surface->ensureGeometryLight()->configure(geometry->index(),
            surface->material()->index(),
            surface->material()->emissionAverage(),
            area);
        return;
      }
    }
    surface->clearGeometryLight();
  };

  std::set<Group *> visited;
  for (auto *inst : m_instances) {
    auto *group = inst->group();
    if (!visited.insert(group).second)
      continue;
    for (auto *surface : group->surfacesTriangle())
      configure(surface);
    for (auto *surface : group->surfacesUser())
      configure(surface);
  }
}

void World::buildInstanceLightGPUData()
{
  synthesizeGeometryLights();

  // Calculate total lights (authored + synthesized Geometry Lights)
  size_t totalLights = 0;
  size_t totalHdriLights = 0;

  std::for_each(m_instances.begin(), m_instances.end(), [&](auto *inst) {
    auto *group = inst->group();
    group->rebuildLights();
    const auto &lights = group->lights();
    const size_t numTransforms = inst->numTransforms();

    totalLights +=
        (lights.size() + countGeometryLights(group)) * numTransforms;

    // Count HDRI lights separately
    for (auto *light : lights) {
      if (light->isHDRI())
        totalHdriLights += numTransforms;
    }
  });

  // Allocate both arrays
  m_instanceLightGPUData.resize(totalLights);
  m_instanceHdriLightGPUData.resize(totalHdriLights);
  m_lightPickCdf.resize(totalLights);

  // Bounding-sphere radius over the committed scene, sizing the infinite
  // lights' Pick Power. Fall back to unit radius so an empty scene still
  // weights them nonzero.
  box3 sceneBounds = m_surfaceBounds;
  if (!empty(m_volumeBounds)) {
    if (empty(sceneBounds))
      sceneBounds = m_volumeBounds;
    else {
      sceneBounds.lower = glm::min(sceneBounds.lower, m_volumeBounds.lower);
      sceneBounds.upper = glm::max(sceneBounds.upper, m_volumeBounds.upper);
    }
  }
  m_sceneRadius = empty(sceneBounds)
      ? 1.0f
      : 0.5f * glm::length(sceneBounds.upper - sceneBounds.lower);
  if (m_sceneRadius <= 0.0f)
    m_sceneRadius = 1.0f;

  size_t lightIndex = 0;
  size_t hdriIndex = 0;
  // Mirrors buildInstanceSurfaceGPUData's instID: surface instances are laid out
  // per (instance, transform) as [triangle?][curve?][user?]. Tracking the same
  // cursor here recovers each Geometry Light's surface-instance index without a
  // side table.
  size_t surfaceInstanceCursor = 0;

  // Filled with each instance's raw Pick Power, then normalized into the
  // cumulative CDF in place once the total is known.
  auto *pickCdf = m_lightPickCdf.dataHost();
  m_totalLightPower = 0.0f;
  m_hdriPower = 0.0f;

  std::for_each(m_instances.begin(), m_instances.end(), [&](auto *inst) {
    auto *group = inst->group();
    group->rebuildLights();

    auto *lights = m_instanceLightGPUData.dataHost();
    auto *hdris = m_instanceHdriLightGPUData.dataHost();

    auto appendLight = [&](Light *light,
                           const mat4 &xfm,
                           DeviceObjectIndex surfaceInstanceIndex) {
      // Sanitize: a NaN/Inf/negative Pick Power (bad param, degenerate xfm)
      // would corrupt the cumulative CDF and make cub::LowerBound undefined.
      // Clamp to 0 so the light is simply never picked. `!(power > 0)` catches
      // NaN.
      const float raw = light->pickPower(xfm, m_sceneRadius);
      const float power = (raw > 0.0f && std::isfinite(raw)) ? raw : 0.0f;
      pickCdf[lightIndex] = power;
      m_totalLightPower += power;
      lights[lightIndex++] = {light->index(), xfm, surfaceInstanceIndex};
      return power;
    };

    for (size_t t = 0; t < inst->numTransforms(); t++) {
      const mat4 xfm = mat4(inst->xfm(t));

      // This transform's surface-instance slot indices, in the same order the
      // surface-instance array was built (triangle, then curve, then user).
      DeviceObjectIndex triangleSI = -1, userSI = -1;
      if (group->containsTriangleGeometry())
        triangleSI = DeviceObjectIndex(surfaceInstanceCursor++);
      if (group->containsCurveGeometry())
        ++surfaceInstanceCursor;
      if (group->containsUserGeometry())
        userSI = DeviceObjectIndex(surfaceInstanceCursor++);

      for (auto *light : group->lights()) {
        const float power = appendLight(light, xfm, -1);
        // HDRI lights also go into hdriLights
        if (light->isHDRI()) {
          m_hdriPower += power;
          hdris[hdriIndex++] = {light->index(), xfm, -1};
        }
      }

      // Synthesized Geometry Lights, instanced exactly like authored lights but
      // carrying their surface-instance index for instance-attribute emission.
      for (auto *surface : group->surfacesTriangle()) {
        if (auto *gl = surface->geometryLight())
          appendLight(gl, xfm, triangleSI);
      }
      for (auto *surface : group->surfacesUser()) {
        if (auto *gl = surface->geometryLight())
          appendLight(gl, xfm, userSI);
      }
    }
  });

  // Tripwire: the surface-instance cursor is a hand-mirror of
  // buildInstanceSurfaceGPUData's layout with no shared source of truth. If the
  // two ever drift, a Geometry Light would index the wrong (or an out-of-range)
  // surface instance and emit silently wrong radiance — catch it here.
  assert(surfaceInstanceCursor == m_instanceSurfaceGPUData.size());

  // Turn the per-instance Pick Powers into a normalized cumulative CDF in place.
  // A zero total (every light dark) leaves the CDF unused: the renderer falls
  // back to a uniform pick.
  if (m_totalLightPower > 0.0f) {
    float cumulative = 0.0f;
    for (size_t i = 0; i < totalLights; ++i) {
      cumulative += pickCdf[i];
      pickCdf[i] = cumulative / m_totalLightPower;
    }
  }

  m_instanceLightGPUData.upload();
  m_instanceHdriLightGPUData.upload();
  m_lightPickCdf.upload();
}

} // namespace visrtx

VISRTX_ANARI_TYPEFOR_DEFINITION(visrtx::World *);
