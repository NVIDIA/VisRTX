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

#include "array/ObjectArray.h"
#include "light/Light.h"
#include "surface/Surface.h"
#include "utility/HostDeviceArray.h"
#include "volume/Volume.h"

namespace visrtx {

struct Group : public Object
{
  Group(DeviceGlobalState *d);
  ~Group() override;

  bool getProperty(const std::string_view &name,
      ANARIDataType type,
      void *ptr,
      uint64_t size,
      uint32_t flags) override;

  void commitParameters() override;
  void finalize() override;
  void markFinalized() override;

  OptixTraversableHandle optixTraversableTriangle() const;
  OptixTraversableHandle optixTraversableCurve() const;
  OptixTraversableHandle optixTraversableUser() const;
  OptixTraversableHandle optixTraversableVolume() const;

  bool containsTriangleGeometry() const;
  bool containsCurveGeometry() const;
  bool containsUserGeometry() const;
  bool containsVolumes() const;
  bool containsLights() const;

  Span<DeviceObjectIndex> surfaceTriangleGPUIndices() const;
  Span<DeviceObjectIndex> surfaceCurveGPUIndices() const;
  Span<DeviceObjectIndex> surfaceUserGPUIndices() const;
  Span<DeviceObjectIndex> volumeGPUIndices() const;
  Span<DeviceObjectIndex> lightGPUIndices() const;

  DeviceObjectIndex firstHDRI() const;
  const std::vector<Light *> &lights() const;

  void rebuildSurfaceBVHs();
  void rebuildVolumeBVH();
  void rebuildLights();

 private:
  void partitionValidGeometriesByType();
  void partitionValidVolumes();
  void partitionValidLights();
  void buildSurfaceGPUData();
  void buildVolumeGPUData();
  void buildLightGPUData();

  // Geometry //

  helium::ChangeObserverPtr<ObjectArray> m_surfaceData;

  std::vector<Surface *> m_surfacesTriangle;
  std::vector<Surface *> m_surfacesCurve;
  std::vector<Surface *> m_surfacesUser;

  DeviceBuffer m_surfaceTriangleObjectIndices;
  DeviceBuffer m_surfaceCurveObjectIndices;
  DeviceBuffer m_surfaceUserObjectIndices;

  // Volume //

  helium::ChangeObserverPtr<ObjectArray> m_volumeData;
  std::vector<Volume *> m_volumes;

  DeviceBuffer m_volumeObjectIndices;

  // Light //

  helium::ChangeObserverPtr<ObjectArray> m_lightData;
  std::vector<Light *> m_lights;

  DeviceBuffer m_lightObjectIndices;
  DeviceObjectIndex m_firstHDRI{-1};

  // BVH //

  struct ObjectUpdates
  {
    helium::TimeStamp lastSurfaceBVHBuilt{0};
    helium::TimeStamp lastVolumeBVHBuilt{0};
    helium::TimeStamp lastLightRebuild{0};
  } m_objectUpdates;

  box3 m_triangleBounds;
  box3 m_curveBounds;
  box3 m_userBounds;
  box3 m_volumeBounds;

  OptixTraversableHandle m_traversableTriangle{};
  DeviceBuffer m_bvhTriangle;

  OptixTraversableHandle m_traversableCurve{};
  DeviceBuffer m_bvhCurve;

  OptixTraversableHandle m_traversableUser{};
  DeviceBuffer m_bvhUser;

  OptixTraversableHandle m_traversableVolume{};
  DeviceBuffer m_bvhVolume;
};

} // namespace visrtx

VISRTX_ANARI_TYPEFOR_SPECIALIZATION(visrtx::Group *, ANARI_GROUP);
