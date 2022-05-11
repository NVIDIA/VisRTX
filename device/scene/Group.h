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

#include "array/ObjectArray.h"
#include "light/Light.h"
#include "surface/Surface.h"
#include "utility/HostDeviceArray.h"
#include "volume/Volume.h"

namespace visrtx {

struct Group : public Object
{
  static size_t objectCount();

  Group();
  ~Group() override;

  bool getProperty(const std::string_view &name,
      ANARIDataType type,
      void *ptr,
      uint32_t flags) override;

  void commit() override;

  OptixTraversableHandle optixTraversableTriangle() const;
  OptixTraversableHandle optixTraversableUser() const;
  OptixTraversableHandle optixTraversableVolume() const;

  bool containsTriangleGeometry() const;
  bool containsUserGeometry() const;
  bool containsVolumes() const;
  bool containsLights() const;

  anari::Span<const DeviceObjectIndex> surfaceTriangleGPUIndices() const;
  anari::Span<const DeviceObjectIndex> surfaceUserGPUIndices() const;
  anari::Span<const DeviceObjectIndex> volumeGPUIndices() const;
  anari::Span<const DeviceObjectIndex> lightGPUIndices() const;

  void rebuildSurfaceBVHs();
  void rebuildVolumeBVH();
  void rebuildLights();

 private:
  void partitionGeometriesByType();
  void buildSurfaceGPUData();
  void buildVolumeGPUData();
  void buildLightGPUData();
  void cleanup();

  // Geometry //

  anari::IntrusivePtr<ObjectArray> m_surfaceData;
  anari::Span<Surface *> m_surfaces;

  std::vector<Surface *> m_surfacesTriangle;
  std::vector<Surface *> m_surfacesUser;

  DeviceBuffer m_surfaceTriangleObjectIndices;
  DeviceBuffer m_surfaceUserObjectIndices;

  // Volume //

  anari::IntrusivePtr<ObjectArray> m_volumeData;
  anari::Span<Volume *> m_volumes;

  DeviceBuffer m_volumeObjectIndices;

  // Light //

  anari::IntrusivePtr<ObjectArray> m_lightData;
  anari::Span<Light *> m_lights;

  DeviceBuffer m_lightObjectIndices;

  // BVH //

  struct ObjectUpdates
  {
    TimeStamp lastSurfaceBVHBuilt{0};
    TimeStamp lastVolumeBVHBuilt{0};
    TimeStamp lastLightRebuild{0};
  } m_objectUpdates;

  box3 m_triangleBounds;
  box3 m_userBounds;
  box3 m_volumeBounds;

  OptixTraversableHandle m_traversableTriangle{};
  DeviceBuffer m_bvhTriangle;

  OptixTraversableHandle m_traversableUser{};
  DeviceBuffer m_bvhUser;

  OptixTraversableHandle m_traversableVolume{};
  DeviceBuffer m_bvhVolume;
};

} // namespace visrtx

VISRTX_ANARI_TYPEFOR_SPECIALIZATION(visrtx::Group *, ANARI_GROUP);
