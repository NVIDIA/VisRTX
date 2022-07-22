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

#include "Instance.h"
#include "utility/HostDeviceArray.h"

namespace visrtx {

ptx_ptr intersection_ptx();

struct World : public Object
{
  static size_t objectCount();

  World();
  ~World() override;

  bool getProperty(const std::string_view &name,
      ANARIDataType type,
      void *ptr,
      uint32_t flags) override;

  void commit() override;

  OptixTraversableHandle optixTraversableHandleSurfaces() const;
  OptixTraversableHandle optixTraversableHandleVolumes() const;

  anari::Span<const InstanceSurfaceGPUData> instanceSurfaceGPUData() const;
  anari::Span<const InstanceVolumeGPUData> instanceVolumeGPUData() const;
  anari::Span<const InstanceLightGPUData> instanceLightGPUData() const;

  void rebuildBVHs();

 private:
  void populateOptixInstances();
  void rebuildBLASs();
  void buildInstanceSurfaceGPUData();
  void buildInstanceVolumeGPUData();
  void buildInstanceLightGPUData();
  void cleanup();

  anari::IntrusivePtr<ObjectArray> m_zeroSurfaceData;
  anari::IntrusivePtr<ObjectArray> m_zeroVolumeData;
  anari::IntrusivePtr<ObjectArray> m_zeroLightData;

  anari::IntrusivePtr<ObjectArray> m_instanceData;
  anari::Span<Instance *> m_instances;

  bool m_addZeroInstance{false};
  anari::IntrusivePtr<Group> m_zeroGroup;
  anari::IntrusivePtr<Instance> m_zeroInstance;

  size_t m_numTriangleInstances{0};
  size_t m_numUserInstances{0};
  size_t m_numVolumeInstances{0};
  size_t m_numLightInstances{0};

  box3 m_surfaceBounds;
  box3 m_volumeBounds;

  struct ObjectUpdates
  {
    TimeStamp lastTLASBuild{0};
    TimeStamp lastBLASCheck{0};
  } m_objectUpdates;

  // Surfaces //

  OptixTraversableHandle m_traversableSurfaces{};
  DeviceBuffer m_bvhSurfaces;
  HostDeviceArray<OptixInstance> m_optixSurfaceInstances;

  HostDeviceArray<InstanceSurfaceGPUData> m_instanceSurfaceGPUData;

  // Volumes //

  OptixTraversableHandle m_traversableVolumes{};
  DeviceBuffer m_bvhVolumes;
  HostDeviceArray<OptixInstance> m_optixVolumeInstances;

  HostDeviceArray<InstanceVolumeGPUData> m_instanceVolumeGPUData;

  // Lights //

  HostDeviceArray<InstanceLightGPUData> m_instanceLightGPUData;
};

} // namespace visrtx

VISRTX_ANARI_TYPEFOR_SPECIALIZATION(visrtx::World *, ANARI_WORLD);
