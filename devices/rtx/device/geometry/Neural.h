// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include "Geometry.h"

namespace visrtx {

struct Neural : public Geometry
{
  Neural(DeviceGlobalState *d);
  ~Neural() override;

  void commitParameters() override;
  void finalize() override;
  bool isValid() const override;

  void populateBuildInput(OptixBuildInput &) const override;

  int optixGeometryType() const override;

 private:
  GeometryGPUData gpuData() const override;
  mutable std::vector<helium::IntrusivePtr<Array1D>> m_layers;

  // Bounding box
  mutable box3 m_aabb;
  HostDeviceArray<box3> m_aabbs;
  CUdeviceptr m_aabbsBufferPtr{};
};

} // namespace visrtx