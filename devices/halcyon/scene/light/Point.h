// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "Light.h"

namespace halcyon {

// The 'point' light (KHR_LIGHT_POINT): a delta point, or with 'radius' a
// sphere of uniform radiance (Light.h has the units)
struct Point : public Light
{
  Point(HalcyonDeviceGlobalState *s);

  void commitParameters() override;
  void finalize() override;

  bool writeWorldRecord(void *record, const mat4 &xfm) const override;

 private:
  float3 m_position{0.f};
  float m_radius{0.f};
  // see Light.h
  float3 m_emission{1.f};
};

} // namespace halcyon
