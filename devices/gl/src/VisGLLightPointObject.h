// Copyright (c) 2019-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause


#pragma once

#include "VisGLDevice.h"

#include <array>

namespace visgl{


template <>
class Object<LightPoint> : public DefaultObject<LightPoint, LightObjectBase>
{

  std::array<float, 4> color;
  std::array<float, 4> position;
  size_t light_index;
  bool dirty = true;
public:

  Object(ANARIDevice d, ANARIObject handle);

  void commit() override;
  void update() override;
  uint32_t index() override;
};

} //namespace visgl

