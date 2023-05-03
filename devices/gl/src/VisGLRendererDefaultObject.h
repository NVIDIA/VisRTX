// Copyright (c) 2019-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause


#pragma once

#include "VisGLDevice.h"

namespace visgl{

template <>
class Object<RendererDefault> : public DefaultObject<RendererDefault>
{
  std::array<float, 4> ambient;
  size_t ambient_index;
  bool dirty = true;
public:

  Object(ANARIDevice d, ANARIObject handle);

  void commit() override;
  void update() override;
  uint32_t index();
};

} //namespace visgl

