// Copyright (c) 2019-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "VisGLDevice.h"

#include <array>
#include <vector>

namespace visgl{


template <>
class Object<VolumeScivis> : public DefaultObject<VolumeScivis, VolumeObjectBase>
{
  bool dirty = true;

  ObjectRef<SpatialFieldObjectBase> field;
  ObjectRef<DataArray1D> color;
  ObjectRef<DataArray1D> color_position;
  ObjectRef<DataArray1D> opacity;
  ObjectRef<DataArray1D> opacity_position;
  std::array<float, 2> valueRange;
  float densityScale;

  friend void scivis_init_objects(ObjectRef<VolumeScivis> scivisObj);

  size_t material_index;

  GLuint lut = 0;
  std::vector<std::array<float, 4>> lutData;
 public:
  GLuint shader = 0;

  Object(ANARIDevice d, ANARIObject handle);
  ~Object();

  void commit() override;
  void update() override;
  void drawCommand(DrawCommand&) override;
  uint32_t index() override;
};

} //namespace visgl

