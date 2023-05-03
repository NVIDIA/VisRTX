// Copyright (c) 2019-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause


#pragma once

#include "VisGLDevice.h"
#include "AppendableShader.h"

#include <array>

namespace visgl{


template <>
class Object<SamplerImage1D> : public DefaultObject<SamplerImage1D, SamplerObjectBase>
{
  GLuint sampler = 0;
  size_t transform_index;
  ObjectRef<DataArray1D> image;

  friend void image1d_init_objects(ObjectRef<SamplerImage1D> samplerObj, int filter, GLenum wrapS);
public:

  Object(ANARIDevice d, ANARIObject handle);

  void commit() override;
  void update() override;

  void allocateResources(SurfaceObjectBase*, int) override;
  void drawCommand(int index, DrawCommand &command) override;
  void declare(int index, AppendableShader &shader) override;
  void sample(int index, AppendableShader &shader, const char *meta) override;
  std::array<uint32_t, 4> metadata() override;

  ~Object();

};

} //namespace visgl

