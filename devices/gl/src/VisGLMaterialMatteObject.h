// Copyright (c) 2019-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause


#pragma once

#include "VisGLDevice.h"

namespace visgl{


template <>
class Object<MaterialMatte> : public DefaultObject<MaterialMatte, MaterialObjectBase>
{
  size_t material_index;
public:

  Object(ANARIDevice d, ANARIObject handle);

  void commit() override;
  void allocateResources(SurfaceObjectBase*) override;
  void drawCommand(SurfaceObjectBase*, DrawCommand&) override;
  void fragmentShaderDeclarations(SurfaceObjectBase*, AppendableShader&) override;
  void fragmentShaderMain(SurfaceObjectBase*, AppendableShader&) override;

  void fragmentShaderShadowDeclarations(SurfaceObjectBase*, AppendableShader&) override;
  void fragmentShaderShadowMain(SurfaceObjectBase*, AppendableShader&) override;

  uint32_t index() override;
};

} //namespace visgl

