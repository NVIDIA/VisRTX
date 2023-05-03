// Copyright (c) 2019-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause


#pragma once

#include "VisGLDevice.h"
#include "shader_blocks.h"

namespace visgl{

template <>
class Object<Surface> : public DefaultObject<Surface, SurfaceObjectBase>
{
  ObjectRef<GeometryObjectBase> geometry;
  uint64_t geometry_epoch = 0;
  ObjectRef<MaterialObjectBase> material;
  uint64_t material_epoch = 0;

  enum ResourceType {NONE = 0, SSBO, TEX, TRANSFORM};
  struct TexInfo {
    GLuint texture;
    GLuint sampler;
    GLenum target;
  };
  struct SSBOInfo {
    GLuint buffer;
  };

  struct ResourceBinding {
    ResourceType type;
    int index;
    union {
      TexInfo tex;
      SSBOInfo ssbo;
    };
  };

  std::array<ResourceBinding, SURFACE_MAX_RESOURCES> resources;
  int ssboCount;
  int texCount;
  int transformCount;
  std::array<uint32_t, ATTRIBUTE_COUNT> attributeFlags;

public:
  GLuint shader = 0;
  GLuint shadow_shader = 0;
  GLuint occlusion_shader = 0;


  Object(ANARIDevice d, ANARIObject handle);

  void allocateTexture(int slot, GLenum target, GLuint texture, GLuint sampler) override;
  void allocateStorageBuffer(int slot, GLuint buffer) override;
  void allocateTransform(int slot) override;
  int resourceIndex(int slot) override;
  void addAttributeFlags(int attrib, uint32_t flags) override;
  uint32_t getAttributeFlags(int attrib) override;


  void commit() override;
  void update() override;
  void drawCommand(DrawCommand&) override;
};

} //namespace visgl

