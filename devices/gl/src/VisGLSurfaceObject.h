/*
 * Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "VisGLDevice.h"
#include "shader_blocks.h"

namespace visgl {

template <>
class Object<Surface> : public DefaultObject<Surface, SurfaceObjectBase>
{
  ObjectRef<GeometryObjectBase> geometry;
  uint64_t geometry_epoch = 0;
  ObjectRef<MaterialObjectBase> material;
  uint64_t material_epoch = 0;

  enum ResourceType
  {
    NONE = 0,
    SSBO,
    TEX,
    TRANSFORM
  };
  struct TexInfo
  {
    GLuint texture;
    GLuint sampler;
    GLenum target;
  };
  struct SSBOInfo
  {
    GLuint buffer;
  };

  struct ResourceBinding
  {
    ResourceType type;
    int index;
    union
    {
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

  void allocateTexture(
      int slot, GLenum target, GLuint texture, GLuint sampler) override;
  void allocateStorageBuffer(int slot, GLuint buffer) override;
  void allocateTransform(int slot) override;
  int resourceIndex(int slot) override;
  void addAttributeFlags(int attrib, uint32_t flags) override;
  uint32_t getAttributeFlags(int attrib) override;

  void commit() override;
  void update() override;
  void drawCommand(DrawCommand &) override;
};

} // namespace visgl
