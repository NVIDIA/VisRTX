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

#include "VisGLSpecializations.h"
#include "shader_blocks.h"
#include "anari2gl_types.h"

namespace visgl {
// clang-format off
const float box_vertices[] = {
  0.0f, 0.0f, 0.0f,
  0.0f, 0.0f, 1.0f,
  0.0f, 1.0f, 0.0f,
  0.0f, 1.0f, 1.0f,
  1.0f, 0.0f, 0.0f,
  1.0f, 0.0f, 1.0f,
  1.0f, 1.0f, 0.0f,
  1.0f, 1.0f, 1.0f,
};

const uint32_t box_indices[] {
  0, 1, 2, 3, 2, 1,
  0, 4, 1, 5, 1, 4,
  0, 2, 4, 6, 4, 2,
  4, 6, 5, 7, 5, 6,
  2, 3, 6, 7, 6, 3,
  1, 5, 3, 7, 3, 5,
};
// clang-format on

const char *box_vert = R"GLSL(
layout(location = 0) in vec4 in_position;

out Data {
  vec4 vertexPosition;
  vec4 cells;
};

void main() {
  mat4 transform = transforms[instanceIndices.x];
  mat4 projection = transforms[cameraIdx];

  vec4 origin = materials[instanceIndices.z];
  vec4 spacing = materials[instanceIndices.z+1u];
  uvec4 dims = floatBitsToUint(materials[instanceIndices.z+2u]);

  cells = in_position;
  vertexPosition = origin + in_position*spacing*vec4(dims);
  vertexPosition = transform*vertexPosition;
  gl_Position = projection*vertexPosition;
}
)GLSL";

const char *box_frag = R"GLSL(
in Data {
  vec4 vertexPosition;
  vec4 cells;
};

layout(location = 0) out vec4 FragColor;

layout(binding = 0) uniform highp sampler3D fieldSampler;
vec4 sampleField(vec4 coord, uint index) {
  mat4 transform = transforms[index];
  return texture(fieldSampler, (transform*coord).xyz);
}

void main() {
  mat4 transform = transforms[instanceIndices.x];
  mat4 inverseTransform = transforms[instanceIndices.x+1u];
  mat4 projection = transforms[cameraIdx];

  vec4 origin = materials[instanceIndices.z];
  vec4 spacing = materials[instanceIndices.z+1u];
  uvec4 dims = floatBitsToUint(materials[instanceIndices.z+2u]);

  // view direction
  vec4 direction = transforms[cameraIdx+6u][1];
  direction = vec4(vertexPosition.xyz*direction.w - direction.xyz, 0);

  // pretransform ray direction
  vec3 ray_dir = (inverseTransform*direction).xyz;
  vec3 ray_origin = (inverseTransform*vertexPosition).xyz;

  ray_origin = (ray_origin - origin.xyz)/(spacing.xyz*vec3(dims.xyz));
  ray_dir = ray_dir/(spacing.xyz*vec3(dims.xyz));

  vec3 intersects = (step(0.0, ray_dir) - ray_origin)/ray_dir;
  float s = min(min(intersects.x, intersects.y), intersects.z);
  vec3 ray_end = ray_origin + s*ray_dir;

  FragColor = vec4(0.0);

  float density_scale = 0.03;

  density_scale *= distance(ray_end, ray_origin);

  for(float i = 0.0;i<=1.0;i+=0.01) {
    vec3 x = mix(ray_end, ray_origin, vec3(i));
    vec4 c = transferSample(texture(fieldSampler, x));
    FragColor.xyz = mix(FragColor.xyz, c.xyz, density_scale*c.w);
    FragColor.w += density_scale*c.w;
  }
}
)GLSL";

Object<Spatial_FieldStructuredRegular>::Object(
    ANARIDevice d, ANARIObject handle)
    : DefaultObject(d, handle)
{
  transform_index = thisDevice->materials.allocate(3);
}

void field_init_objects(
    ObjectRef<Spatial_FieldStructuredRegular> fieldObj, int filter)
{
  auto &gl = fieldObj->thisDevice->gl;
  if (fieldObj->sampler == 0) {
    gl.GenSamplers(1, &fieldObj->sampler);
  }
  gl.SamplerParameteri(
      fieldObj->sampler, GL_TEXTURE_MAG_FILTER, gl_mag_filter(filter));
  gl.SamplerParameteri(
      fieldObj->sampler, GL_TEXTURE_MIN_FILTER, gl_min_filter(filter));
  gl.SamplerParameteri(fieldObj->sampler, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  gl.SamplerParameteri(fieldObj->sampler, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  gl.SamplerParameteri(fieldObj->sampler, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

  if (fieldObj->vao == 0) {
    gl.GenVertexArrays(1, &fieldObj->vao);
    gl.BindVertexArray(fieldObj->vao);

    gl.GenBuffers(1, &fieldObj->box_position);
    gl.BindBuffer(GL_ARRAY_BUFFER, fieldObj->box_position);
    gl.BufferData(
        GL_ARRAY_BUFFER, sizeof(box_vertices), box_vertices, GL_STATIC_DRAW);
    gl.EnableVertexAttribArray(0);
    gl.VertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);

    gl.GenBuffers(1, &fieldObj->box_index);
    gl.BindBuffer(GL_ELEMENT_ARRAY_BUFFER, fieldObj->box_index);
    gl.BufferData(GL_ELEMENT_ARRAY_BUFFER,
        sizeof(box_indices),
        box_indices,
        GL_STATIC_DRAW);
  }
}

void Object<Spatial_FieldStructuredRegular>::commit()
{
  DefaultObject::commit();
  data = acquire<Object<Array3D> *>(current.data);
}

void Object<Spatial_FieldStructuredRegular>::update()
{
  DefaultObject::update();

  current.origin.get(ANARI_FLOAT32_VEC3, origin.data());
  current.spacing.get(ANARI_FLOAT32_VEC3, spacing.data());

  uint64_t dims[3] = {0u, 0u, 0u};
  data->dims(dims);
  std::array<uint32_t, 4> udims{
      (uint32_t)dims[0], (uint32_t)dims[1], (uint32_t)dims[2], 0u};

  thisDevice->materials.set(transform_index, origin);
  thisDevice->materials.set(transform_index + 1, spacing);
  thisDevice->materials.setMem(transform_index + 2, &udims);

  int filter = current.filter.getStringEnum();

  thisDevice->queue.enqueue(field_init_objects, this, filter);
}

void Object<Spatial_FieldStructuredRegular>::drawCommand(
    VolumeObjectBase *, DrawCommand &command)
{
  command.vao = vao;
  command.prim = GL_TRIANGLES;
  command.count = 36;
  command.indexType = GL_UNSIGNED_INT;
  command.instanceCount = 1;
  command.cullMode = 0;

  if (data) {
    auto &tex = command.textures[command.texcount];
    tex.index = 0;
    tex.target = GL_TEXTURE_3D;
    tex.texture = data->getTexture3D();
    tex.sampler = sampler;
    command.texcount += 1;
  }
}

void Object<Spatial_FieldStructuredRegular>::vertexShaderMain(
    VolumeObjectBase *, AppendableShader &shader)
{
  shader.append(box_vert);
}

void Object<Spatial_FieldStructuredRegular>::fragmentShaderMain(
    VolumeObjectBase *, AppendableShader &shader)
{
  shader.append(box_frag);
}

uint32_t Object<Spatial_FieldStructuredRegular>::index()
{
  return transform_index;
}

std::array<float, 6> Object<Spatial_FieldStructuredRegular>::bounds()
{
  uint64_t dims[3] = {0u, 0u, 0u};
  data->dims(dims);
  return std::array<float, 6>{origin[0],
      origin[1],
      origin[2],
      origin[0] + spacing[0] * dims[0],
      origin[1] + spacing[1] * dims[1],
      origin[2] + spacing[2] * dims[2]};
}

static void field_delete_objects(Object<Device> *deviceObj, GLuint sampler)
{
  deviceObj->gl.DeleteSamplers(1, &sampler);
}

Object<Spatial_FieldStructuredRegular>::~Object()
{
  thisDevice->queue.enqueue(field_delete_objects, thisDevice, sampler);
}

} // namespace visgl
