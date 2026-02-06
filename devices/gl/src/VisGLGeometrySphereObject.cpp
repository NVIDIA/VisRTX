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


#include "anari2gl_types.h"
#include "shader_compile_segmented.h"
#include "shader_blocks.h"

#include <cstdlib>
#include <cstring>

namespace visgl {

#include "icosphere.h"

const char *ico_vert_shadow = R"GLSL(
layout(location = 0) in vec3 in_position;
layout(location = 2) in float in_radius;
layout(location = 7) in vec3 ico_position;

out Data {
  vec4 vertexPosition;
  flat vec4 center_radius;
};

void main() {
  mat4 transform = transforms[instanceIndices.x];

  float radius = max(materials[instanceIndices.z].x, abs(in_radius));
  center_radius = vec4(in_position, radius);
  vec3 offset = 1.26*ico_position*radius;
  vertexPosition = transform*vec4(in_position+offset, 1.0);
  gl_Position = vertexPosition;
}
)GLSL";

const char *ico_geom_shadow = R"GLSL(
layout(triangles) in;
layout(triangle_strip, max_vertices=36) out;

in Data {
  vec4 vertexPosition;
  flat vec4 center_radius;
} vertex_in[];

out Data {
  vec4 vertexPosition;
  flat vec4 center_radius;
} vertex_out;

void main() {

  vec4 v1 = gl_in[0].gl_Position;
  vec4 v2 = gl_in[1].gl_Position;
  vec4 v3 = gl_in[2].gl_Position;

  for(int i = 0;i<int(meta.y);++i) {

    gl_Position = shadowProjection[i].matrix*v1;
    vertex_out.vertexPosition = vertex_in[0].vertexPosition;
    vertex_out.center_radius = vertex_in[0].center_radius;
    gl_Layer = i;
    EmitVertex();

    gl_Position = shadowProjection[i].matrix*v2;
    vertex_out.vertexPosition = vertex_in[1].vertexPosition;
    vertex_out.center_radius = vertex_in[1].center_radius;
    gl_Layer = i;
    EmitVertex();

    gl_Position = shadowProjection[i].matrix*v3;
    vertex_out.vertexPosition = vertex_in[2].vertexPosition;
    vertex_out.center_radius = vertex_in[2].center_radius;
    gl_Layer = i;
    EmitVertex();

    EndPrimitive();
  }
}
)GLSL";

const char *ico_frag_shadow = R"GLSL(
in Data {
  vec4 vertexPosition;
  flat vec4 center_radius;
};

bool intersect_sphere(vec3 dir, vec3 origin, vec3 center, float radius, out float x, out vec3 normal) {
  vec3 diff = origin - center;

  float c = dot(diff, diff) - radius*radius;
  float b = 2.0*dot(diff, dir);
  float a = dot(dir, dir);

  float discr = b*b-4.0*a*c;

  float x_close = -b/(2.0*a);

  x = x_close;
  if(discr>=0.0) {
    x = (-b-sqrt(discr))/(2.0*a);
    normal = (diff + dir*x)/radius;
    return true;
  } else {
    return false;
  }
}

void main() {
  mat4 projection = shadowProjection[gl_Layer].matrix;
  mat4 transform = transforms[instanceIndices.x];
  mat4 inverseTransform = transforms[instanceIndices.x+1u];

  vec4 direction = vec4(projection[0][2], projection[1][2], projection[2][2], 0.0);

  vec3 dir = (inverseTransform*direction).xyz;
  vec3 origin = (inverseTransform*vertexPosition).xyz;

  float x = 0.0;
  vec3 normal;
  bool hit = intersect_sphere(dir, origin, center_radius.xyz, center_radius.w, x, normal);

  if(!hit) {
    discard;
  }

  vec4 projected = projection*(transform*vec4(center_radius.xyz, 1.0));
  gl_FragDepth = projected.z/projected.w*0.5 + 0.5;
}
)GLSL";

const char *ico_vert_occlusion_resolve = R"GLSL(
layout(location = 0) in vec3 in_position;
layout(location = 2) in float in_radius;

void main() {
  mat4 transform = transforms[instanceIndices.x];
  mat4 inverseTransform = transforms[instanceIndices.x+1u];

  vec4 vertexPosition = transform*vec4(in_position, 1.0);
  uint vertexId = instanceIndices.w + 4u*uint(gl_VertexID);

  vec4 sum = vec4(0.0);
  for(uint i = 0u;i<12u;++i) {
    vec3 dir = normalize(sampleShadowDir(i));

    vec4 offset = inverseTransform*vec4(dir, 0.0);
    offset.xyz = abs(in_radius)*normalize(offset.xyz);
    offset = transform*offset;

    float shadow = sampleShadowBias(vertexPosition - offset, 0.001, i);
    sum += shadow*vec4(dir, 1.0);
  }

  for(uint i = 0u;i<4u;++i) {
    float prev = occlusion[vertexId + i];
    occlusion[vertexId + i] = (prev*meta.x + sum[i])*(1.0/(meta.x + 12.0));
  }
}
)GLSL";

const char *ico_vert = R"GLSL(
layout(location = 0) in vec3 in_position;
layout(location = 1) in vec4 in_color;
layout(location = 2) in float in_radius;
layout(location = 3) in vec4 in_attr0;
layout(location = 4) in vec4 in_attr1;
layout(location = 5) in vec4 in_attr2;
layout(location = 6) in vec4 in_attr3;

layout(location = 7) in vec3 ico_position;

out Data {
  vec4 vertexPosition;
  vec4 vertexColor;
  vec4 vertexAttribute0;
  vec4 vertexAttribute1;
  vec4 vertexAttribute2;
  vec4 vertexAttribute3;
  flat vec4 center_radius;
  flat uint primitiveId;
};

void main() {
  mat4 transform = transforms[instanceIndices.x];
  mat4 projection = transforms[cameraIdx];

  vertexColor = in_color;
  vertexAttribute0 = in_attr0;
  vertexAttribute1 = in_attr1;
  vertexAttribute2 = in_attr2;
  vertexAttribute3 = in_attr3;
  primitiveId = uint(gl_InstanceID);
  float radius = max(materials[instanceIndices.z].x, abs(in_radius));
  center_radius = vec4(in_position, radius);
  vec3 offset = 1.26*ico_position*radius;
  vertexPosition = transform*vec4(in_position+offset, 1.0);

  gl_Position = projection*vertexPosition;
}
)GLSL";

const char *ico_frag = R"GLSL(
in Data {
  vec4 vertexPosition;
  vec4 vertexColor;
  vec4 vertexAttribute0;
  vec4 vertexAttribute1;
  vec4 vertexAttribute2;
  vec4 vertexAttribute3;
  flat vec4 center_radius;
  flat uint primitiveId;
};

layout(location = 0) out vec4 FragColor;

bool intersect_sphere(vec3 dir, vec3 origin, vec3 center, float radius, out float x, out vec3 normal) {
  vec3 diff = origin - center;

  float c = dot(diff, diff) - radius*radius;
  float b = 2.0*dot(diff, dir);
  float a = dot(dir, dir);

  float discr = b*b-4.0*a*c;

  float x_close = -b/(2.0*a);

  x = x_close;
  if(discr>=0.0) {
    x = (-b-sqrt(discr))/(2.0*a);
    normal = (diff + dir*x)/radius;
    return true;
  } else {
    return false;
  }
}

void main() {
  float coverage = 1.0;

  vec4 direction = transforms[cameraIdx+6u][1];
  direction = vec4(vertexPosition.xyz*direction.w - direction.xyz, 0);

  mat4 inverseTransform = transforms[instanceIndices.x+1u];
  mat4 projection = transforms[cameraIdx];

  vec3 dir = (inverseTransform*direction).xyz;
  vec3 origin = (inverseTransform*vertexPosition).xyz;

  float x = 0.0;
  vec3 normal;
  bool hit = intersect_sphere(dir, origin, center_radius.xyz, center_radius.w, x, normal);

  if(!hit) {
    discard;
  }

  mat4 normalTransform = transforms[instanceIndices.x+2u];

  vec4 objectPosition = vec4(origin + x*dir, 1.0);
  vec4 objectNormal = vec4(normal, 0);
  vec4 worldNormal = normalTransform*objectNormal;
  vec4 worldPosition = vec4(vertexPosition.xyz + x*direction.xyz, 1);
  vec3 geometryNormal = worldNormal.xyz;

  vec4 projected = projection*worldPosition;
  gl_FragDepth = projected.z/projected.w*0.5 + 0.5;

  float fragmentOcclusion = 1.0;

  if(occlusionMode != 0u) {
    uint occlusion_offset = instanceIndices.w + 4u*primitiveId;
    vec4 occ = vec4(
      occlusion[occlusion_offset + 0u],
      occlusion[occlusion_offset + 1u],
      occlusion[occlusion_offset + 2u],
      occlusion[occlusion_offset + 3u]
    );


    vec3 directionality = occ.xyz/occ.w;
    fragmentOcclusion = occ.w*(1.0 - dot(worldNormal.xyz, directionality));
    fragmentOcclusion = max(0.0, 2.0*fragmentOcclusion);
  }

  vec4 color = vertexColor;
  vec4 attribute0 = vertexAttribute0;
  vec4 attribute1 = vertexAttribute1;
  vec4 attribute2 = vertexAttribute2;
  vec4 attribute3 = vertexAttribute3;
)GLSL";

Object<GeometrySphere>::Object(ANARIDevice d, ANARIObject handle)
    : DefaultObject(d, handle)
{
  geometry_index = thisDevice->materials.allocate(1);
}

uint32_t Object<GeometrySphere>::index()
{
  return geometry_index;
}

template <typename A, typename B>
static bool compare_and_assign(A &a, const B &b)
{
  bool cmp = (a == b);
  a = b;
  return cmp;
}

void Object<GeometrySphere>::commit()
{
  DefaultObject::commit();

  dirty |= compare_and_assign(
      position_array, acquire<DataArray1D *>(current.vertex_position));
  dirty |= compare_and_assign(
      color_array, acquire<DataArray1D *>(current.vertex_color));
  dirty |= compare_and_assign(
      radius_array, acquire<DataArray1D *>(current.vertex_radius));
  dirty |= compare_and_assign(
      attribute0_array, acquire<DataArray1D *>(current.vertex_attribute0));
  dirty |= compare_and_assign(
      attribute1_array, acquire<DataArray1D *>(current.vertex_attribute1));
  dirty |= compare_and_assign(
      attribute2_array, acquire<DataArray1D *>(current.vertex_attribute2));
  dirty |= compare_and_assign(
      attribute3_array, acquire<DataArray1D *>(current.vertex_attribute3));

  // for spheres the notion of primitive is the same as vertex
  if (!color_array) {
    dirty |= compare_and_assign(
        color_array, acquire<DataArray1D *>(current.primitive_color));
  }
  if (!attribute0_array) {
    dirty |= compare_and_assign(
        attribute0_array, acquire<DataArray1D *>(current.primitive_attribute0));
  }
  if (!attribute1_array) {
    dirty |= compare_and_assign(
        attribute1_array, acquire<DataArray1D *>(current.primitive_attribute1));
  }
  if (!attribute2_array) {
    dirty |= compare_and_assign(
        attribute2_array, acquire<DataArray1D *>(current.primitive_attribute2));
  }
  if (!attribute3_array) {
    dirty |= compare_and_assign(
        attribute3_array, acquire<DataArray1D *>(current.primitive_attribute3));
  }

  dirty |= compare_and_assign(
      primitive_id_array, acquire<DataArray1D *>(current.primitive_id));

  dirty |= compare_and_assign(
      index_array, acquire<DataArray1D *>(current.primitive_index));

  radius = -1.0f;
  current.radius.get(ANARI_FLOAT32, &radius);
}

template <typename G>
void configure_vertex_array(
    G &gl, ObjectRef<DataArray1D> &array, int index, int divisor)
{
  if (array) {
    ANARIDataType bufferType = array->getBufferType();
    gl.BindBuffer(GL_ARRAY_BUFFER, array->getBuffer());
    gl.EnableVertexAttribArray(index);
    gl.VertexAttribPointer(index,
        anari::componentsOf(bufferType),
        gl_type(bufferType),
        gl_normalized(bufferType),
        0,
        0);
    if (divisor) {
      gl.VertexAttribDivisor(index, divisor);
    }
  }
}

void sphere_init_objects(ObjectRef<GeometrySphere> sphereObj)
{
  auto &gl = sphereObj->thisDevice->gl;
  if (sphereObj->vao == 0) {
    gl.GenVertexArrays(1, &sphereObj->vao);
    gl.GenVertexArrays(1, &sphereObj->occlusion_resolve_vao);

    gl.BindVertexArray(sphereObj->vao);

    gl.GenBuffers(1, &sphereObj->ico_position);
    gl.BindBuffer(GL_ARRAY_BUFFER, sphereObj->ico_position);
    gl.BufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    gl.EnableVertexAttribArray(7);
    gl.VertexAttribPointer(7, 3, GL_FLOAT, GL_FALSE, 0, 0);

    gl.GenBuffers(1, &sphereObj->ico_index);
    gl.BindBuffer(GL_ELEMENT_ARRAY_BUFFER, sphereObj->ico_index);
    gl.BufferData(
        GL_ELEMENT_ARRAY_BUFFER, sizeof(indices0), indices0, GL_STATIC_DRAW);
  }
  gl.BindVertexArray(sphereObj->vao);
  configure_vertex_array(gl, sphereObj->position_array, 0, 1);
  configure_vertex_array(gl, sphereObj->color_array, 1, 1);
  configure_vertex_array(gl, sphereObj->radius_array, 2, 1);
  configure_vertex_array(gl, sphereObj->attribute0_array, 3, 1);
  configure_vertex_array(gl, sphereObj->attribute1_array, 4, 1);
  configure_vertex_array(gl, sphereObj->attribute2_array, 5, 1);
  configure_vertex_array(gl, sphereObj->attribute3_array, 6, 1);

  gl.BindVertexArray(sphereObj->occlusion_resolve_vao);
  configure_vertex_array(gl, sphereObj->position_array, 0, 0);
  configure_vertex_array(gl, sphereObj->radius_array, 2, 0);
}

void Object<GeometrySphere>::update()
{
  DefaultObject::update();

  if (dirty) {
    thisDevice->queue.enqueue(sphere_init_objects, this);
    std::array<float, 4> data{radius, 0.0f, 0.0f, 0.0f};
    thisDevice->materials.set(geometry_index, data);
    dirty = false;
  }
}

std::array<float, 6> Object<GeometrySphere>::bounds()
{
  std::array<float, 6> b{0, 0, 0, 0, 0, 0};
  if (position_array) {
    b = position_array->getBounds();
    if (radius_array) {
      std::array<float, 6> a = radius_array->getBounds();

      b[0] += 1.26f * a[0];
      b[1] += 1.26f * a[1];
      b[2] += 1.26f * a[2];
      b[3] += 1.26f * a[3];
      b[4] += 1.26f * a[4];
      b[5] += 1.26f * a[5];
    } else {
      b[0] -= 1.26f * radius;
      b[1] -= 1.26f * radius;
      b[2] -= 1.26f * radius;
      b[3] += 1.26f * radius;
      b[4] += 1.26f * radius;
      b[5] += 1.26f * radius;
    }
  }
  return b;
}

void Object<GeometrySphere>::allocateResources(SurfaceObjectBase *) {}

void Object<GeometrySphere>::drawCommand(
    SurfaceObjectBase *, DrawCommand &command)
{
  command.vao = vao;
  command.prim = GL_TRIANGLES;
  command.count = index_count0;
  command.instanceCount = position_array->size();
  command.indexType = GL_UNSIGNED_INT;
  command.cullMode = GL_BACK;

  command.occlusion_resolve_vao = occlusion_resolve_vao;

  command.vertex_count = 4 * position_array->size();
}

void Object<GeometrySphere>::vertexShader(
    SurfaceObjectBase *, AppendableShader &shader)
{
  shader.append(ico_vert);
}

void Object<GeometrySphere>::fragmentShaderMain(
    SurfaceObjectBase *, AppendableShader &shader)
{
  shader.append(occlusion_declaration);
  shader.append(ico_frag);
}

void Object<GeometrySphere>::vertexShaderShadow(
    SurfaceObjectBase *surf, AppendableShader &shader)
{
  shader.append(ico_vert_shadow);
}

void Object<GeometrySphere>::geometryShaderShadow(
    SurfaceObjectBase *surf, AppendableShader &shader)
{
  shader.append(shadow_block_declaration);
  shader.append(ico_geom_shadow);
}

void Object<GeometrySphere>::fragmentShaderShadowMain(
    SurfaceObjectBase *surf, AppendableShader &shader)
{
  shader.append(shadow_block_declaration);
  shader.append(ico_frag_shadow);
}

void Object<GeometrySphere>::vertexShaderOcclusion(
    SurfaceObjectBase *surf, AppendableShader &shader)
{
  shader.append(shader_conversions);
  shader.append(occlusion_declaration);
  shader.append(shadow_map_declaration);
  shader.append(ico_vert_occlusion_resolve);
}

static void sphere_delete_objects(Object<Device> *deviceObj,
    GLuint vao,
    GLuint ico_position,
    GLuint ico_index)
{
  auto &gl = deviceObj->gl;
  gl.DeleteVertexArrays(1, &vao);
  gl.DeleteBuffers(1, &ico_position);
  gl.DeleteBuffers(1, &ico_index);
};

Object<GeometrySphere>::~Object()
{
  if (vao) {
    thisDevice->queue.enqueue(
        sphere_delete_objects, thisDevice, vao, ico_position, ico_index);
  }
}

} // namespace visgl
