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

const char *triangle_vert_shadow = R"GLSL(
layout(location = 0) in vec3 in_position;

void main() {
  mat4 transform = transforms[instanceIndices.x];

  gl_Position = transform*vec4(in_position, 1.0);
}
)GLSL";

const char *triangle_geom_shadow = R"GLSL(
layout(triangles) in;
layout(triangle_strip, max_vertices=36) out;

void main() {

  vec4 v1 = gl_in[0].gl_Position;
  vec4 v2 = gl_in[1].gl_Position;
  vec4 v3 = gl_in[2].gl_Position;

  for(int i = 0;i<int(meta.y);++i) {

    gl_Position = shadowProjection[i].matrix*v1;
    gl_Layer = i;
    EmitVertex();

    gl_Position = shadowProjection[i].matrix*v2;
    gl_Layer = i;
    EmitVertex();

    gl_Position = shadowProjection[i].matrix*v3;
    gl_Layer = i;
    EmitVertex();

    EndPrimitive();
  }
}
)GLSL";

const char *triangle_vert_occlusion_resolve = R"GLSL(
layout(location = 0) in vec3 in_position;

void main() {
  mat4 transform = transforms[instanceIndices.x];

  vec4 vertexPosition = transform*vec4(in_position, 1.0);
  uint vertexId = instanceIndices.w + uint(gl_VertexID);

  float sum = 0.0;
  for(uint i = 0u;i<12u;++i) {
    sum += sampleShadowBias(vertexPosition, 0.001, i);
  }
  float prev = occlusion[vertexId];
  occlusion[vertexId] = (prev*meta.x + sum)/(meta.x + 12.0);
}
)GLSL";

const char *triangle_vert = R"GLSL(
layout(location = 0) in vec3 in_position;
layout(location = 1) in vec4 in_color;
layout(location = 2) in vec3 in_normal;
layout(location = 3) in vec4 in_attr0;
layout(location = 4) in vec4 in_attr1;
layout(location = 5) in vec4 in_attr2;
layout(location = 6) in vec4 in_attr3;

void main() {
  mat4 transform = transforms[instanceIndices.x];
  mat3 normalTransform = mat3(transforms[instanceIndices.x+2u]);
  mat4 projection = transforms[cameraIdx];

  vertexOcclusion = 1.0;
  if(occlusionMode != 0u) {
    vertexOcclusion = min(1.0, 2.0*occlusion[instanceIndices.w + uint(gl_VertexID)]);
  }

  vertexColor = in_color;
  vertexAttribute0 = in_attr0;
  vertexAttribute1 = in_attr1;
  vertexAttribute2 = in_attr2;
  vertexAttribute3 = in_attr3;
  vertexNormal = normalTransform*in_normal;
  vertexPosition = transform*vec4(in_position, 1.0);
  gl_Position = projection*vertexPosition;
}
)GLSL";

const char *triangle_frag = R"GLSL(
layout(location = 0) out vec4 FragColor;

const float coverage = 1.0;

void main() {
  vec4 worldPosition = vertexPosition;
  mat4 inverseTransform = transforms[instanceIndices.x+1u];
  vec4 objectPosition = inverseTransform*worldPosition;

  //derived geometry normals
  vec3 a = dFdx(vertexPosition.xyz);
  vec3 b = dFdy(vertexPosition.xyz);
  vec3 geometryNormal = normalize(cross(a, b));

  vec4 worldNormal = vec4(0);
  if(vertexNormal == vec3(0)) {
    //synthesize normals if required
    worldNormal.xyz = geometryNormal;
  } else {
    worldNormal.xyz = normalize(vertexNormal);
    if(!gl_FrontFacing) {
      worldNormal.xyz = -worldNormal.xyz;
    }
  }
  mat3 inverseNormalTransform = transpose(mat3(transforms[instanceIndices.x]));
  vec4 objectNormal = vec4(normalize(inverseNormalTransform*worldNormal.xyz), 0);

  float fragmentOcclusion = vertexOcclusion;

  vec4 color = vertexColor;
  vec4 attribute0 = vertexAttribute0;
  vec4 attribute1 = vertexAttribute1;
  vec4 attribute2 = vertexAttribute2;
  vec4 attribute3 = vertexAttribute3;
  uint primitiveId = uint(gl_PrimitiveID);
)GLSL";

const char *gl_primitive_id = "uint(gl_PrimitiveID));\n";

Object<GeometryTriangle>::Object(ANARIDevice d, ANARIObject handle)
    : DefaultObject(d, handle)
{}

uint32_t Object<GeometryTriangle>::index()
{
  return 0;
}

template <typename A, typename B>
static bool compare_and_assign(A &a, const B &b)
{
  bool cmp = (a == b);
  a = b;
  return cmp;
}

void Object<GeometryTriangle>::commit()
{
  DefaultObject::commit();

  dirty |= compare_and_assign(
      position_array, acquire<DataArray1D *>(current.vertex_position));
  dirty |= compare_and_assign(
      color_array, acquire<DataArray1D *>(current.vertex_color));
  dirty |= compare_and_assign(
      normal_array, acquire<DataArray1D *>(current.vertex_normal));
  dirty |= compare_and_assign(
      attribute0_array, acquire<DataArray1D *>(current.vertex_attribute0));
  dirty |= compare_and_assign(
      attribute1_array, acquire<DataArray1D *>(current.vertex_attribute1));
  dirty |= compare_and_assign(
      attribute2_array, acquire<DataArray1D *>(current.vertex_attribute2));
  dirty |= compare_and_assign(
      attribute3_array, acquire<DataArray1D *>(current.vertex_attribute3));

  // vertex arrays take precedence according to spec
  if (!color_array) {
    dirty |= compare_and_assign(
        primitive_color_array, acquire<DataArray1D *>(current.primitive_color));
  }
  if (!attribute0_array) {
    dirty |= compare_and_assign(primitive_attribute0_array,
        acquire<DataArray1D *>(current.primitive_attribute0));
  }
  if (!attribute1_array) {
    dirty |= compare_and_assign(primitive_attribute1_array,
        acquire<DataArray1D *>(current.primitive_attribute1));
  }
  if (!attribute2_array) {
    dirty |= compare_and_assign(primitive_attribute2_array,
        acquire<DataArray1D *>(current.primitive_attribute2));
  }
  if (!attribute3_array) {
    dirty |= compare_and_assign(primitive_attribute3_array,
        acquire<DataArray1D *>(current.primitive_attribute3));
  }

  dirty |= compare_and_assign(
      primitive_id_array, acquire<DataArray1D *>(current.primitive_id));

  dirty |= compare_and_assign(
      index_array, acquire<DataArray1D *>(current.primitive_index));

  if (!position_array) {
    anariReportStatus(device,
        handle,
        ANARI_GEOMETRY,
        ANARI_SEVERITY_ERROR,
        ANARI_STATUS_UNKNOWN_ERROR,
        "Triangle Geometry lacks position array %llu",
        current.vertex_position.getHandle());
  }
}

template <typename G>
void configure_vertex_array(G &gl, ObjectRef<DataArray1D> &array, int index)
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
  }
}

void triangles_init_objects(ObjectRef<GeometryTriangle> triangleObj)
{
  auto &gl = triangleObj->thisDevice->gl;
  if (triangleObj->vao == 0) {
    gl.GenVertexArrays(1, &triangleObj->vao);
  }
  gl.BindVertexArray(triangleObj->vao);
  configure_vertex_array(gl, triangleObj->position_array, 0);
  configure_vertex_array(gl, triangleObj->color_array, 1);
  configure_vertex_array(gl, triangleObj->normal_array, 2);
  configure_vertex_array(gl, triangleObj->attribute0_array, 3);
  configure_vertex_array(gl, triangleObj->attribute1_array, 4);
  configure_vertex_array(gl, triangleObj->attribute2_array, 5);
  configure_vertex_array(gl, triangleObj->attribute3_array, 6);

  if (triangleObj->index_array) {
    ANARIDataType elementType = triangleObj->index_array->getElementType();
    gl.BindBuffer(
        GL_ELEMENT_ARRAY_BUFFER, triangleObj->index_array->getBuffer());
  }
}

void Object<GeometryTriangle>::update()
{
  DefaultObject::update();
  if (!position_array) {
    return;
  }
  if (dirty) {
    thisDevice->queue.enqueue(triangles_init_objects, this);
    dirty = false;
  }
}

std::array<float, 6> Object<GeometryTriangle>::bounds()
{
  if (position_array) {
    return position_array->getBounds();
  } else {
    return std::array<float, 6>{0, 0, 0, 0, 0, 0};
  }
}

#define PRIMITIVE_COLOR_ARRAY GEOMETRY_RESOURCE(0)
#define PRIMITIVE_ATTRIBUTE0_ARRAY GEOMETRY_RESOURCE(1)
#define PRIMITIVE_ATTRIBUTE1_ARRAY GEOMETRY_RESOURCE(2)
#define PRIMITIVE_ATTRIBUTE2_ARRAY GEOMETRY_RESOURCE(3)
#define PRIMITIVE_ATTRIBUTE3_ARRAY GEOMETRY_RESOURCE(4)

void Object<GeometryTriangle>::allocateResources(SurfaceObjectBase *surf)
{
  if (primitive_color_array) {
    surf->allocateStorageBuffer(
        PRIMITIVE_COLOR_ARRAY, primitive_color_array->getBuffer());
  }
  if (primitive_attribute0_array) {
    surf->allocateStorageBuffer(
        PRIMITIVE_ATTRIBUTE0_ARRAY, primitive_attribute0_array->getBuffer());
  }
  if (primitive_attribute1_array) {
    surf->allocateStorageBuffer(
        PRIMITIVE_ATTRIBUTE1_ARRAY, primitive_attribute1_array->getBuffer());
  }
  if (primitive_attribute2_array) {
    surf->allocateStorageBuffer(
        PRIMITIVE_ATTRIBUTE2_ARRAY, primitive_attribute2_array->getBuffer());
  }
  if (primitive_attribute3_array) {
    surf->allocateStorageBuffer(
        PRIMITIVE_ATTRIBUTE3_ARRAY, primitive_attribute3_array->getBuffer());
  }
}

void Object<GeometryTriangle>::drawCommand(
    SurfaceObjectBase *surf, DrawCommand &command)
{
  update();
  if (!position_array) {
    return;
  }
  command.vao = vao;
  command.occlusion_resolve_vao = vao;
  command.prim = GL_TRIANGLES;

  command.vertex_count = position_array->size();

  if (index_array) {
    command.count = 3 * index_array->size();
    command.indexType = GL_UNSIGNED_INT;
  } else {
    command.count = 3 * position_array->size();
  }

  command.instanceCount = 1;

  if (primitive_color_array) {
    int index = surf->resourceIndex(PRIMITIVE_COLOR_ARRAY);
    primitive_color_array->drawCommand(index, command);
  }
  if (primitive_attribute0_array) {
    int index = surf->resourceIndex(PRIMITIVE_ATTRIBUTE0_ARRAY);
    primitive_attribute0_array->drawCommand(index, command);
  }
  if (primitive_attribute1_array) {
    int index = surf->resourceIndex(PRIMITIVE_ATTRIBUTE1_ARRAY);
    primitive_attribute1_array->drawCommand(index, command);
  }
  if (primitive_attribute2_array) {
    int index = surf->resourceIndex(PRIMITIVE_ATTRIBUTE2_ARRAY);
    primitive_attribute2_array->drawCommand(index, command);
  }
  if (primitive_attribute3_array) {
    int index = surf->resourceIndex(PRIMITIVE_ATTRIBUTE3_ARRAY);
    primitive_attribute3_array->drawCommand(index, command);
  }
}

void Object<GeometryTriangle>::interfaceBlock(
    SurfaceObjectBase *surf, AppendableShader &shader)
{
  shader.append("Data {\n");
  if ((surf->getAttributeFlags(ATTRIBUTE_WORLD_POSITION)
          & ATTRIBUTE_FLAG_SAMPLED)
      || (surf->getAttributeFlags(ATTRIBUTE_OBJECT_POSITION)
          & ATTRIBUTE_FLAG_SAMPLED)) {
    shader.append("  centroid vec4 vertexPosition;\n");
  } else {
    shader.append("  vec4 vertexPosition;\n");
  }

  if (surf->getAttributeFlags(ATTRIBUTE_COLOR) & ATTRIBUTE_FLAG_SAMPLED) {
    shader.append("  centroid vec4 vertexColor;\n");
  } else {
    shader.append("  vec4 vertexColor;\n");
  }

  if ((surf->getAttributeFlags(ATTRIBUTE_WORLD_NORMAL) & ATTRIBUTE_FLAG_SAMPLED)
      || (surf->getAttributeFlags(ATTRIBUTE_OBJECT_NORMAL)
          & ATTRIBUTE_FLAG_SAMPLED)) {
    shader.append("  centroid vec3 vertexNormal;\n");
  } else {
    shader.append("  vec3 vertexNormal;\n");
  }

  shader.append("  float vertexOcclusion;\n");
  if (surf->getAttributeFlags(ATTRIBUTE_ATTRIBUTE0) & ATTRIBUTE_FLAG_SAMPLED) {
    shader.append("  centroid vec4 vertexAttribute0;\n");
  } else {
    shader.append("  vec4 vertexAttribute0;\n");
  }
  if (surf->getAttributeFlags(ATTRIBUTE_ATTRIBUTE1) & ATTRIBUTE_FLAG_SAMPLED) {
    shader.append("  centroid vec4 vertexAttribute1;\n");
  } else {
    shader.append("  vec4 vertexAttribute1;\n");
  }
  if (surf->getAttributeFlags(ATTRIBUTE_ATTRIBUTE2) & ATTRIBUTE_FLAG_SAMPLED) {
    shader.append("  centroid vec4 vertexAttribute2;\n");
  } else {
    shader.append("  vec4 vertexAttribute2;\n");
  }
  if (surf->getAttributeFlags(ATTRIBUTE_ATTRIBUTE3) & ATTRIBUTE_FLAG_SAMPLED) {
    shader.append("  centroid vec4 vertexAttribute3;\n");
  } else {
    shader.append("  vec4 vertexAttribute3;\n");
  }
  shader.append("};\n");
}

void Object<GeometryTriangle>::vertexShader(
    SurfaceObjectBase *surf, AppendableShader &shader)
{
  shader.append(occlusion_declaration);

  shader.append("out ");
  interfaceBlock(surf, shader);

  shader.append(triangle_vert);
}

void Object<GeometryTriangle>::fragmentShaderMain(
    SurfaceObjectBase *surf, AppendableShader &shader)
{
  if (primitive_color_array) {
    int index = surf->resourceIndex(PRIMITIVE_COLOR_ARRAY);
    primitive_color_array->declare(index, shader);
  }
  if (primitive_attribute0_array) {
    int index = surf->resourceIndex(PRIMITIVE_ATTRIBUTE0_ARRAY);
    primitive_attribute0_array->declare(index, shader);
  }
  if (primitive_attribute1_array) {
    int index = surf->resourceIndex(PRIMITIVE_ATTRIBUTE1_ARRAY);
    primitive_attribute1_array->declare(index, shader);
  }
  if (primitive_attribute2_array) {
    int index = surf->resourceIndex(PRIMITIVE_ATTRIBUTE2_ARRAY);
    primitive_attribute2_array->declare(index, shader);
  }
  if (primitive_attribute3_array) {
    int index = surf->resourceIndex(PRIMITIVE_ATTRIBUTE3_ARRAY);
    primitive_attribute3_array->declare(index, shader);
  }

  shader.append("in \n");
  interfaceBlock(surf, shader);

  shader.append(triangle_frag);

  if (primitive_color_array) {
    int index = surf->resourceIndex(PRIMITIVE_COLOR_ARRAY);
    shader.append("  color = ");
    primitive_color_array->sample(index, shader);
    shader.append(gl_primitive_id);
  }
  if (primitive_attribute0_array) {
    int index = surf->resourceIndex(PRIMITIVE_ATTRIBUTE0_ARRAY);
    shader.append("  attribute0 = ");
    primitive_attribute0_array->sample(index, shader);
    shader.append(gl_primitive_id);
  }
  if (primitive_attribute1_array) {
    int index = surf->resourceIndex(PRIMITIVE_ATTRIBUTE1_ARRAY);
    shader.append("  attribute1 = ");
    primitive_attribute1_array->sample(index, shader);
    shader.append(gl_primitive_id);
  }
  if (primitive_attribute2_array) {
    int index = surf->resourceIndex(PRIMITIVE_ATTRIBUTE2_ARRAY);
    shader.append("  attribute2 = ");
    primitive_attribute2_array->sample(index, shader);
    shader.append(gl_primitive_id);
  }
  if (primitive_attribute3_array) {
    int index = surf->resourceIndex(PRIMITIVE_ATTRIBUTE3_ARRAY);
    shader.append("  attribute3 = ");
    primitive_attribute3_array->sample(index, shader);
    shader.append(gl_primitive_id);
  }
}

void Object<GeometryTriangle>::vertexShaderShadow(
    SurfaceObjectBase *surf, AppendableShader &shader)
{
  shader.append(triangle_vert_shadow);
}

void Object<GeometryTriangle>::geometryShaderShadow(
    SurfaceObjectBase *surf, AppendableShader &shader)
{
  shader.append(shadow_block_declaration);
  shader.append(triangle_geom_shadow);
}

void Object<GeometryTriangle>::fragmentShaderShadowMain(
    SurfaceObjectBase *surf, AppendableShader &shader)
{
  shader.append(empty_fragment_shader);
}

void Object<GeometryTriangle>::vertexShaderOcclusion(
    SurfaceObjectBase *surf, AppendableShader &shader)
{
  shader.append(shader_conversions);
  shader.append(occlusion_declaration);
  shader.append(shadow_map_declaration);
  shader.append(triangle_vert_occlusion_resolve);
}

static void triangle_delete_objects(Object<Device> *deviceObj, GLuint vao)
{
  auto &gl = deviceObj->gl;
  gl.DeleteVertexArrays(1, &vao);
}

Object<GeometryTriangle>::~Object()
{
  if (vao) {
    thisDevice->queue.enqueue(triangle_delete_objects, thisDevice, vao);
  }
}

} // namespace visgl
