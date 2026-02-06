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
#include <cmath>

namespace visgl {

static void generate_cylinder(float *vertices,
    uint32_t *indices,
    uint32_t sides,
    uint32_t *vertex_count_out,
    uint32_t *index_count_out,
    uint32_t *body_index_count_out)
{
  uint32_t vertex_count = 0;
  for (uint32_t i = 0; i < sides; ++i) {
    float a = 6.28318530718f * (float)i / (float)sides;
    if (vertices) {
      vertices[3 * vertex_count + 0] = std::cos(a);
      vertices[3 * vertex_count + 1] = std::sin(a);
      vertices[3 * vertex_count + 2] = 0.0f;
    }
    vertex_count += 1;

    if (vertices) {
      vertices[3 * vertex_count + 0] = std::cos(a);
      vertices[3 * vertex_count + 1] = std::sin(a);
      vertices[3 * vertex_count + 2] = 1.0f;
    }
    vertex_count += 1;
  }

  if (vertices) {
    vertices[3 * vertex_count + 0] = 0.0f;
    vertices[3 * vertex_count + 1] = 0.0f;
    vertices[3 * vertex_count + 2] = 0.0f;
  }
  vertex_count += 1;

  if (vertices) {
    vertices[3 * vertex_count + 0] = 0.0f;
    vertices[3 * vertex_count + 1] = 0.0f;
    vertices[3 * vertex_count + 2] = 1.0f;
  }
  vertex_count += 1;

  uint32_t index_count = 0;

  // body
  for (uint32_t i = 0; i < sides; ++i) {
    if (indices) {
      indices[index_count + 0] = 2 * i + 1;
      indices[index_count + 1] = 2 * i + 0;
      indices[index_count + 2] = 2 * ((i + 1) % sides) + 0;
    }
    index_count += 3;

    if (indices) {
      indices[index_count + 0] = 2 * i + 1;
      indices[index_count + 1] = 2 * ((i + 1) % sides) + 0;
      indices[index_count + 2] = 2 * ((i + 1) % sides) + 1;
    }
    index_count += 3;
  }
  uint32_t body_index_count = index_count;

  // cap
  for (uint32_t i = 0; i < sides; ++i) {
    if (indices) {
      indices[index_count + 0] = 2 * sides + 0;
      indices[index_count + 1] = 2 * ((i + 1) % sides) + 0;
      indices[index_count + 2] = 2 * i + 0;
    }
    index_count += 3;
  }

  // cap
  for (uint32_t i = 0; i < sides; ++i) {
    if (indices) {
      indices[index_count + 0] = 2 * sides + 1;
      indices[index_count + 1] = 2 * i + 1;
      indices[index_count + 2] = 2 * ((i + 1) % sides) + 1;
    }
    index_count += 3;
  }

  if (vertex_count_out) {
    *vertex_count_out = vertex_count;
  }
  if (index_count_out) {
    *index_count_out = index_count;
  }
  if (body_index_count_out) {
    *body_index_count_out = body_index_count;
  }
}

const char *cyl_vert = R"GLSL(
layout(location = 0) in vec3 in_position;

out Data {
  vec4 vertexPosition;
  float vertexU;
  flat vec3 v1;
  flat vec3 v2;
  flat uvec2 vertexId;
  flat uint primitiveId;
  flat float r;
};

void main() {
  mat4 transform = transforms[instanceIndices.x];
  mat4 projection = transforms[cameraIdx];

  primitiveId = uint(gl_InstanceID);
  vertexId = get_vertices(primitiveId);

  vec3 p1 = get_position(vertexId.x).xyz;
  vec3 p2 = get_position(vertexId.y).xyz;
  r = get_radius(primitiveId);

  vec3 axis = p2 - p1;
  vec3 absaxis = abs(axis);
  float m = min(absaxis.x, min(absaxis.y, absaxis.z));
  bvec3 bmin = equal(absaxis, vec3(m));
  vec3 other = all(bmin) ? vec3(1, 0, 0) : vec3(bmin);

  vec3 normaxis = normalize(axis);
  vec3 other1 = normalize(other - normaxis*dot(normaxis, other));
  vec3 other2 = cross(normaxis, other1);

  vec3 position = 1.155*r*(in_position.x*other1 + in_position.y*other2) + mix(p1, p2, in_position.z);

  vertexPosition = transform*vec4(position, 1.0);
  vertexU = in_position.z;
  v1 = p1;
  v2 = p2;
  gl_Position = projection*vertexPosition;
}
)GLSL";

const char *cyl_frag = R"GLSL(
in Data {
  vec4 vertexPosition;
  float vertexU;
  flat vec3 v1;
  flat vec3 v2;
  flat uvec2 vertexId;
  flat uint primitiveId;
  flat float r;
};

layout(location = 0) out vec4 FragColor;


bool intersect_cylinder(vec3 dir, vec3 origin, vec3 v1, vec3 v2, float radius, bool caps, out float x, out float u, out vec3 normal) {
  vec3 axis = normalize(v2 - v1);

  vec3 diff = origin - v1;

  // project end points on axis
  float p1 = (dot(axis, v1) - dot(axis, origin)) / (dot(axis, dir));
  float p2 = (dot(axis, v2) - dot(axis, origin)) / (dot(axis, dir));
  float p_near = min(p1, p2);
  float p_far = max(p1, p2);

  vec3 dir2 = dir - axis*dot(axis, dir);
  vec3 diff2 = diff - axis*dot(axis, diff);

  float c = dot(diff2, diff2) - r*r;
  float b = 2.0*dot(diff2, dir2);
  float a = dot(dir2, dir2);

  float discr = b*b-4.0*a*c;

  float x_near = 0.0;
  float x_far = 0.0;
  if(discr>=0.0) {
    x_near = (-b-sqrt(discr))/(2.0*a);
    x_far = (-b+sqrt(discr))/(2.0*a);
    if(x_near > p_far || x_far < p_near) {
      // hits the cylinder but not between the vertices
      return false;
    }
  } else {
    // entirely misses cylinder
    return false;
  }

  // x_near >= 0 indicates an outside hit
  x = x_near >= 0.0 ? x_near : x_far;

  // if the front plane is hit between the cylinder intercepts
  // we hit the cap
  bool hit_cap = x_near < p_near && p_near < x_far;
  if(hit_cap && caps) {
    // hit cap
    x = p_near;
  } else if(x > p_far) {
    //miss by tunnelling through
    return false;
  }

  float p = dot(axis, origin) + x*dot(axis, dir);
  u = (p - dot(axis, v1))/ dot(axis, v2 - v1);

  vec3 center = mix(v1, v2, u);

  normal = normalize(origin - center + x*dir);
  if(hit_cap && caps) {
    normal = p_far == p1 ? axis : -axis;
  } else if(x_near < 0.0) {
    normal = -normal;
  }
  return true;
}



void main() {
  float coverage = 1.0;

  float u = vertexU;

  mat4 transform = transforms[instanceIndices.x];
  mat4 inverseTransform = transforms[instanceIndices.x+1u];
  mat4 projection = transforms[cameraIdx];

  vec4 worldPosition = vertexPosition;

  vec4 direction = transforms[cameraIdx+6u][1];
  direction = vec4(vertexPosition.xyz*direction.w - direction.xyz, 0);

  bool caps = materials[instanceIndices.z].y != 0.0;

  vec3 dir = (inverseTransform*direction).xyz;
  vec3 origin = (inverseTransform*vertexPosition).xyz;
  vec3 diff = origin - v1;

  float x = 0.0;
  vec3 normal;
  bool hit = intersect_cylinder(dir, origin, v1, v2, r, caps, x, u, normal);

  if(!hit) {
    discard;
  }

  vec4 objectPosition = vec4(origin + x*dir, 1.0);
  vec4 objectNormal = vec4(normal, 0.0);

  worldPosition = transform*objectPosition;

  vec4 projected = projection*worldPosition;
  gl_FragDepth = projected.z/projected.w*0.5 + 0.5;


  mat4 normalTransform = transforms[instanceIndices.x+2u];
  vec4 worldNormal = normalTransform*objectNormal;
  vec3 geometryNormal = worldNormal.xyz;

  float fragmentOcclusion = 1.0;

  if(occlusionMode != 0u) {
    uint occlusion_offset = instanceIndices.w + 8u*primitiveId;
    vec4 occ1 = vec4(
      occlusion[occlusion_offset + 0u],
      occlusion[occlusion_offset + 1u],
      occlusion[occlusion_offset + 2u],
      occlusion[occlusion_offset + 3u]
    );

    vec4 occ2 = vec4(
      occlusion[occlusion_offset + 4u],
      occlusion[occlusion_offset + 5u],
      occlusion[occlusion_offset + 6u],
      occlusion[occlusion_offset + 7u]
    );

    vec3 directionality1 = occ1.xyz/occ1.w;
    float fragmentOcclusion1 = occ1.w*(1.0 - dot(worldNormal.xyz, directionality1));
    fragmentOcclusion1 = max(0.0, 2.0*fragmentOcclusion1);

    vec3 directionality2 = occ2.xyz/occ2.w;
    float fragmentOcclusion2 = occ2.w*(1.0 - dot(worldNormal.xyz, directionality2));
    fragmentOcclusion2 = max(0.0, 2.0*fragmentOcclusion2);

    fragmentOcclusion = mix(fragmentOcclusion1, fragmentOcclusion2, u);
  }

  vec4 color = vec4(1.0, 0.0, 0.0, 1.0);
  vec4 attribute0 = vec4(0.0, 0.0, 0.0, 1.0);
  vec4 attribute1 = vec4(0.0, 0.0, 0.0, 1.0);
  vec4 attribute2 = vec4(0.0, 0.0, 0.0, 1.0);
  vec4 attribute3 = vec4(0.0, 0.0, 0.0, 1.0);
)GLSL";

const char *cyl_vert_shadow = R"GLSL(
layout(location = 0) in vec3 in_position;
layout(std430, binding = 4) buffer CylinderPositionBlock {
  float position[];
};

out Data {
  vec4 vertexPosition;
  flat vec3 v1;
  flat vec3 v2;
  flat float r;
};

void main() {
  mat4 transform = transforms[instanceIndices.x];

  uint primitiveId = uint(gl_InstanceID);
  uvec2 vertexId = get_vertices(primitiveId);

  vec3 p1 = vec3(
    position[3u*vertexId.x],
    position[3u*vertexId.x+1u],
    position[3u*vertexId.x+2u]
  );
  vec3 p2 = vec3(
    position[3u*vertexId.y],
    position[3u*vertexId.y+1u],
    position[3u*vertexId.y+2u]
  );
  r = get_radius(primitiveId);

  vec3 axis = p2 - p1;
  vec3 absaxis = abs(axis);
  float m = min(absaxis.x, min(absaxis.y, absaxis.z));
  bvec3 bmin = equal(absaxis, vec3(m));
  vec3 other = all(bmin) ? vec3(1, 0, 0) : vec3(bmin);

  vec3 normaxis = normalize(axis);
  vec3 other1 = normalize(other - normaxis*dot(normaxis, other));
  vec3 other2 = cross(normaxis, other1);

  vec3 position = 1.155*r*(in_position.x*other1 + in_position.y*other2) + mix(p1, p2, in_position.z);

  vertexPosition = transform*vec4(position, 1.0);
  v1 = p1;
  v2 = p2;
  gl_Position = vertexPosition;
}
)GLSL";

const char *cyl_geom_shadow = R"GLSL(
layout(triangles) in;
layout(triangle_strip, max_vertices=36) out;

in Data {
  vec4 vertexPosition;
  flat vec3 v1;
  flat vec3 v2;
  flat float r;
} vertex_in[];

out Data {
  vec4 vertexPosition;
  flat vec3 v1;
  flat vec3 v2;
  flat float r;
} vertex_out;

void main() {

  vec4 v1 = gl_in[0].gl_Position;
  vec4 v2 = gl_in[1].gl_Position;
  vec4 v3 = gl_in[2].gl_Position;

  for(int i = 0;i<int(meta.y);++i) {

    gl_Position = shadowProjection[i].matrix*v1;
    vertex_out.vertexPosition = vertex_in[0].vertexPosition;
    vertex_out.v1 = vertex_in[0].v1;
    vertex_out.v2 = vertex_in[0].v2;
    vertex_out.r = vertex_in[0].r;
    gl_Layer = i;
    EmitVertex();

    gl_Position = shadowProjection[i].matrix*v2;
    vertex_out.vertexPosition = vertex_in[1].vertexPosition;
    vertex_out.v1 = vertex_in[1].v1;
    vertex_out.v2 = vertex_in[1].v2;
    vertex_out.r = vertex_in[1].r;
    gl_Layer = i;
    EmitVertex();

    gl_Position = shadowProjection[i].matrix*v3;
    vertex_out.vertexPosition = vertex_in[2].vertexPosition;
    vertex_out.v1 = vertex_in[2].v1;
    vertex_out.v2 = vertex_in[2].v2;
    vertex_out.r = vertex_in[2].r;
    gl_Layer = i;
    EmitVertex();

    EndPrimitive();
  }
}
)GLSL";

const char *cyl_frag_shadow = R"GLSL(
in Data {
  vec4 vertexPosition;
  flat vec3 v1;
  flat vec3 v2;
  flat float r;
};

layout(location = 0) out vec4 FragColor;

bool intersect_cylinder(vec3 dir, vec3 origin, vec3 v1, vec3 v2, float radius, bool caps, out float x, out float u, out vec3 normal) {
  vec3 axis = normalize(v2 - v1);

  vec3 diff = origin - v1;

  // project end points on axis
  float p1 = (dot(axis, v1) - dot(axis, origin)) / (dot(axis, dir));
  float p2 = (dot(axis, v2) - dot(axis, origin)) / (dot(axis, dir));
  float p_near = min(p1, p2);
  float p_far = max(p1, p2);

  vec3 dir2 = dir - axis*dot(axis, dir);
  vec3 diff2 = diff - axis*dot(axis, diff);

  float c = dot(diff2, diff2) - r*r;
  float b = 2.0*dot(diff2, dir2);
  float a = dot(dir2, dir2);

  float discr = b*b-4.0*a*c;

  float x_near = 0.0;
  float x_far = 0.0;
  if(discr>=0.0) {
    x_near = (-b-sqrt(discr))/(2.0*a);
    x_far = (-b+sqrt(discr))/(2.0*a);
    if(x_near > p_far || x_far < p_near) {
      // hits the cylinder but not between the vertices
      return false;
    }
  } else {
    // entirely misses cylinder
    return false;
  }

  // x_near >= 0 indicates an outside hit
  x = x_near >= 0.0 ? x_near : x_far;

  // if the front plane is hit between the cylinder intercepts
  // we hit the cap
  bool hit_cap = x_near < p_near && p_near < x_far;
  if(hit_cap && caps) {
    // hit cap
    x = p_near;
  } else if(x > p_far) {
    //miss by tunnelling through
    return false;
  }

  float p = dot(axis, origin) + x*dot(axis, dir);
  u = (p - dot(axis, v1))/ dot(axis, v2 - v1);

  vec3 center = mix(v1, v2, u);

  normal = normalize(origin - center + x*dir);
  if(hit_cap && caps) {
    normal = p_far == p1 ? axis : -axis;
  } else if(x_near < 0.0) {
    normal = -normal;
  }
  return true;
}

void main() {
  mat4 projection = shadowProjection[gl_Layer].matrix;
  mat4 transform = transforms[instanceIndices.x];
  mat4 inverseTransform = transforms[instanceIndices.x+1u];

  vec4 worldPosition = vertexPosition;

  vec4 direction = vec4(projection[0][2], projection[1][2], projection[2][2], 0.0);

  vec3 dir = (inverseTransform*direction).xyz;
  vec3 origin = (inverseTransform*vertexPosition).xyz;

  bool caps = materials[instanceIndices.z].y != 0.0;

  float x = 0.0;
  float u;
  vec3 normal;
  bool hit = intersect_cylinder(dir, origin, v1, v2, r, caps, x, u, normal);

  if(!hit) {
    discard;
  }

  vec4 objectPosition = vec4(origin + x*dir, 1);

  worldPosition = transform*objectPosition;

  vec4 projected = projection*worldPosition;
  gl_FragDepth = projected.z/projected.w*0.5 + 0.5;

}
)GLSL";

const char *cyl_vert_occlusion_resolve = R"GLSL(
void main() {
  mat4 transform = transforms[instanceIndices.x];
  mat4 inverseTransform = transforms[instanceIndices.x+1u];

  uint primitiveId = uint(gl_VertexID);
  uvec2 vertexId = get_vertices(primitiveId);

  vec3 p1 = get_position(vertexId.x).xyz;
  vec3 p2 = get_position(vertexId.y).xyz;
  float r = abs(get_radius(primitiveId));

  vec3 axis = p2 - p1;
  vec3 normaxis = normalize(axis);

  vec4 sum1 = vec4(0.0);
  vec4 sum2 = vec4(0.0);
  for(uint i = 0u;i<12u;++i) {
    vec3 dir = sampleShadowDir(i);

    vec3 tdir = (inverseTransform*vec4(dir, 0)).xyz;
    vec3 radial = normalize(tdir - normaxis*dot(tdir, normaxis));

    float shadow1 = sampleShadowBias(transform*vec4(p1, 1.0), 0.001, i);
    float shadow2 = sampleShadowBias(transform*vec4(p2, 1.0), 0.001, i);
    float shadow3 = sampleShadowBias(transform*vec4(mix(p1, p2, 0.25) - r*radial, 1.0), 0.001, i);
    float shadow4 = sampleShadowBias(transform*vec4(mix(p1, p2, 0.75) - r*radial, 1.0), 0.001, i);

    sum1 += max(shadow1, shadow3)*vec4(dir, 1.0);
    sum2 += max(shadow2, shadow4)*vec4(dir, 1.0);
  }

  uint occlusion_offset = instanceIndices.w + 8u*primitiveId;

  for(uint i = 0u;i<4u;++i) {
    float prev = occlusion[occlusion_offset + i];
    occlusion[occlusion_offset + i] = (prev*meta.x + sum1[i])*(1.0/(meta.x + 12.0));
  }
  for(uint i = 0u;i<4u;++i) {
    float prev = occlusion[occlusion_offset + 4u + i];
    occlusion[occlusion_offset + 4u + i] = (prev*meta.x + sum2[i])*(1.0/(meta.x + 12.0));
  }
}
)GLSL";

#define POSITION_ARRAY GEOMETRY_RESOURCE(0)
#define RADIUS_ARRAY GEOMETRY_RESOURCE(1)
#define INDEX_ARRAY GEOMETRY_RESOURCE(2)
#define COLOR_ARRAY GEOMETRY_RESOURCE(3)
#define ATTRIBUTE0_ARRAY GEOMETRY_RESOURCE(4)
#define ATTRIBUTE1_ARRAY GEOMETRY_RESOURCE(5)
#define ATTRIBUTE2_ARRAY GEOMETRY_RESOURCE(6)
#define ATTRIBUTE3_ARRAY GEOMETRY_RESOURCE(7)

Object<GeometryCylinder>::Object(ANARIDevice d, ANARIObject handle)
    : DefaultObject(d, handle)
{
  geometry_index = thisDevice->materials.allocate(1);
}

uint32_t Object<GeometryCylinder>::index()
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

void Object<GeometryCylinder>::commit()
{
  DefaultObject::commit();

  dirty |= compare_and_assign(
      position_array, acquire<DataArray1D *>(current.vertex_position));
  dirty |=
      compare_and_assign(cap_array, acquire<DataArray1D *>(current.vertex_cap));
  dirty |= compare_and_assign(
      color_array, acquire<DataArray1D *>(current.vertex_color));
  dirty |= compare_and_assign(
      attribute0_array, acquire<DataArray1D *>(current.vertex_attribute0));
  dirty |= compare_and_assign(
      attribute1_array, acquire<DataArray1D *>(current.vertex_attribute1));
  dirty |= compare_and_assign(
      attribute2_array, acquire<DataArray1D *>(current.vertex_attribute2));
  dirty |= compare_and_assign(
      attribute3_array, acquire<DataArray1D *>(current.vertex_attribute3));

  dirty |= compare_and_assign(
      primitive_color_array, acquire<DataArray1D *>(current.primitive_color));
  dirty |= compare_and_assign(
      primitive_radius_array, acquire<DataArray1D *>(current.primitive_radius));
  dirty |= compare_and_assign(primitive_attribute0_array,
      acquire<DataArray1D *>(current.primitive_attribute0));
  dirty |= compare_and_assign(primitive_attribute1_array,
      acquire<DataArray1D *>(current.primitive_attribute1));
  dirty |= compare_and_assign(primitive_attribute2_array,
      acquire<DataArray1D *>(current.primitive_attribute2));
  dirty |= compare_and_assign(primitive_attribute3_array,
      acquire<DataArray1D *>(current.primitive_attribute3));

  dirty |= compare_and_assign(
      primitive_id_array, acquire<DataArray1D *>(current.primitive_id));

  dirty |= compare_and_assign(
      index_array, acquire<DataArray1D *>(current.primitive_index));

  float new_radius = 0;
  if (current.radius.get(ANARI_FLOAT32, &new_radius)) {
    dirty |= compare_and_assign(radius, new_radius);
  }

  caps = current.caps.getStringEnum();
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

const char *cyl_vert_radius_value = R"GLSL(
float get_radius(uint i) {
  return materials[instanceIndices.z].x;
}
)GLSL";

const char *cyl_vert_index_implicit = R"GLSL(
uvec2 get_vertices(uint i) {
  return uvec2(2u*i, 2u*i+1u);
}
)GLSL";

const char *cyl_vert_radius_array = R"GLSL(
layout(std430, binding = 5) buffer CylinderRadiusBlock {
  float radius[];
};
float get_radius(uint i) {
  return abs(radius[i]);
}
)GLSL";

const char *cyl_vert_index_array = R"GLSL(
layout(std430, binding = 6) buffer CylinderIndexBlock {
  uvec2 vertex_indices[];
};
uvec2 get_vertices(uint i) {
  return vertex_indices[i];
}
)GLSL";

#define DECLARE_IF(ARRAY, SLOT, SHADER)                                        \
  if (ARRAY) {                                                                 \
    int index = surf->resourceIndex(SLOT);                                     \
    ARRAY->declare(index, SHADER);                                             \
  }

void Object<GeometryCylinder>::declarations(
    SurfaceObjectBase *surf, AppendableShader &shader)
{
  DECLARE_IF(position_array, POSITION_ARRAY, shader)
  if (position_array) {
    int index = surf->resourceIndex(POSITION_ARRAY);
    shader.append("vec4 get_position(uint i) {\n  return ");
    position_array->sample(index, shader);
    shader.append("i);\n}\n");
  } else {
    //???
  }

  DECLARE_IF(primitive_radius_array, RADIUS_ARRAY, shader)
  if (primitive_radius_array) {
    int index = surf->resourceIndex(RADIUS_ARRAY);
    shader.append("float get_radius(uint i) {\n  return ");
    shader.append(ssboArrayName[index]);
    shader.append("[i];\n}\n");
  } else {
    shader.append(cyl_vert_radius_value);
  }

  DECLARE_IF(index_array, INDEX_ARRAY, shader)
  if (index_array) {
    int index = surf->resourceIndex(INDEX_ARRAY);
    shader.append("uvec2 get_vertices(uint i) {\n  return ");
    shader.append(ssboArrayName[index]);
    shader.append("[i];\n}\n");
  } else {
    shader.append(cyl_vert_index_implicit);
  }
}

void cylinder_init_objects(ObjectRef<GeometryCylinder> cylinderObj)
{
  auto &gl = cylinderObj->thisDevice->gl;
  if (cylinderObj->vao == 0) {
    gl.GenVertexArrays(1, &cylinderObj->vao);
    gl.GenVertexArrays(1, &cylinderObj->occlusion_resolve_vao);

    gl.BindVertexArray(cylinderObj->vao);

    uint32_t sides = 6;

    uint32_t vertex_count = 0;
    uint32_t index_count = 0;
    uint32_t body_index_count = 0;
    generate_cylinder(nullptr,
        nullptr,
        sides,
        &vertex_count,
        &index_count,
        &body_index_count);
    std::vector<float> vertices(3 * vertex_count);
    std::vector<uint32_t> indices(index_count);
    generate_cylinder(vertices.data(),
        indices.data(),
        sides,
        &vertex_count,
        &index_count,
        &body_index_count);

    gl.GenBuffers(1, &cylinderObj->cyl_position);
    gl.BindBuffer(GL_ARRAY_BUFFER, cylinderObj->cyl_position);
    gl.BufferData(GL_ARRAY_BUFFER,
        sizeof(float) * vertices.size(),
        vertices.data(),
        GL_STATIC_DRAW);
    gl.EnableVertexAttribArray(0);
    gl.VertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);

    gl.GenBuffers(1, &cylinderObj->cyl_index);
    gl.BindBuffer(GL_ELEMENT_ARRAY_BUFFER, cylinderObj->cyl_index);
    gl.BufferData(GL_ELEMENT_ARRAY_BUFFER,
        sizeof(uint32_t) * indices.size(),
        indices.data(),
        GL_STATIC_DRAW);

    cylinderObj->index_count = index_count;
    cylinderObj->body_index_count = body_index_count;
  }
}

void Object<GeometryCylinder>::update()
{
  DefaultObject::update();

  if (dirty) {
    thisDevice->queue.enqueue(cylinder_init_objects, this);
    std::array<float, 4> data{radius, float(caps != STRING_ENUM_none), 0, 0};
    thisDevice->materials.set(geometry_index, data);
    dirty = false;
  }
}

std::array<float, 6> Object<GeometryCylinder>::bounds()
{
  std::array<float, 6> b{0, 0, 0, 0, 0, 0};
  if (position_array) {
    b = position_array->getBounds();
    if (primitive_radius_array) {
      std::array<float, 6> a = primitive_radius_array->getBounds();

      b[0] += a[0];
      b[1] += a[1];
      b[2] += a[2];
      b[3] += a[3];
      b[4] += a[4];
      b[5] += a[5];
    } else {
      b[0] -= radius;
      b[1] -= radius;
      b[2] -= radius;
      b[3] += radius;
      b[4] += radius;
      b[5] += radius;
    }
  }
  return b;
}

#define ALLOCATE_IF(ARRAY, SLOT)                                               \
  if (ARRAY) {                                                                 \
    surf->allocateStorageBuffer(SLOT, ARRAY->getBuffer());                     \
  }

void Object<GeometryCylinder>::allocateResources(SurfaceObjectBase *surf)
{
  surf->allocateStorageBuffer(POSITION_ARRAY, position_array->getBuffer());

  ALLOCATE_IF(primitive_radius_array, RADIUS_ARRAY)

  ALLOCATE_IF(index_array, INDEX_ARRAY)

  ALLOCATE_IF(color_array, COLOR_ARRAY)
  ALLOCATE_IF(primitive_color_array, COLOR_ARRAY)

  ALLOCATE_IF(attribute0_array, ATTRIBUTE0_ARRAY)
  ALLOCATE_IF(primitive_attribute0_array, ATTRIBUTE0_ARRAY)

  ALLOCATE_IF(attribute1_array, ATTRIBUTE1_ARRAY)
  ALLOCATE_IF(primitive_attribute1_array, ATTRIBUTE1_ARRAY)

  ALLOCATE_IF(attribute3_array, ATTRIBUTE3_ARRAY)
  ALLOCATE_IF(primitive_attribute3_array, ATTRIBUTE3_ARRAY)

  ALLOCATE_IF(attribute3_array, ATTRIBUTE3_ARRAY)
  ALLOCATE_IF(primitive_attribute3_array, ATTRIBUTE3_ARRAY)
}

#define DRAW_COMMAND_IF(ARRAY, SLOT)                                           \
  if (ARRAY) {                                                                 \
    int index = surf->resourceIndex(SLOT);                                     \
    ARRAY->drawCommand(index, command);                                        \
  }

void Object<GeometryCylinder>::drawCommand(
    SurfaceObjectBase *surf, DrawCommand &command)
{
  // command.shadow_shader = shadow_shader;
  command.vao = vao;
  command.prim = GL_TRIANGLES;
  command.count =
      index_count; // caps == STRING_ENUM_none ? body_index_count : index_count;
  if (index_array) {
    command.instanceCount = index_array->size();
  } else {
    command.instanceCount = position_array->size() / 2;
  }
  command.indexType = GL_UNSIGNED_INT;
  command.cullMode = GL_BACK;

  command.occlusion_resolve_vao = occlusion_resolve_vao;
  command.vertex_count = 8 * command.instanceCount;

  DRAW_COMMAND_IF(position_array, POSITION_ARRAY)

  DRAW_COMMAND_IF(primitive_radius_array, RADIUS_ARRAY)

  DRAW_COMMAND_IF(index_array, INDEX_ARRAY)

  DRAW_COMMAND_IF(color_array, COLOR_ARRAY)
  else DRAW_COMMAND_IF(primitive_color_array, COLOR_ARRAY)

      DRAW_COMMAND_IF(attribute0_array, ATTRIBUTE0_ARRAY) else DRAW_COMMAND_IF(
          primitive_attribute0_array, ATTRIBUTE0_ARRAY)

          DRAW_COMMAND_IF(attribute1_array,
              ATTRIBUTE1_ARRAY) else DRAW_COMMAND_IF(primitive_attribute1_array,
              ATTRIBUTE1_ARRAY)

              DRAW_COMMAND_IF(attribute3_array,
                  ATTRIBUTE3_ARRAY) else DRAW_COMMAND_IF(primitive_attribute3_array,
                  ATTRIBUTE3_ARRAY)

                  DRAW_COMMAND_IF(attribute3_array,
                      ATTRIBUTE3_ARRAY) else DRAW_COMMAND_IF(primitive_attribute3_array,
                      ATTRIBUTE3_ARRAY)
}

void Object<GeometryCylinder>::vertexShader(
    SurfaceObjectBase *surf, AppendableShader &shader)
{
  declarations(surf, shader);
  shader.append(cyl_vert);
}

#define SAMPLE_IF(ARRAY, PRIMITIVE_ARRAY, SLOT, VARIABLE)                      \
  if (ARRAY) {                                                                 \
    int index = surf->resourceIndex(SLOT);                                     \
    shader.append("  " VARIABLE " = mix(");                                    \
    ARRAY->sample(index, shader);                                              \
    shader.append("vertexId.x), ");                                            \
    ARRAY->sample(index, shader);                                              \
    shader.append("vertexId.y), u);\n");                                       \
  } else if (PRIMITIVE_ARRAY) {                                                \
    int index = surf->resourceIndex(SLOT);                                     \
    shader.append("  " VARIABLE " = ");                                        \
    PRIMITIVE_ARRAY->sample(index, shader);                                    \
    shader.append("primitiveId);\n");                                          \
  }

void Object<GeometryCylinder>::fragmentShaderMain(
    SurfaceObjectBase *surf, AppendableShader &shader)
{
  DECLARE_IF(color_array, COLOR_ARRAY, shader)
  else DECLARE_IF(primitive_color_array, COLOR_ARRAY, shader)

      DECLARE_IF(attribute0_array, ATTRIBUTE0_ARRAY, shader) else DECLARE_IF(
          primitive_attribute0_array, ATTRIBUTE0_ARRAY, shader)

          DECLARE_IF(attribute1_array,
              ATTRIBUTE1_ARRAY,
              shader) else DECLARE_IF(primitive_attribute1_array,
              ATTRIBUTE1_ARRAY,
              shader)

              DECLARE_IF(attribute3_array,
                  ATTRIBUTE3_ARRAY,
                  shader) else DECLARE_IF(primitive_attribute3_array,
                  ATTRIBUTE3_ARRAY,
                  shader)

                  DECLARE_IF(attribute3_array,
                      ATTRIBUTE3_ARRAY,
                      shader) else DECLARE_IF(primitive_attribute3_array,
                      ATTRIBUTE3_ARRAY,
                      shader)

                      shader.append(occlusion_declaration);

  shader.append(cyl_frag);

  SAMPLE_IF(color_array, primitive_color_array, COLOR_ARRAY, "color")
  SAMPLE_IF(attribute0_array,
      primitive_attribute0_array,
      ATTRIBUTE0_ARRAY,
      "attribute0")
  SAMPLE_IF(attribute1_array,
      primitive_attribute1_array,
      ATTRIBUTE1_ARRAY,
      "attribute1")
  SAMPLE_IF(attribute2_array,
      primitive_attribute2_array,
      ATTRIBUTE2_ARRAY,
      "attribute2")
  SAMPLE_IF(attribute3_array,
      primitive_attribute3_array,
      ATTRIBUTE3_ARRAY,
      "attribute3")
}

void Object<GeometryCylinder>::vertexShaderShadow(
    SurfaceObjectBase *surf, AppendableShader &shader)
{
  declarations(surf, shader);
  shader.append(cyl_vert_shadow);
}

void Object<GeometryCylinder>::geometryShaderShadow(
    SurfaceObjectBase *surf, AppendableShader &shader)
{
  shader.append(shadow_block_declaration);
  shader.append(cyl_geom_shadow);
}

void Object<GeometryCylinder>::fragmentShaderShadowMain(
    SurfaceObjectBase *surf, AppendableShader &shader)
{
  shader.append(shadow_block_declaration);
  shader.append(cyl_frag_shadow);
}

void Object<GeometryCylinder>::vertexShaderOcclusion(
    SurfaceObjectBase *surf, AppendableShader &shader)
{
  shader.append(shader_conversions);
  shader.append(occlusion_declaration);
  shader.append(shadow_map_declaration);
  declarations(surf, shader);
  shader.append(cyl_vert_occlusion_resolve);
}

static void sphere_delete_objects(Object<Device> *deviceObj,
    GLuint vao,
    GLuint cyl_position,
    GLuint cyl_index)
{
  auto &gl = deviceObj->gl;
  gl.DeleteVertexArrays(1, &vao);
  gl.DeleteBuffers(1, &cyl_position);
  gl.DeleteBuffers(1, &cyl_index);
};

Object<GeometryCylinder>::~Object()
{
  if (vao) {
    thisDevice->queue.enqueue(
        sphere_delete_objects, thisDevice, vao, cyl_position, cyl_index);
  }
}

} // namespace visgl
