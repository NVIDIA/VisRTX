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


#include "shader_compile_segmented.h"
#include "shader_blocks.h"
#include "math_util.h"

#include <cstdlib>
#include <cstring>

namespace visgl {

Object<Frame>::Object(ANARIDevice d, ANARIObject handle)
    : DefaultObject(d, handle)
{
  camera_index = thisDevice->transforms.allocate(8);
}

void Object<Frame>::commit()
{
  DefaultObject::commit();

  std::array<uint32_t, 2> next_size{16, 16};
  current.size.get(ANARI_UINT32_VEC2, next_size.data());
  if (size[0] < next_size[0] || size[1] < next_size[1]) {
    configuration_changed = true;
  }

  size = next_size;
  uint32_t elements = size[0] * size[1];

  ANARIDataType nextColorType = ANARI_UNKNOWN;
  ANARIDataType nextDepthType = ANARI_UNKNOWN;
  current.channel_color.get(ANARI_DATA_TYPE, &nextColorType);
  current.channel_depth.get(ANARI_DATA_TYPE, &nextDepthType);
  if (nextColorType != colorType) {
    configuration_changed = true;
  }
  if (nextDepthType != depthType) {
    configuration_changed = true;
  }
  colorType = nextColorType;
  depthType = nextDepthType;
}

static GLenum anari2gl(ANARIDataType format)
{
  switch (format) {
  case ANARI_UFIXED8_RGBA_SRGB: return GL_SRGB8_ALPHA8;

  case ANARI_FLOAT32_VEC4: return GL_RGBA32F;

  case ANARI_UFIXED8_VEC4:
  default: return GL_RGBA8;
  }
}

const char *full_screen_vert = R"GLSL(
precision highp float;
precision highp int;

const vec4 tri[3] = vec4[3](
  vec4(-1.0, -1.0, 0.0, 1.0),
  vec4(-1.0, 3.0, 0.0, 1.0),
  vec4(3.0, -1.0, 0.0, 1.0)
);

out vec2 screen_coord;

void main() {
  gl_Position = tri[gl_VertexID];
  screen_coord = gl_Position.xy;
}
)GLSL";

const char *multisample_resolve_frag = R"GLSL(
highp layout(binding = 0) uniform sampler2DMS color;
highp layout(binding = 1) uniform sampler2DMS depth;

in vec2 screen_coord;

layout(location = 0) out vec4 FragColor;
layout(location = 1) out float LinearDepth;

void main() {
  mat4 projection = transforms[cameraIdx];
  mat4 inverse_projection = transforms[cameraIdx+1u];
  vec4 camera_position = transforms[cameraIdx+6u][0];

  ivec2 icoord = ivec2(gl_FragCoord.xy);

  int colorsamples = int(samples);
  FragColor = vec4(0.0);
  float min_depth = 1.0;

  for(int i=0;i<colorsamples;++i) {
    FragColor += texelFetch(color, icoord, i);
    min_depth = min(min_depth, texelFetch(depth, icoord, i).x);
  }
  FragColor *= 1.0/float(colorsamples);

  vec4 position = inverse_projection*vec4(screen_coord, 2.0*min_depth-1.0, 1.0);
  position *= 1.0/position.w;

  LinearDepth = distance(position.xyz, camera_position.xyz);
}
)GLSL";

void frame_allocate_objects(ObjectRef<Frame> frameObj)
{
  auto &gl = frameObj->thisDevice->gl;
  uint32_t width = frameObj->size[0];
  uint32_t height = frameObj->size[1];
  uint32_t elements = width * height;
  uint32_t element_size = anari::sizeOf(frameObj->colorType);
  GLenum format = anari2gl(frameObj->colorType);

  GLint max_samples = 4;
  gl.GetIntegerv(GL_MAX_SAMPLES, &max_samples);
  frameObj->samples = std::min(8, max_samples);

  if(frameObj->resolve_shader == 0) {
    const char *version = gl.VERSION_4_3 ? version_430 : version_320_es;

    const char *resolve_vert[] = {version, full_screen_vert, nullptr};
    const char *resolve_frag[] = {version, shader_preamble, multisample_resolve_frag, nullptr};

    frameObj->resolve_shader = shader_build_graphics_segmented(gl,
      resolve_vert, nullptr, nullptr, nullptr, resolve_frag);
    gl.GenVertexArrays(1, &frameObj->resolve_vao);
  }


  // delete these in case this is a rebuild
  gl.DeleteBuffers(1, &frameObj->colorbuffer);
  gl.DeleteBuffers(1, &frameObj->depthbuffer);
  gl.DeleteTextures(1, &frameObj->colortarget);
  gl.DeleteTextures(1, &frameObj->depthtarget);
  gl.DeleteFramebuffers(1, &frameObj->fbo);

  // setup framebuffer and pack buffers
  gl.GenBuffers(1, &frameObj->colorbuffer);
  gl.BindBuffer(GL_PIXEL_PACK_BUFFER, frameObj->colorbuffer);
  gl.BufferData(
      GL_PIXEL_PACK_BUFFER, elements * element_size, 0, GL_DYNAMIC_READ);

  gl.GenBuffers(1, &frameObj->depthbuffer);
  gl.BindBuffer(GL_PIXEL_PACK_BUFFER, frameObj->depthbuffer);
  gl.BufferData(
      GL_PIXEL_PACK_BUFFER, elements * sizeof(float), 0, GL_DYNAMIC_READ);

  gl.GenTextures(1, &frameObj->colortarget);
  gl.BindTexture(GL_TEXTURE_2D, frameObj->colortarget);
  gl.TexStorage2D(GL_TEXTURE_2D, 1, format, width, height);
  gl.TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  gl.TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  gl.TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  gl.TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

  gl.GenTextures(1, &frameObj->depthtarget);
  gl.BindTexture(GL_TEXTURE_2D, frameObj->depthtarget);
  gl.TexStorage2D(GL_TEXTURE_2D, 1, GL_R32F, width, height);
  gl.TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  gl.TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  gl.TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  gl.TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

  gl.GenFramebuffers(1, &frameObj->fbo);
  gl.BindFramebuffer(GL_FRAMEBUFFER, frameObj->fbo);
  gl.FramebufferTexture(
      GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, frameObj->colortarget, 0);
  gl.FramebufferTexture(
      GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, frameObj->depthtarget, 0);
  GLenum bufs[] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1};
  gl.DrawBuffers(2, bufs);

  // clear to signal color
  gl.ClearColor(1, 0, 1, 1);
  gl.Clear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  // second framebuffer to resolve multisampling
  gl.DeleteTextures(1, &frameObj->multicolortarget);
  gl.DeleteTextures(1, &frameObj->multidepthtarget);
  gl.DeleteFramebuffers(1, &frameObj->multifbo);

  gl.GenTextures(1, &frameObj->multicolortarget);
  gl.BindTexture(GL_TEXTURE_2D_MULTISAMPLE, frameObj->multicolortarget);
  gl.TexStorage2DMultisample(
      GL_TEXTURE_2D_MULTISAMPLE, frameObj->samples, format, width, height, GL_TRUE);

  gl.GenTextures(1, &frameObj->multidepthtarget);
  gl.BindTexture(GL_TEXTURE_2D_MULTISAMPLE, frameObj->multidepthtarget);
  gl.TexStorage2DMultisample(
      GL_TEXTURE_2D_MULTISAMPLE, frameObj->samples, GL_DEPTH_COMPONENT32F, width, height, GL_TRUE);

  gl.GenFramebuffers(1, &frameObj->multifbo);
  gl.BindFramebuffer(GL_FRAMEBUFFER, frameObj->multifbo);
  gl.FramebufferTexture(GL_FRAMEBUFFER,
      GL_COLOR_ATTACHMENT0,
      frameObj->multicolortarget,
      0);
  gl.FramebufferTexture(GL_FRAMEBUFFER,
      GL_DEPTH_ATTACHMENT,
      frameObj->multidepthtarget,
      0);
  gl.DrawBuffers(1, bufs);

  // clear to signal color
  gl.ClearColor(1, 0, 1, 1);
  gl.Clear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  // gl.BindFramebuffer(GL_READ_FRAMEBUFFER, frameObj->multifbo);
  // gl.BindFramebuffer(GL_DRAW_FRAMEBUFFER, frameObj->fbo);
  // gl.BlitFramebuffer(
  //     0, 0, width, height, 0, 0, width, height, GL_COLOR_BUFFER_BIT, GL_LINEAR);

  gl.BindFramebuffer(GL_READ_FRAMEBUFFER, frameObj->fbo);
  gl.BindBuffer(GL_PIXEL_PACK_BUFFER, frameObj->colorbuffer);
  gl.ReadBuffer(GL_COLOR_ATTACHMENT0);
  if(frameObj->colorType == ANARI_FLOAT32_VEC4) {
    gl.ReadPixels(0, 0, width, height, GL_RGBA, GL_FLOAT, 0);
  } else {
    gl.ReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
  }

  if (frameObj->sceneubo == 0) {
    gl.GenBuffers(1, &frameObj->sceneubo);
    gl.BindBuffer(GL_UNIFORM_BUFFER, frameObj->sceneubo);
    gl.BufferData(GL_UNIFORM_BUFFER, sizeof(GLuint) * 1024, 0, GL_STREAM_DRAW);
  }

  if (frameObj->shadowubo == 0) {
    gl.GenBuffers(1, &frameObj->shadowubo);
  }

  if (gl.VERSION_3_3 && frameObj->duration_query == 0) {
    gl.GenQueries(1, &frameObj->duration_query);
  } else if (gl.EXT_disjoint_timer_query) {
    gl.GenQueriesEXT(1, &frameObj->duration_query);
  }
}

void Object<Frame>::update()
{
  DefaultObject::update();

  if (configuration_changed) {
    configuration_changed = false;

    thisDevice->queue.enqueue(frame_allocate_objects, ObjectRef<Frame>(this));
  }
}

void frame_map_color(ObjectRef<Frame> frameObj, uint64_t size, void **ptr)
{
  auto &gl = frameObj->thisDevice->gl;
  gl.BindBuffer(GL_PIXEL_PACK_BUFFER, frameObj->colorbuffer);
  *ptr = gl.MapBufferRange(GL_PIXEL_PACK_BUFFER, 0, size, GL_MAP_READ_BIT);

  if (gl.VERSION_3_3) {
    gl.GetQueryObjectui64v(
        frameObj->duration_query, GL_QUERY_RESULT, &frameObj->duration);
  } else if (gl.EXT_disjoint_timer_query) {
    gl.GetQueryObjectui64vEXT(
        frameObj->duration_query, GL_QUERY_RESULT_EXT, &frameObj->duration);
  }
}

void frame_map_depth(ObjectRef<Frame> frameObj, uint64_t size, void **ptr)
{
  auto &gl = frameObj->thisDevice->gl;
  uint32_t width = frameObj->size[0];
  uint32_t height = frameObj->size[1];

  gl.BindFramebuffer(GL_READ_FRAMEBUFFER, frameObj->fbo);
  gl.BindBuffer(GL_PIXEL_PACK_BUFFER, frameObj->depthbuffer);
  gl.ReadBuffer(GL_COLOR_ATTACHMENT1);
  gl.ReadPixels(0, 0, width, height, GL_DEPTH_COMPONENT, GL_FLOAT, 0);

  *ptr = gl.MapBufferRange(GL_PIXEL_PACK_BUFFER, 0, size, GL_MAP_READ_BIT);
}

void *Object<Frame>::mapFrame(const char *channel,
    uint32_t *width,
    uint32_t *height,
    ANARIDataType *pixelType)
{
  update();

  *width = size[0];
  *height = size[1];
  *pixelType = ANARI_UNKNOWN;

  uint32_t elements = size[0] * size[1];
  uint32_t element_size = anari::sizeOf(colorType);

  if (std::strncmp(channel, "channel.color", 13) == 0) {
    *pixelType = colorType;

    void *ptr = nullptr;
    thisDevice->queue
        .enqueue(frame_map_color,
            ObjectRef<Frame>(this),
            elements * element_size,
            &ptr)
        .wait();

    return ptr;
  } else if (std::strncmp(channel, "channel.depth", 13) == 0) {
    *pixelType = depthType;

    void *ptr = nullptr;
    thisDevice->queue
        .enqueue(frame_map_depth,
            ObjectRef<Frame>(this),
            elements * sizeof(float),
            &ptr)
        .wait();

    return ptr;
  } else {
    *width = *height = 0;
    return nullptr;
  }
}

void frame_unmap_color(ObjectRef<Frame> frameObj)
{
  auto &gl = frameObj->thisDevice->gl;
  gl.BindBuffer(GL_PIXEL_PACK_BUFFER, frameObj->colorbuffer);
  gl.UnmapBuffer(GL_PIXEL_PACK_BUFFER);
}

void frame_unmap_depth(ObjectRef<Frame> frameObj)
{
  auto &gl = frameObj->thisDevice->gl;
  gl.BindBuffer(GL_PIXEL_PACK_BUFFER, frameObj->depthbuffer);
  gl.UnmapBuffer(GL_PIXEL_PACK_BUFFER);
}

void Object<Frame>::unmapFrame(const char *channel)
{
  if (std::strncmp(channel, "channel.color", 13) == 0) {
    thisDevice->queue.enqueue(frame_unmap_color, ObjectRef<Frame>(this));
  } else if (std::strncmp(channel, "channel.depth", 13) == 0) {
    thisDevice->queue.enqueue(frame_unmap_depth, ObjectRef<Frame>(this));
  }
}

class CollectScene : public ObjectVisitorBase
{
  InstanceObjectBase *instance = 0;
  GeometryObjectBase *geometry = 0;
  MaterialObjectBase *material = 0;
  SurfaceObjectBase *surface = 0;
  SpatialFieldObjectBase *field = 0;
  VolumeObjectBase *volume = 0;

 public:
  uint64_t epoch = 0;
  uint64_t light_epoch = 0;
  uint64_t geometry_epoch = 0;

  float directions[12][3];
  int shadow_caster_count = 0;

  uint32_t vertex_count = 0;

  std::array<float, 6> world_bounds{
      FLT_MAX, FLT_MAX, FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX};

  std::vector<GLuint> lights;
  std::vector<DrawCommand> draws;

  void visit(InstanceObjectBase *obj) override
  {
    geometry_epoch = epoch = std::max(epoch, obj->objectEpoch());
    obj->update();
    instance = obj;
    obj->traverse(this);
    instance = 0;
  }

  void visit(GeometryObjectBase *obj) override
  {
    geometry_epoch = epoch = std::max(epoch, obj->objectEpoch());
    obj->update();
    geometry = obj;
  }

  void visit(MaterialObjectBase *obj) override
  {
    epoch = std::max(epoch, obj->objectEpoch());
    obj->update();
    material = obj;
  }

  void visit(SurfaceObjectBase *obj) override
  {
    epoch = std::max(epoch, obj->objectEpoch());
    surface = obj;
    surface->traverse(this);
    surface->update();
    if (material && geometry) {
      DrawCommand c{};
      geometry->drawCommand(surface, c);
      material->drawCommand(surface, c);
      surface->drawCommand(c);
      c.uniform[0] = instance ? instance->index() : 0;
      c.uniform[1] = material->index();
      c.uniform[2] = geometry->index();
      c.uniform[3] = vertex_count;
      vertex_count += c.vertex_count;
      draws.push_back(c);

      auto bounds = geometry->bounds();
      if (instance) {
        transformBoundingBox(
            bounds.data(), instance->transform().data(), bounds.data());
      }
      foldBoundingBox(world_bounds.data(), bounds.data());
    }

    geometry = 0;
    material = 0;
    surface = 0;
  }

  void visit(SpatialFieldObjectBase *obj) override
  {
    epoch = std::max(epoch, obj->objectEpoch());
    obj->update();
    field = obj;
  }

  void visit(VolumeObjectBase *obj) override
  {
    epoch = std::max(epoch, obj->objectEpoch());
    volume = obj;
    volume->traverse(this);
    volume->update();
    if (field) {
      DrawCommand c{};
      field->drawCommand(volume, c);
      volume->drawCommand(c);
      c.uniform[0] = instance ? instance->index() : 0;
      c.uniform[1] = volume->index();
      c.uniform[2] = field->index();
      c.uniform[3] = vertex_count;
      vertex_count += c.vertex_count;
      draws.push_back(c);

      auto bounds = field->bounds();
      if (instance) {
        transformBoundingBox(
            bounds.data(), instance->transform().data(), bounds.data());
      }
      foldBoundingBox(world_bounds.data(), bounds.data());
    }

    field = 0;
    volume = 0;
  }

  void visit(LightObjectBase *obj) override
  {
    light_epoch = epoch = std::max(epoch, obj->objectEpoch());
    obj->update();
    int current = lights.size();
    lights.push_back(obj->index());
    lights.push_back(instance ? instance->index() : 0);
    lights.push_back(0xFFFFFFFFu); // shadow map index
    lights.push_back(obj->lightType()); // type

    if (shadow_caster_count < 12
        && is_convertible<Object<LightDirectional>>::check(obj)) {
      Object<LightDirectional> *directional =
          static_cast<Object<LightDirectional> *>(obj);
      directional->current.direction.get(
          ANARI_FLOAT32_VEC3, directions[shadow_caster_count]);
      lights[current + 2] = shadow_caster_count;
      shadow_caster_count += 1;
    }
  }

  void visit(ObjectBase *obj) override
  {
    epoch = std::max(epoch, obj->objectEpoch());
    obj->update();
    obj->traverse(this);
  }

  void reset()
  {
    instance = 0;
    geometry = 0;
    material = 0;
    field = 0;
    volume = 0;

    epoch = 0;

    shadow_caster_count = 0;

    vertex_count = 0;

    lights.clear();
    draws.clear();
    world_bounds = std::array<float, 6>{
        FLT_MAX, FLT_MAX, FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX};
  }
};

int absmin_index(const float *x)
{
  if (fabs(x[0]) < fabs(x[1])) {
    if (fabs(x[0]) < fabs(x[2])) {
      return 0;
    } else {
      return 2;
    }
  } else {
    if (fabs(x[1]) < fabs(x[2])) {
      return 1;
    } else {
      return 2;
    }
  }
}

std::array<float, 20> bounds_projection(const float *dir, const float *bounds, int size)
{
  std::array<float, 3> unitdir{dir[0], dir[1], dir[2]};
  std::array<float, 3> ortho1{0.0f, 0.0f, 0.0f};
  std::array<float, 3> ortho2{0.0f, 0.0f, 0.0f};
  normalize3(unitdir.data());
  ortho1[absmin_index(dir)] = 1.0f;

  float s = dot3(unitdir.data(), ortho1.data());
  ortho1[0] -= s * unitdir[0];
  ortho1[1] -= s * unitdir[1];
  ortho1[2] -= s * unitdir[2];
  normalize3(ortho1.data());
  cross(ortho2.data(), unitdir.data(), ortho1.data());

  float near = FLT_MAX;
  float far = -FLT_MAX;
  float bottom = FLT_MAX;
  float top = -FLT_MAX;
  float left = FLT_MAX;
  float right = -FLT_MAX;

  for (int i = 0; i < 8; ++i) {
    float corner[3];

    // enumerate the corners of the bounding box
    corner[0] = (i & 1) ? bounds[0] : bounds[3];
    corner[1] = (i & 2) ? bounds[1] : bounds[4];
    corner[2] = (i & 4) ? bounds[2] : bounds[5];

    float x = dot3(corner, ortho2.data());
    float y = dot3(corner, ortho1.data());
    float z = dot3(corner, unitdir.data());

    left = fast_minf(left, x);
    bottom = fast_minf(bottom, y);
    near = fast_minf(near, z);

    right = fast_maxf(right, x);
    top = fast_maxf(top, y);
    far = fast_maxf(far, z);
  }

  float x_scale = 1.0f / (right - left);
  float x_offset = -(right + left) * x_scale;

  float y_scale = 1.0f / (top - bottom);
  float y_offset = -(top + bottom) * y_scale;

  float z_scale = 1.0f / (far - near);
  float z_offset = -(far + near) * z_scale;

  std::array<float, 20> projection;
  projection[0] = 2.0f * ortho2[0] * x_scale;
  projection[4] = 2.0f * ortho2[1] * x_scale;
  projection[8] = 2.0f * ortho2[2] * x_scale;
  projection[12] = x_offset;

  projection[1] = 2.0f * ortho1[0] * y_scale;
  projection[5] = 2.0f * ortho1[1] * y_scale;
  projection[9] = 2.0f * ortho1[2] * y_scale;
  projection[13] = y_offset;

  projection[2] = 2.0f * unitdir[0] * z_scale;
  projection[6] = 2.0f * unitdir[1] * z_scale;
  projection[10] = 2.0f * unitdir[2] * z_scale;
  projection[14] = z_offset;

  projection[3] = 0.0f;
  projection[7] = 0.0f;
  projection[11] = 0.0f;
  projection[15] = 1.0f;

  projection[16] = 0.0f;
  projection[17] = 0.0f;
  projection[18] = 0.0f;
  projection[19] = fast_maxf(right - left, top - bottom)/size;

  return projection;
}

std::array<float, 20> cone_projection(const float *pos, const float *dir, float angle, const float *bounds, int size) {
  std::array<float, 3> unitdir{dir[0], dir[1], dir[2]};
  std::array<float, 3> up{0.0f, 0.0f, 0.0f};
  normalize3(unitdir.data());
  up[absmin_index(dir)] = 1.0f;

  float s = dot3(unitdir.data(), up.data());
  up[0] -= s * unitdir[0];
  up[1] -= s * unitdir[1];
  up[2] -= s * unitdir[2];
  normalize3(up.data());

  std::array<float, 20> projection;

  float c = tanf(angle*0.5f);
  float near = 0.1f;
  float far = 10.0f;
  setFrustum(projection.data(), -s*near, s*near, s*near, -s*near, near, far);
  mulLookDirection(projection.data(), pos, unitdir.data(), up.data());

  projection[16] = c/size*unitdir[0];
  projection[17] = c/size*unitdir[1];
  projection[18] = c/size*unitdir[2];
  projection[19] = -dot3(projection.data()+16, pos);
  return projection;
}


extern const float sphere_sample_directions[1800];

void frame_render(ObjectRef<Frame> frameObj,
    uint32_t width,
    uint32_t height,
    uint32_t camera_index,
    uint32_t ambient_index,
    std::array<float, 4> clearColor)
{
  auto &gl = frameObj->thisDevice->gl;
  auto deviceObj = frameObj->thisDevice;
  auto worldObj = handle_cast<Object<World> *>(
      frameObj->device, frameObj->current.world.getHandle());
  CollectScene &collector = *(frameObj->collector);

  if (gl.VERSION_3_3) {
    gl.BeginQuery(GL_TIME_ELAPSED, frameObj->duration_query);
  } else if (gl.EXT_disjoint_timer_query) {
    gl.BeginQueryEXT(GL_TIME_ELAPSED_EXT, frameObj->duration_query);
  }

  gl.BindBufferBase(
      GL_SHADER_STORAGE_BUFFER, 0, deviceObj->transforms.consume());
  gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, deviceObj->lights.consume());
  gl.BindBufferBase(
      GL_SHADER_STORAGE_BUFFER, 2, deviceObj->materials.consume());

  gl.BindBuffer(GL_UNIFORM_BUFFER, frameObj->sceneubo);
  GLuint *mapping = (GLuint *)gl.MapBufferRange(GL_UNIFORM_BUFFER,
      0,
      sizeof(GLuint) * 1024,
      GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);

  mapping[0] = camera_index;
  mapping[1] = ambient_index;
  mapping[2] = collector.lights.size() / 4; // light count
  mapping[3] = frameObj->occlusionMode != STRING_ENUM_none;
  mapping[4] = width;
  mapping[5] = height;
  mapping[6] = frameObj->samples; // padding
  mapping[7] = 0; // padding
  std::memcpy(mapping + 8,
      collector.lights.data(),
      sizeof(GLuint) * collector.lights.size());

  gl.UnmapBuffer(GL_UNIFORM_BUFFER);
  gl.BindBufferBase(GL_UNIFORM_BUFFER, 0, frameObj->sceneubo);

  if (worldObj->occlusionbuffer == 0) {
    gl.GenBuffers(1, &worldObj->occlusionbuffer);
    worldObj->occlusionsamples = 0;
  }
  if (collector.vertex_count > worldObj->occlusioncapacity) {
    gl.BindBuffer(GL_SHADER_STORAGE_BUFFER, worldObj->occlusionbuffer);
    gl.BufferData(GL_SHADER_STORAGE_BUFFER,
        sizeof(float) * collector.vertex_count,
        0,
        GL_STATIC_DRAW);
    worldObj->occlusioncapacity = collector.vertex_count;
    worldObj->occlusionsamples = 0;
  }

  if (frameObj->occlusionMode != STRING_ENUM_none) {

    OcclusionResources *occlusion = deviceObj->getOcclusionResources();
    gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, worldObj->occlusionbuffer);
    if (worldObj->occlusionsamples == 0) {
      float zerof = 0.0f;
      //gl.BindBuffer(GL_SHADER_STORAGE_BUFFER, worldObj->occlusionbuffer);
      //gl.ClearBufferData(
      //    GL_SHADER_STORAGE_BUFFER, GL_R32F, GL_RED, GL_FLOAT, &zerof);

      gl.UseProgram(occlusion->clear_shader);
      gl.Uniform1i(0, worldObj->occlusioncapacity);
      gl.DispatchCompute(worldObj->occlusioncapacity/(128), 1, 1);
    }
    while (worldObj->occlusionsamples < 600) {
      gl.BindBuffer(GL_SHADER_STORAGE_BUFFER, worldObj->occlusionbuffer);
      ShadowData oc{};

      for (int i = 0; i < 12; ++i) {
        oc.projections[i].matrix = bounds_projection(
            sphere_sample_directions + 3 * (i + worldObj->occlusionsamples),
            collector.world_bounds.data(), occlusion->size);
      }
      oc.samples = worldObj->occlusionsamples;
      oc.count = 12;
      worldObj->occlusionsamples += 12;

      gl.BindBuffer(GL_UNIFORM_BUFFER, frameObj->shadowubo);
      gl.BufferData(GL_UNIFORM_BUFFER, sizeof(oc), &oc, GL_STREAM_DRAW);
      gl.BindBufferBase(GL_UNIFORM_BUFFER, 1, frameObj->shadowubo);

      // render shadow map
      gl.BindFramebuffer(GL_FRAMEBUFFER, occlusion->fbo);
      gl.Viewport(0, 0, occlusion->size, occlusion->size);
      gl.Clear(GL_DEPTH_BUFFER_BIT);

      gl.Enable(GL_DEPTH_TEST);
      for (auto &command : collector.draws) {
        command(gl, 1);
      }

      // just bind any other fbo
      gl.BindFramebuffer(GL_FRAMEBUFFER, frameObj->fbo);

      gl.MemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);
      gl.ActiveTexture(GL_TEXTURE0 + 0);
      gl.BindTexture(GL_TEXTURE_2D_ARRAY, occlusion->tex);

      gl.Enable(GL_RASTERIZER_DISCARD);
      gl.DepthMask(GL_FALSE);
      gl.Disable(GL_DEPTH_TEST);
      for (auto &command : collector.draws) {
        command(gl, 2);
      }
      gl.DepthMask(GL_TRUE);
      gl.Disable(GL_RASTERIZER_DISCARD);
      gl.MemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

      if (frameObj->occlusionMode == STRING_ENUM_incremental) {
        break;
      }
    }
  }
  // occlusion

  // shadowmaps
  if (collector.shadow_caster_count > 0
      && (worldObj->shadow_map_count < collector.shadow_caster_count
          || worldObj->shadow_map_size != frameObj->shadow_map_size)) {
    worldObj->shadow_map_count = collector.shadow_caster_count;
    worldObj->shadow_map_size = frameObj->shadow_map_size;
    frameObj->shadow_dirty = true;

    gl.DeleteTextures(1, &worldObj->shadowtex);
    gl.GenTextures(1, &worldObj->shadowtex);

    gl.BindTexture(GL_TEXTURE_2D_ARRAY, worldObj->shadowtex);
    gl.TexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    gl.TexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // clamp to border so everything outside the shadow map is considered lit
    float ones4f[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    gl.TexParameteri(
        GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    gl.TexParameteri(
        GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    gl.TexParameterfv(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_BORDER_COLOR, ones4f);

    // user percentage close filtering
    gl.TexParameteri(GL_TEXTURE_2D_ARRAY,
        GL_TEXTURE_COMPARE_MODE,
        GL_COMPARE_REF_TO_TEXTURE);
    gl.TexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL);

    gl.TexStorage3D(GL_TEXTURE_2D_ARRAY,
        1,
        GL_DEPTH_COMPONENT24,
        worldObj->shadow_map_size,
        worldObj->shadow_map_size,
        worldObj->shadow_map_count);

    if (worldObj->shadowfbo == 0) {
      gl.GenFramebuffers(1, &worldObj->shadowfbo);
    }
    gl.BindFramebuffer(GL_FRAMEBUFFER, worldObj->shadowfbo);
    gl.FramebufferTexture(
        GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, worldObj->shadowtex, 0);
  }
  // shadowmaps

  ShadowData oc{};

  for (int i = 0; i < collector.shadow_caster_count; ++i) {
    oc.projections[i].matrix = bounds_projection(
        collector.directions[i], collector.world_bounds.data(), worldObj->shadow_map_size);
  }
  oc.samples = 0;
  oc.count = collector.shadow_caster_count;

  gl.BindBuffer(GL_UNIFORM_BUFFER, frameObj->shadowubo);
  gl.BufferData(GL_UNIFORM_BUFFER, sizeof(oc), &oc, GL_STREAM_DRAW);
  gl.BindBufferBase(GL_UNIFORM_BUFFER, 1, frameObj->shadowubo);

  if (frameObj->shadow_dirty) {
    // render shadow map
    gl.BindFramebuffer(GL_FRAMEBUFFER, worldObj->shadowfbo);
    gl.Viewport(0, 0, worldObj->shadow_map_size, worldObj->shadow_map_size);
    gl.Clear(GL_DEPTH_BUFFER_BIT);
    gl.Enable(GL_DEPTH_TEST);

    for (auto &command : collector.draws) {
      command(gl, 1);
    }
    frameObj->shadow_dirty = false;
  }

  // render frame
  gl.BindFramebuffer(GL_FRAMEBUFFER, frameObj->multifbo);
  gl.Enable(GL_FRAMEBUFFER_SRGB);

  gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, worldObj->occlusionbuffer);

  // bind shadow maps
  gl.ActiveTexture(GL_TEXTURE0 + 0);
  gl.BindTexture(GL_TEXTURE_2D_ARRAY, worldObj->shadowtex);

  gl.Viewport(0, 0, width, height);
  gl.ClearColor(clearColor[0], clearColor[1], clearColor[2], clearColor[3]);
  gl.Clear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  gl.Enable(GL_DEPTH_TEST);

  gl.Enable(GL_SAMPLE_ALPHA_TO_COVERAGE);
  if(gl.VERSION_3_3) {
    gl.Enable(GL_SAMPLE_ALPHA_TO_ONE);
  }

  for (auto &command : collector.draws) {
    command(gl, 0);
  }

/*
  gl.BindFramebuffer(GL_READ_FRAMEBUFFER, frameObj->multifbo);
  gl.BindFramebuffer(GL_DRAW_FRAMEBUFFER, frameObj->fbo);
  gl.BlitFramebuffer(
      0, 0, width, height, 0, 0, width, height, GL_COLOR_BUFFER_BIT, GL_LINEAR);
*/

  // custom msaa resolve with depth linearization
  gl.BindFramebuffer(GL_FRAMEBUFFER, frameObj->fbo);
  gl.UseProgram(frameObj->resolve_shader);

  gl.ActiveTexture(GL_TEXTURE0 + 0);
  gl.BindTexture(GL_TEXTURE_2D_MULTISAMPLE, frameObj->multicolortarget);

  gl.ActiveTexture(GL_TEXTURE0 + 1);
  gl.BindTexture(GL_TEXTURE_2D_MULTISAMPLE, frameObj->multidepthtarget);

  gl.Disable(GL_CULL_FACE);
  gl.BindVertexArray(frameObj->resolve_vao);
  gl.Disable(GL_DEPTH_TEST);
  gl.DrawArrays(GL_TRIANGLES, 0, 3);

  gl.BindBuffer(GL_PIXEL_PACK_BUFFER, frameObj->colorbuffer);
  gl.ReadBuffer(GL_COLOR_ATTACHMENT0);
  if(frameObj->colorType == ANARI_FLOAT32_VEC4) {
    gl.ReadPixels(0, 0, width, height, GL_RGBA, GL_FLOAT, 0);
  } else {
    gl.ReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
  }

  if (gl.VERSION_3_3) {
    gl.EndQuery(GL_TIME_ELAPSED);
  } else if (gl.EXT_disjoint_timer_query) {
    gl.EndQueryEXT(GL_TIME_ELAPSED_EXT);
  }
}

void Object<Frame>::renderFrame()
{
  update();

  auto world = acquire<Object<World> *>(current.world);
  auto renderer = acquire<Object<RendererDefault> *>(current.renderer);
  auto camera = acquire<CameraObjectBase *>(current.camera);

  uint32_t width = size[0];
  uint32_t height = size[1];

  if (collector) {
    collector->reset();
  } else {
    collector.reset(new CollectScene);
  }

  world->accept(collector.get());

  std::array<float, 4> clearColor = {0, 0, 0, 1};
  uint32_t ambient_index = 0;

  if (renderer) {
    renderer->current.background.get(ANARI_FLOAT32_VEC4, clearColor.data());
    ambient_index = renderer->index();

    renderer->current.shadowMapSize.get(ANARI_INT32, &shadow_map_size);
    if (shadow_map_size == 0) {
      shadow_map_size = 4096;
    }
    occlusionMode = renderer->current.occlusionMode.getStringEnum();
  }

  if (world->geometryEpoch < collector->geometry_epoch) {
    shadow_dirty = true;
    world->occlusionsamples = 0;
  }

  if (world->lightEpoch < collector->light_epoch) {
    shadow_dirty = true;
  }

  world->worldEpoch = collector->epoch;
  world->geometryEpoch = collector->geometry_epoch;
  world->lightEpoch = collector->light_epoch;

  auto bounds = collector->world_bounds.data();
  camera->updateAt(camera_index, collector->world_bounds.data());

  thisDevice->transforms.lock();
  thisDevice->lights.lock();
  thisDevice->materials.lock();

  thisDevice->queue.enqueue(frame_render,
      ObjectRef<Frame>(this),
      width,
      height,
      camera_index,
      ambient_index,
      clearColor);
}

void Object<Frame>::discardFrame() {}

int Object<Frame>::frameReady(ANARIWaitMask mask)
{
  (void)mask;
  return 1;
}

int Object<Frame>::getProperty(const char *propname,
    ANARIDataType type,
    void *mem,
    uint64_t size,
    ANARIWaitMask mask)
{
  if (std::strncmp(propname, "duration", 8) == 0 && type == ANARI_FLOAT32) {
    float seconds = duration * 1.0e-9;
    if (size <= sizeof(float)) {
      std::memcpy(mem, &seconds, sizeof(float));
      return 1;
    }
  }
  return 0;
}

void frame_free_objects(Object<Device> *deviceObj,
    GLuint colortarget,
    GLuint colorbuffer,
    GLuint depthtarget,
    GLuint depthbuffer,
    GLuint fbo,
    GLuint multicolortarget,
    GLuint multidepthtarget,
    GLuint multifbo,
    GLuint duration_query,
    GLuint resolve_shader)
{
  auto &gl = deviceObj->gl;
  gl.DeleteBuffers(1, &colorbuffer);
  gl.DeleteBuffers(1, &depthbuffer);
  gl.DeleteTextures(1, &colortarget);
  gl.DeleteTextures(1, &depthtarget);
  gl.DeleteFramebuffers(1, &fbo);

  gl.DeleteRenderbuffers(1, &multicolortarget);
  gl.DeleteRenderbuffers(1, &multidepthtarget);
  gl.DeleteFramebuffers(1, &multifbo);

  if (gl.VERSION_3_3) {
    gl.DeleteQueries(1, &duration_query);
  } else if (gl.EXT_disjoint_timer_query) {
    gl.DeleteQueriesEXT(1, &duration_query);
  }
  gl.DeleteProgram(resolve_shader);
}

Object<Frame>::~Object()
{
  thisDevice->queue.enqueue(frame_free_objects,
      thisDevice,
      colortarget,
      colorbuffer,
      depthtarget,
      depthbuffer,
      fbo,
      multicolortarget,
      multidepthtarget,
      multifbo,
      duration_query,
      resolve_shader);
}

} // namespace visgl
