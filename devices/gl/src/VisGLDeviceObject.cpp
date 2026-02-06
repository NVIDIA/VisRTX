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

#include "VisGLDeviceObject.h"
#include "VisGLSpecializations.h"
#include "AppendableShader.h"
#include "shader_blocks.h"

#include "anari_library_visgl_queries.h"

#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cstdio>

#ifdef VISGL_USE_EGL
#include "egl_context.h"
#endif

#ifdef VISGL_USE_GLX
#include "glx_context.h"
#endif


#ifdef VISGL_USE_WGL
#include "wgl_context.h"
#endif
namespace visgl {

Object<Device>::Object(ANARIDevice d) : DefaultObject(d, this), queue(128) {}

int Object<Device>::getProperty(const char *propname,
    ANARIDataType type,
    void *mem,
    uint64_t size,
    ANARIWaitMask mask)
{
  if (type == ANARI_INT32 && size >= sizeof(int32_t)
      && std::strncmp("version", propname, 7) == 0) {
    int32_t version = 0; // use actual version number
    std::memcpy(mem, &version, sizeof(version));
    return 1;
  } else if (type == ANARI_UINT64 && size >= sizeof(uint64_t)
      && std::strncmp("geometryMaxIndex", propname, 16) == 0) {
    uint64_t geometryMaxIndex = INT32_MAX; // use actual value
    std::memcpy(mem, &geometryMaxIndex, sizeof(geometryMaxIndex));
    return 1;
  } else if (type == ANARI_STRING_LIST && size >= sizeof(const char**)
      && std::strncmp("extension", propname, 9) == 0) {
    const char ** value = extensions.data();
    std::memcpy(mem, &value, sizeof(const char**));
    return 1;
  } else {
    return 0;
  }
}

void Object<Device>::commit()
{
  DefaultObject::commit();
}

static void debug_callback(GLenum source,
    GLenum type,
    GLenum id,
    GLenum severity,
    GLsizei length,
    const GLchar *message,
    const void *userdata)
{
  if (severity != GL_DEBUG_SEVERITY_HIGH) {
    return;
  }
  ANARIDevice device =
      reinterpret_cast<ANARIDevice>(const_cast<void *>(userdata));
  anariReportStatus(device,
      device,
      ANARI_DEVICE,
      ANARI_SEVERITY_INFO,
      ANARI_STATUS_NO_ERROR,
      "[OpenGL] %s",
      message);
}

static void device_context_init(
    Object<Device> *deviceObj, int clientapi, int32_t debug)
{
  deviceObj->context->init();
  deviceObj->context->makeCurrent();

  int version = 0;
  if (clientapi == STRING_ENUM_OpenGL_ES) {
    version = gladLoadGLES2Context(
        &deviceObj->gl, (GLADloadfunc)deviceObj->context->loaderFunc());
  } else if (clientapi == STRING_ENUM_OpenGL) {
    version = gladLoadGLContext(
        &deviceObj->gl, (GLADloadfunc)deviceObj->context->loaderFunc());
  }

  auto &gl = deviceObj->gl;

  const char **ext = query_extensions();
  for(int i = 0;ext[i] != nullptr;++i) {
    if(strncmp("ANARI_EXT_SAMPLER_COMPRESSED_FORMAT_BC123", ext[i], 41)==0) {
      if(gl.EXT_texture_compression_s3tc) {
        deviceObj->extensions.push_back(ext[i]);
      }
    } else if(strncmp("ANARI_EXT_SAMPLER_COMPRESSED_FORMAT_BC45", ext[i], 40)==0) {
      if(gl.ARB_texture_compression_rgtc) {
        deviceObj->extensions.push_back(ext[i]);
      }
    } else if(strncmp("ANARI_EXT_SAMPLER_COMPRESSED_FORMAT_BC67", ext[i], 40)==0) {
      if(gl.ARB_texture_compression_bptc) {
        deviceObj->extensions.push_back(ext[i]);
      }
    } else if(strncmp("ANARI_EXT_SAMPLER_COMPRESSED_FORMAT_ASTC", ext[i], 40)==0) {
      if(gl.KHR_texture_compression_astc_ldr || gl.ES_VERSION_3_2) {
        deviceObj->extensions.push_back(ext[i]);
      }
    } else if(strncmp("ANARI_EXT_SAMPLER_COMPRESSED_FORMAT_ETC2", ext[i], 40)==0) {
      if(gl.VERSION_4_3 || gl.ES_VERSION_3_2) {
        deviceObj->extensions.push_back(ext[i]);
      }
    } else if(strncmp("ANARI_EXT_SAMPLER_COMPRESSED_FORMAT_EAC", ext[i], 39)==0) {
      if(gl.VERSION_4_3 || gl.ES_VERSION_3_2) {
        deviceObj->extensions.push_back(ext[i]);
      }
    } else {
      deviceObj->extensions.push_back(ext[i]);
    }
  }
  deviceObj->extensions.push_back(nullptr);

  if (version == 0) {
    anariReportStatus(deviceObj->device,
        deviceObj->handle,
        ANARI_DEVICE,
        ANARI_SEVERITY_INFO,
        ANARI_STATUS_NO_ERROR,
        "[GLAD] Failed to load GLES entry points");
  }

  if (debug && deviceObj->gl.DebugMessageCallback) {
    anariReportStatus(deviceObj->device,
      deviceObj->handle,
      ANARI_DEVICE,
      ANARI_SEVERITY_INFO,
      ANARI_STATUS_NO_ERROR,
      "[OpenGL] setup debug callback\n");
    gl.DebugMessageCallback(debug_callback, deviceObj->device);
    gl.DebugMessageInsert(
      GL_DEBUG_SOURCE_APPLICATION,
      GL_DEBUG_TYPE_OTHER,
      0,
      GL_DEBUG_SEVERITY_NOTIFICATION,
      -1, "test message callback.");
  }

  anariReportStatus(deviceObj->device,
      deviceObj->handle,
      ANARI_DEVICE,
      ANARI_SEVERITY_INFO,
      ANARI_STATUS_NO_ERROR,
      "%s\n",
      gl.GetString(GL_VERSION));

  deviceObj->transforms.init(&deviceObj->gl);
  deviceObj->materials.init(&deviceObj->gl);
  deviceObj->lights.init(&deviceObj->gl);
  deviceObj->shaders.init(&deviceObj->gl);

  // insert matrices for the 0 instance
  // clang-format off
  std::array<float, 16> identity_matrix = {
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0,
      0, 0, 0, 1,
  };
  // clang-format on
  deviceObj->transforms.allocate(3);
  deviceObj->transforms.set(0, identity_matrix);
  deviceObj->transforms.set(1, identity_matrix);
  deviceObj->transforms.set(2, identity_matrix);

  deviceObj->lights.allocate(1);
  std::array<float, 4> zero = {0.0f, 0.0f, 0.0f, 0.0f};
  deviceObj->lights.set(0, zero);

#define REPORT_GL_INT(X)                                                       \
  {                                                                            \
    GLint value;                                                               \
    gl.GetIntegerv(X, &value);                                                 \
    anariReportStatus(deviceObj->device,                                       \
        deviceObj->handle,                                                     \
        ANARI_DEVICE,                                                          \
        ANARI_SEVERITY_INFO,                                                   \
        ANARI_STATUS_NO_ERROR,                                                 \
        "[OpenGL] " #X " %d",                                                  \
        value);                                                                \
  }
}

void Object<Device>::update()
{
  DefaultObject::update();

  if (context) {
    return;
  }

  // retrieve gl context parameters
  clientapi = current.glAPI.getStringEnum();
  int8_t debug = 0;
  current.glDebug.get(ANARI_BOOL, &debug);

#ifdef VISGL_USE_GLX
  Display *display = glXGetCurrentDisplay();
  if (display) {
    GLXContext glx_context = glXGetCurrentContext();
    context.reset(new glxContext(device, display, glx_context, debug));
  } else {
#endif

#ifdef VISGL_USE_EGL
    EGLenum api;
    if (clientapi == STRING_ENUM_OpenGL_ES) {
      api = EGL_OPENGL_ES_API;
    } else if (clientapi == STRING_ENUM_OpenGL) {
      api = EGL_OPENGL_API;
    }

    EGLDisplay egldisplay = EGL_NO_DISPLAY;
    current.EGLDisplay.get(ANARI_VOID_POINTER, &egldisplay);
    if (egldisplay == EGL_NO_DISPLAY) {
      egldisplay = eglGetCurrentDisplay();
    }
    context.reset(
        new eglContext(device, egldisplay, api, debug ? EGL_TRUE : EGL_FALSE));
#endif

#ifdef VISGL_USE_GLX
  }
#endif

#ifdef VISGL_USE_WGL
  HDC dc = wglGetCurrentDC();
  HGLRC wgl_context = wglGetCurrentContext();
  context.reset(new wglContext(
      device, dc, wgl_context, clientapi == STRING_ENUM_OpenGL_ES, debug));
#endif

  queue.enqueue(device_context_init, this, clientapi, debug).wait();
}

void device_init_occlusion(ObjectRef<Device> device, OcclusionResources *res)
{
  auto &gl = device->gl;
  gl.GenTextures(1, &res->tex);
  gl.BindTexture(GL_TEXTURE_2D_ARRAY, res->tex);
  gl.TexStorage3D(
      GL_TEXTURE_2D_ARRAY, 1, GL_DEPTH_COMPONENT24, res->size, res->size, 12);
  gl.TexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  gl.TexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

  // clamp to border so everything outside the shadow map is considered lit
  float ones4f[4] = {1.0f, 1.0f, 1.0f, 1.0f};
  gl.TexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
  gl.TexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
  gl.TexParameterfv(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_BORDER_COLOR, ones4f);

  // user percentage close filtering
  gl.TexParameteri(
      GL_TEXTURE_2D_ARRAY, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_REF_TO_TEXTURE);
  gl.TexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL);

  gl.GenFramebuffers(1, &res->fbo);
  gl.BindFramebuffer(GL_FRAMEBUFFER, res->fbo);
  gl.FramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, res->tex, 0);

  const char *version = gl.VERSION_4_3 ? version_430 : version_320_es;

  StaticAppendableShader<SHADER_SEGMENTS> clear_shader;
  clear_shader.append(version);
  clear_shader.append(occlusion_declaration);
  clear_shader.append(clear_occlusion_source);

  res->clear_shader =
    device->shaders.getCompute(clear_shader);

}

OcclusionResources *Object<Device>::getOcclusionResources()
{
  if (occlusion.tex == 0) {
    // queue.enqueue(device_init_occlusion, gl, &occlusion).wait();
    device_init_occlusion(ObjectRef<Device>(this), &occlusion);
  }
  return &occlusion;
}

uint64_t anariIncrementEpoch(Object<Device> *device, ObjectBase *)
{
  return ++device->epochCounter;
}

uint64_t Object<Device>::globalEpoch() const
{
  return epochCounter;
}

static void device_context_free(Object<Device> *deviceObj)
{
  deviceObj->transforms.release();
  deviceObj->materials.release();
  deviceObj->lights.release();
  deviceObj->shaders.release();

  deviceObj->context->release();
}

Object<Device>::~Object()
{
  queue.enqueue(device_context_free, this).wait();
  context.reset(nullptr);
}

} // namespace visgl
