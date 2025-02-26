// Copyright 2025 NVIDIA Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "VisGL2Device.h"

#include "array/Array1D.h"
#include "array/Array2D.h"
#include "array/Array3D.h"
#include "array/ObjectArray.h"
#include "frame/Frame.h"
#include "scene/volume/spatial_field/SpatialField.h"

#include "anari_library_visgl2_queries.h"

#ifdef VISGL2_USE_EGL
#include "gl/egl/egl_context.h"
#endif
#ifdef VISGL2_USE_GLX
#include "gl/glx/glx_context.h"
#endif
#ifdef VISGL2_USE_WGL
#include "gl/wgl/wgl_context.h"
#endif

namespace visgl2 {

// Helper functions ///////////////////////////////////////////////////////////

static void gl_debug_callback(GLenum source,
    GLenum type,
    GLenum id,
    GLenum severity,
    GLsizei length,
    const GLchar *message,
    const void *userdata)
{
  if (severity != GL_DEBUG_SEVERITY_HIGH)
    return;
  auto device = reinterpret_cast<ANARIDevice>(const_cast<void *>(userdata));
  anariDeviceReportStatus(device,
      ANARI_SEVERITY_INFO,
      ANARI_STATUS_NO_ERROR,
      "[OpenGL] %s",
      message);
}

static void gl_context_init(VisGL2DeviceGlobalState *state, bool debug)
{
  auto &context = state->gl.context;
  auto &extensions = state->gl.extensions;
  auto &gl = state->gl.gl;

  context->init();
  context->makeCurrent();

  int version = 0;
  if (state->gl.useGLES)
    version = gladLoadGLES2Context(&gl, (GLADloadfunc)context->loaderFunc());
  else
    version = gladLoadGLContext(&gl, (GLADloadfunc)context->loaderFunc());

  const char **ext = query_extensions();
  for (int i = 0; ext[i] != nullptr; ++i) {
    if (strncmp("ANARI_EXT_SAMPLER_COMPRESSED_FORMAT_BC123", ext[i], 41) == 0) {
      if (gl.EXT_texture_compression_s3tc)
        extensions.push_back(ext[i]);
    } else if (strncmp("ANARI_EXT_SAMPLER_COMPRESSED_FORMAT_BC45", ext[i], 40)
        == 0) {
      if (gl.ARB_texture_compression_rgtc)
        extensions.push_back(ext[i]);
    } else if (strncmp("ANARI_EXT_SAMPLER_COMPRESSED_FORMAT_BC67", ext[i], 40)
        == 0) {
      if (gl.ARB_texture_compression_bptc)
        extensions.push_back(ext[i]);
    } else if (strncmp("ANARI_EXT_SAMPLER_COMPRESSED_FORMAT_ASTC", ext[i], 40)
        == 0) {
      if (gl.KHR_texture_compression_astc_ldr || gl.ES_VERSION_3_2)
        extensions.push_back(ext[i]);
    } else if (strncmp("ANARI_EXT_SAMPLER_COMPRESSED_FORMAT_ETC2", ext[i], 40)
        == 0) {
      if (gl.VERSION_4_3 || gl.ES_VERSION_3_2)
        extensions.push_back(ext[i]);
    } else if (strncmp("ANARI_EXT_SAMPLER_COMPRESSED_FORMAT_EAC", ext[i], 39)
        == 0) {
      if (gl.VERSION_4_3 || gl.ES_VERSION_3_2)
        extensions.push_back(ext[i]);
    } else
      extensions.push_back(ext[i]);
  }
  extensions.push_back(nullptr);

  if (version == 0) {
    anariDeviceReportStatus(state->device,
        ANARI_SEVERITY_INFO,
        ANARI_STATUS_NO_ERROR,
        "[GLAD] Failed to load GLES entry points");
  }

  if (debug && gl.DebugMessageCallback) {
    anariDeviceReportStatus(state->device,
        ANARI_SEVERITY_INFO,
        ANARI_STATUS_NO_ERROR,
        "[OpenGL] setup debug callback\n");
    gl.DebugMessageCallback(gl_debug_callback, state->device);
    gl.DebugMessageInsert(GL_DEBUG_SOURCE_APPLICATION,
        GL_DEBUG_TYPE_OTHER,
        0,
        GL_DEBUG_SEVERITY_NOTIFICATION,
        -1,
        "test message callback.");
  }

  anariDeviceReportStatus(state->device,
      ANARI_SEVERITY_INFO,
      ANARI_STATUS_NO_ERROR,
      "%s\n",
      gl.GetString(GL_VERSION));
}

// API Objects ////////////////////////////////////////////////////////////////

ANARIArray1D VisGL2Device::newArray1D(const void *appMemory,
    ANARIMemoryDeleter deleter,
    const void *userData,
    ANARIDataType type,
    uint64_t numItems)
{
  initDevice();

  Array1DMemoryDescriptor md;
  md.appMemory = appMemory;
  md.deleter = deleter;
  md.deleterPtr = userData;
  md.elementType = type;
  md.numItems = numItems;

  if (anari::isObject(type))
    return (ANARIArray1D) new ObjectArray(deviceState(), md);
  else
    return (ANARIArray1D) new Array1D(deviceState(), md);
}

ANARIArray2D VisGL2Device::newArray2D(const void *appMemory,
    ANARIMemoryDeleter deleter,
    const void *userData,
    ANARIDataType type,
    uint64_t numItems1,
    uint64_t numItems2)
{
  initDevice();

  Array2DMemoryDescriptor md;
  md.appMemory = appMemory;
  md.deleter = deleter;
  md.deleterPtr = userData;
  md.elementType = type;
  md.numItems1 = numItems1;
  md.numItems2 = numItems2;

  return (ANARIArray2D) new Array2D(deviceState(), md);
}

ANARIArray3D VisGL2Device::newArray3D(const void *appMemory,
    ANARIMemoryDeleter deleter,
    const void *userData,
    ANARIDataType type,
    uint64_t numItems1,
    uint64_t numItems2,
    uint64_t numItems3)
{
  initDevice();

  Array3DMemoryDescriptor md;
  md.appMemory = appMemory;
  md.deleter = deleter;
  md.deleterPtr = userData;
  md.elementType = type;
  md.numItems1 = numItems1;
  md.numItems2 = numItems2;
  md.numItems3 = numItems3;

  return (ANARIArray3D) new Array3D(deviceState(), md);
}

ANARICamera VisGL2Device::newCamera(const char *subtype)
{
  initDevice();
  return (ANARICamera)Camera::createInstance(subtype, deviceState());
}

ANARIFrame VisGL2Device::newFrame()
{
  initDevice();
  return (ANARIFrame) new Frame(deviceState());
}

ANARIGeometry VisGL2Device::newGeometry(const char *subtype)
{
  initDevice();
  return (ANARIGeometry)Geometry::createInstance(subtype, deviceState());
}

ANARIGroup VisGL2Device::newGroup()
{
  initDevice();
  return (ANARIGroup) new Group(deviceState());
}

ANARIInstance VisGL2Device::newInstance(const char * /*subtype*/)
{
  initDevice();
  return (ANARIInstance) new Instance(deviceState());
}

ANARILight VisGL2Device::newLight(const char *subtype)
{
  initDevice();
  return (ANARILight)Light::createInstance(subtype, deviceState());
}

ANARIMaterial VisGL2Device::newMaterial(const char *subtype)
{
  initDevice();
  return (ANARIMaterial)Material::createInstance(subtype, deviceState());
}

ANARIRenderer VisGL2Device::newRenderer(const char *subtype)
{
  initDevice();
  return (ANARIRenderer)Renderer::createInstance(subtype, deviceState());
}

ANARISampler VisGL2Device::newSampler(const char *subtype)
{
  initDevice();
  return (ANARISampler)Sampler::createInstance(subtype, deviceState());
}

ANARISpatialField VisGL2Device::newSpatialField(const char *subtype)
{
  initDevice();
  return (ANARISpatialField)SpatialField::createInstance(
      subtype, deviceState());
}

ANARISurface VisGL2Device::newSurface()
{
  initDevice();
  return (ANARISurface) new Surface(deviceState());
}

ANARIVolume VisGL2Device::newVolume(const char *subtype)
{
  initDevice();
  return (ANARIVolume)Volume::createInstance(subtype, deviceState());
}

ANARIWorld VisGL2Device::newWorld()
{
  initDevice();
  return (ANARIWorld) new World(deviceState());
}

// Query functions ////////////////////////////////////////////////////////////

const char **VisGL2Device::getObjectSubtypes(ANARIDataType objectType)
{
  return visgl2::query_object_types(objectType);
}

const void *VisGL2Device::getObjectInfo(ANARIDataType objectType,
    const char *objectSubtype,
    const char *infoName,
    ANARIDataType infoType)
{
  return visgl2::query_object_info(
      objectType, objectSubtype, infoName, infoType);
}

const void *VisGL2Device::getParameterInfo(ANARIDataType objectType,
    const char *objectSubtype,
    const char *parameterName,
    ANARIDataType parameterType,
    const char *infoName,
    ANARIDataType infoType)
{
  return visgl2::query_param_info(objectType,
      objectSubtype,
      parameterName,
      parameterType,
      infoName,
      infoType);
}

// Other VisGL2Device definitions /////////////////////////////////////////////

VisGL2Device::VisGL2Device(ANARIStatusCallback cb, const void *ptr)
    : helium::BaseDevice(cb, ptr)
{
  m_state = std::make_unique<VisGL2DeviceGlobalState>(this_device());
  deviceCommitParameters();
}

VisGL2Device::VisGL2Device(ANARILibrary l) : helium::BaseDevice(l)
{
  m_state = std::make_unique<VisGL2DeviceGlobalState>(this_device());
  deviceCommitParameters();
}

VisGL2Device::~VisGL2Device()
{
  auto &state = *deviceState();
  state.commitBuffer.clear();
  reportMessage(ANARI_SEVERITY_DEBUG, "destroying VisGL2 device (%p)", this);
}

void VisGL2Device::initDevice()
{
  if (m_initialized)
    return;

  reportMessage(ANARI_SEVERITY_DEBUG, "initializing VisGL2 device (%p)", this);

  auto &state = *deviceState();
  auto &context = state.gl.context;

  state.gl.useGLES = m_glAPI == "OpenGL_ES";

#ifdef VISGL2_USE_GLX
  Display *display = glXGetCurrentDisplay();
  if (display) {
    GLXContext glx_context = glXGetCurrentContext();
    context.reset(
        new glxContext(this_device(), display, glx_context, m_glDebug));
  } else {
#endif

#ifdef VISGL2_USE_EGL
    EGLenum api = state.gl.useGLES ? EGL_OPENGL_ES_API : EGL_OPENGL_API;
    auto egldisplay = (EGLDisplay)m_eglDisplay;
    if (egldisplay == EGL_NO_DISPLAY)
      egldisplay = eglGetCurrentDisplay();
    context.reset(new eglContext(
        this_device(), egldisplay, api, m_glDebug ? EGL_TRUE : EGL_FALSE));
#endif

#ifdef VISGL2_USE_GLX
  }
#endif

#ifdef VISGL2_USE_WGL
  HDC dc = wglGetCurrentDC();
  HGLRC wgl_context = wglGetCurrentContext();
  context.reset(new wglContext(
      this_device(), dc, wgl_context, state.gl.useGLES, m_glDebug));
#endif

  state.gl.thread.enqueue(gl_context_init, deviceState(), m_glDebug).wait();

  m_initialized = true;
}

void VisGL2Device::deviceCommitParameters()
{
  helium::BaseDevice::deviceCommitParameters();
  m_glAPI = getParamString("glAPI", "OpenGL");
  m_glDebug = getParam<bool>("glDebug", false);
  m_eglDisplay = getParam<void *>("EGLDisplay", nullptr);
}

int VisGL2Device::deviceGetProperty(
    const char *name, ANARIDataType type, void *mem, uint64_t size)
{
  std::string_view prop = name;
  if (prop == "extension" && type == ANARI_STRING_LIST) {
    helium::writeToVoidP(mem, query_extensions());
    return 1;
  } else if (prop == "VisGL2" && type == ANARI_BOOL) {
    helium::writeToVoidP(mem, true);
    return 1;
  }
  return 0;
}

VisGL2DeviceGlobalState *VisGL2Device::deviceState() const
{
  return (VisGL2DeviceGlobalState *)helium::BaseDevice::m_state.get();
}

} // namespace visgl2
