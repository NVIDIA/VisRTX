// Copyright 2025 NVIDIA Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "glx_context.h"
#include "VisGL2Device.h"
// std
#include <cstring>
// X11
#include <X11/Xlib.h>
#include <X11/Xutil.h>

namespace visgl2 {

glxContext::glxContext(
    ANARIDevice device, Display *display, GLXContext glx_context, int32_t debug)
    : GLContextInterface(device),
      display(display),
      share(glx_context),
      debug(debug)
{}

glxContext::~glxContext()
{
  glXDestroyContext(display, context);
  glXDestroyPbuffer(display, pbuffer);
}

void glxContext::init()
{
  auto *d = device();

  anariDeviceReportStatus(
      d, ANARI_SEVERITY_INFO, ANARI_STATUS_NO_ERROR, "[OpenGL] using GLX");

  int attrib[] = {
      GLX_RENDER_TYPE, GLX_RGBA_BIT, GLX_DRAWABLE_TYPE, GLX_PBUFFER_BIT, None};

  int pbAttrib[] = {GLX_PBUFFER_WIDTH,
      128,
      GLX_PBUFFER_HEIGHT,
      128,
      GLX_LARGEST_PBUFFER,
      0,
      None};

  if (!display) {
    display = XOpenDisplay(nullptr);
  }

  int screen = DefaultScreen(display);

  const char *extensions = glXQueryExtensionsString(display, screen);
  bool create_context_profile =
      std::strstr(extensions, "GLX_ARB_create_context_profile") != nullptr;
  bool no_config_context =
      std::strstr(extensions, "GLX_EXT_no_config_context") != nullptr;
  bool create_context_es2_profile =
      std::strstr(extensions, "GLX_EXT_create_context_es2_profile") != nullptr;

  int count;
  GLXFBConfig *config = glXChooseFBConfig(display, screen, attrib, &count);

  if (count == 0) {
    anariDeviceReportStatus(d,
        ANARI_SEVERITY_FATAL_ERROR,
        ANARI_STATUS_UNKNOWN_ERROR,
        "[OpenGL] no config");
    return;
  }

  pbuffer = glXCreatePbuffer(display, config[0], pbAttrib);

  if (!pbuffer) {
    anariDeviceReportStatus(d,
        ANARI_SEVERITY_FATAL_ERROR,
        ANARI_STATUS_UNKNOWN_ERROR,
        "[OpenGL] failed to create pbuffer");
    return;
  }

  if (create_context_profile) {
    PFNGLXCREATECONTEXTATTRIBSARBPROC glXCreateContextAttribsARB =
        (PFNGLXCREATECONTEXTATTRIBSARBPROC)glXGetProcAddress(
            (const GLubyte *)"glXCreateContextAttribsARB");

    if (debug) {
      anariDeviceReportStatus(d,
          ANARI_SEVERITY_INFO,
          ANARI_STATUS_NO_ERROR,
          "[OpenGL] create debug context");
    }

    const int contextAttribs[] = {GLX_CONTEXT_PROFILE_MASK_ARB,
        GLX_CONTEXT_CORE_PROFILE_BIT_ARB,
        GLX_CONTEXT_FLAGS_ARB,
        (debug ? GLX_CONTEXT_DEBUG_BIT_ARB : 0),
        GLX_CONTEXT_MAJOR_VERSION_ARB,
        4,
        GLX_CONTEXT_MINOR_VERSION_ARB,
        3,
        None};
    context = glXCreateContextAttribsARB(
        display, config[0], share, true, contextAttribs);
  } else {
    anariDeviceReportStatus(d,
        ANARI_SEVERITY_INFO,
        ANARI_STATUS_NO_ERROR,
        "[OpenGL] fall back to legacy context creation");
    context =
        glXCreateNewContext(display, config[0], GLX_RGBA_TYPE, share, true);
  }
  if (!context) {
    anariDeviceReportStatus(d,
        ANARI_SEVERITY_FATAL_ERROR,
        ANARI_STATUS_UNKNOWN_ERROR,
        "[OpenGL] failed to create context");
    return;
  }

  XFree(config);
}

void glxContext::makeCurrent()
{
  glXMakeCurrent(display, pbuffer, context);
}

static void (*glx_loader(char const *name))(void)
{
  return glXGetProcAddress((const GLubyte *)name);
}

GLContextInterface::loader_func_t *glxContext::loaderFunc()
{
  return glx_loader;
}

void glxContext::release()
{
  glXMakeCurrent(display, None, nullptr);
}

} // namespace visgl2
