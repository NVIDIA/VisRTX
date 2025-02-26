// Copyright 2025 NVIDIA Corporation
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#define EGL_EGLEXT_PROTOTYPES
#include <EGL/egl.h>
#include <EGL/eglext.h>

#include "GLContextInterface.h"
#include "anari/anari.h"

namespace visgl2 {

struct eglContext : public GLContextInterface
{
  eglContext(ANARIDevice device, EGLDisplay display, EGLenum api, EGLint debug);
  ~eglContext() override;

  void init() override;
  void release() override;
  void makeCurrent() override;
  loader_func_t *loaderFunc() override;

 private:
  EGLint major{}, minor{};
  EGLenum api{};
  int32_t debug{};
  EGLDisplay display{};
  EGLConfig config{};
  EGLContext context{};
};

} // namespace visgl2
