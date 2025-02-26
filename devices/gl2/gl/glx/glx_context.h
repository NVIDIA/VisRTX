// Copyright 2025 NVIDIA Corporation
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "ogl.h"
// gl
#include <GL/glx.h>

#include "GLContextInterface.h"
#include "anari/anari.h"

namespace visgl2 {

struct glxContext : public GLContextInterface
{
  glxContext(
      ANARIDevice device, Display *display, GLXContext context, int32_t debug);
  ~glxContext() override;

  void init() override;
  void release() override;
  void makeCurrent() override;
  loader_func_t *loaderFunc() override;

 private:
  Display *display{nullptr};
  int32_t debug{};
  GLXContext share{};
  GLXContext context{};
  GLXPbuffer pbuffer{};
};

} // namespace visgl2
