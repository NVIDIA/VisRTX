// Copyright 2025 NVIDIA Corporation
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "ogl.h"

#define WIN32_LEAN_AND_MEAN
#include <glad/wgl.h>
#include <windows.h>

#undef near
#undef far

#include "GLContextInterface.h"
#include "anari/anari.h"

namespace visgl2 {

class wglContext : public GLContextInterface
{
  wglContext(ANARIDevice device,
      HDC dc,
      HGLRC wgl_context,
      bool ues_es,
      int32_t debug);
  ~wglContext() override;

  void init() override;
  void release() override;
  void makeCurrent() override;
  loader_func_t *loaderFunc() override;

 private:
  HDC host_dc{};
  HGLRC host_wgl_context{};
  bool use_es{false};
  int32_t debug{};

  HWND hwnd{};
  HDC dc{};
  HGLRC wgl_context{};
};

} // namespace visgl2
