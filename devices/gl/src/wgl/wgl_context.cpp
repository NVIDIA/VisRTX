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

#include "wgl_context.h"
#include "VisGLDevice.h"

#include <cstring>
#include <glad/wgl.h>



static void (*wgl_loader(char const *name))(void)
{
  static HMODULE libGL = nullptr;
  if (libGL == nullptr) {
    libGL = LoadLibraryW(L"opengl32.dll");
  }

  void (*fptr)() = nullptr;

  if (wglGetProcAddress) {
    fptr = (void (*)())wglGetProcAddress((const LPCSTR)name);
  }
  if (fptr) {
    return fptr;
  } else {
    return (void (*)())GetProcAddress(libGL, (const LPCSTR)name);
  }
}

using visgl::anariReportStatus;

wglContext::wglContext(
    ANARIDevice device, HDC dc, HGLRC wgl_context, bool use_es, int32_t debug)
    : device(device),
      host_dc(dc),
      host_wgl_context(wgl_context),
      use_es(use_es),
      debug(debug)
{
}

static HWND create_window() {
  WNDCLASSEX wndClass;
  memset(&wndClass, 0, sizeof wndClass);
  wndClass.cbSize = sizeof(WNDCLASSEX);
  wndClass.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC | CS_DBLCLKS;
  wndClass.lpfnWndProc = &DefWindowProc;
  wndClass.cbClsExtra = 0;
  wndClass.cbWndExtra = 0;
  wndClass.hInstance = 0;
  wndClass.hIcon = 0;
  wndClass.hCursor = LoadCursor(0, IDC_ARROW);
  wndClass.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);
  wndClass.lpszMenuName = 0;
  wndClass.lpszClassName = "AnariWndClass";
  wndClass.hIconSm = 0;
  RegisterClassEx(&wndClass);
  return CreateWindowEx(0,
      "AnariWndClass",
      "",
      WS_CLIPSIBLINGS | WS_CLIPCHILDREN | WS_POPUP,
      CW_USEDEFAULT,
      CW_USEDEFAULT,
      1, // width
      1, // height
      0,
      0,
      0,
      0);
}

void wglContext::init()
{
  hwnd = create_window();
  if (!hwnd) {
    anariReportStatus(device,
        device,
        ANARI_DEVICE,
        ANARI_SEVERITY_FATAL_ERROR,
        ANARI_STATUS_UNKNOWN_ERROR,
        "[OpenGL] failed to open window");
    return;
  }
  dc = GetDC(hwnd);

  PIXELFORMATDESCRIPTOR pfd = {};
  pfd.nSize = sizeof(pfd);
  pfd.nSize = sizeof(PIXELFORMATDESCRIPTOR);
  pfd.dwFlags = PFD_DOUBLEBUFFER | PFD_SUPPORT_OPENGL | PFD_DRAW_TO_WINDOW;
  pfd.iPixelType = PFD_TYPE_RGBA;
  pfd.cColorBits = 32;
  pfd.cDepthBits = 32;
  pfd.iLayerType = PFD_MAIN_PLANE;
  int format = ChoosePixelFormat(dc, &pfd);

  if (!SetPixelFormat(dc, format, &pfd)) {
    anariReportStatus(device,
        device,
        ANARI_DEVICE,
        ANARI_SEVERITY_FATAL_ERROR,
        ANARI_STATUS_UNKNOWN_ERROR,
        "[OpenGL] failed to set pixel format");
    ReleaseDC(hwnd, dc);
    DestroyWindow(hwnd);
    return;
  }
  HGLRC tmp = wglCreateContext(dc);

  if (!tmp) {
    anariReportStatus(device,
        device,
        ANARI_DEVICE,
        ANARI_SEVERITY_FATAL_ERROR,
        ANARI_STATUS_UNKNOWN_ERROR,
        "[OpenGL] failed to create intermediate context");
    ReleaseDC(hwnd, dc);
    DestroyWindow(hwnd);
    return;
  }

  if (!wglMakeCurrent(dc, tmp)) {
    anariReportStatus(device,
        device,
        ANARI_DEVICE,
        ANARI_SEVERITY_FATAL_ERROR,
        ANARI_STATUS_UNKNOWN_ERROR,
        "[OpenGL] make current context");
    ReleaseDC(hwnd, dc);
    DestroyWindow(hwnd);
    return;
  }

  if (gladLoadWGL(dc, wgl_loader) == 0) {
    anariReportStatus(device,
        device,
        ANARI_DEVICE,
        ANARI_SEVERITY_FATAL_ERROR,
        ANARI_STATUS_UNKNOWN_ERROR,
        "[OpenGL] failed to load wgl");
    wglMakeCurrent(dc, nullptr);
    ReleaseDC(hwnd, dc);
    DestroyWindow(hwnd);
    return;
  }

  if (!WGL_ARB_create_context) {
    anariReportStatus(device,
        device,
        ANARI_DEVICE,
        ANARI_SEVERITY_FATAL_ERROR,
        ANARI_STATUS_UNKNOWN_ERROR,
        "[OpenGL] WGL_ARB_create_context unsupported");
    wglMakeCurrent(dc, nullptr);
    wglDeleteContext(tmp);
    ReleaseDC(hwnd, dc);
    DestroyWindow(hwnd);
    return;
  }
  if (!WGL_ARB_create_context_profile) {
    anariReportStatus(device,
        device,
        ANARI_DEVICE,
        ANARI_SEVERITY_FATAL_ERROR,
        ANARI_STATUS_UNKNOWN_ERROR,
        "[OpenGL] WGL_ARB_create_context_profile unsupported");
    wglMakeCurrent(dc, nullptr);
    wglDeleteContext(tmp);
    ReleaseDC(hwnd, dc);
    DestroyWindow(hwnd);
    return;
  }


  int context_attributes[] = {
      WGL_CONTEXT_MAJOR_VERSION_ARB, 4,
      WGL_CONTEXT_MINOR_VERSION_ARB, 3,
      WGL_CONTEXT_FLAGS_ARB,
      WGL_CONTEXT_CORE_PROFILE_BIT_ARB | (debug ? WGL_CONTEXT_DEBUG_BIT_ARB : 0),
      0};

    int context_attributes_ES[] = {
      WGL_CONTEXT_MAJOR_VERSION_ARB, 3,
      WGL_CONTEXT_MINOR_VERSION_ARB, 2,
      WGL_CONTEXT_FLAGS_ARB,
      WGL_CONTEXT_CORE_PROFILE_BIT_ARB | WGL_CONTEXT_ES2_PROFILE_BIT_EXT
          | (debug ? WGL_CONTEXT_DEBUG_BIT_ARB : 0),
      0};

  wgl_context = wglCreateContextAttribsARB(
        dc, NULL, use_es ? context_attributes_ES : context_attributes);

  if (!wgl_context) {
    anariReportStatus(device,
        device,
        ANARI_DEVICE,
        ANARI_SEVERITY_FATAL_ERROR,
        ANARI_STATUS_UNKNOWN_ERROR,
        "[OpenGL] failed to create context");
    wglMakeCurrent(dc, nullptr);
    wglDeleteContext(tmp);
    ReleaseDC(hwnd, dc);
    DestroyWindow(hwnd);
    return;
  }
  wglMakeCurrent(dc, nullptr);
  wglDeleteContext(tmp);
}

void wglContext::makeCurrent() {
  wglMakeCurrent(dc, wgl_context);
}

glContextInterface::loader_func_t* wglContext::loaderFunc() {
  return wgl_loader;
}

void wglContext::release() {
  wglMakeCurrent(dc, nullptr);
}

wglContext::~wglContext() {
  wglDeleteContext(wgl_context);
  ReleaseDC(hwnd, dc);
  DestroyWindow(hwnd);
}