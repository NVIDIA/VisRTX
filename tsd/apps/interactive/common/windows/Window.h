// Copyright 2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// anari-viewer
#include "anari_viewer/windows/Window.h"

namespace tsd_viewer {

struct AppCore;

struct Window : public anari_viewer::windows::Window
{
  Window(AppCore *core, const char *name = "Window");
  virtual ~Window() override;

  virtual void buildUI() = 0;

 protected:
  AppCore *m_core{nullptr};
};

} // namespace tsd_viewer
