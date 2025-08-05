// Copyright 2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// anari-viewer
#include "anari_viewer/windows/Window.h"
// tsd
#include "tsd/core/DataTree.hpp"

namespace tsd_viewer {

struct AppCore;

constexpr float INDENT_AMOUNT = 20.f;

struct Window : public anari_viewer::windows::Window
{
  Window(AppCore *core, const char *name = "Window");
  virtual ~Window() override;

  virtual void buildUI() override = 0;
  virtual void saveSettings(tsd::core::DataNode &thisWindowRoot);
  virtual void loadSettings(tsd::core::DataNode &thisWindowRoot);

 protected:
  AppCore *m_core{nullptr};
};

} // namespace tsd_viewer
