// Copyright 2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// anari-viewer
#include "anari_viewer/windows/Window.h"
// tsd
#include "tsd/containers/DataTree.hpp"

namespace tsd_viewer {

struct AppCore;

struct Window : public anari_viewer::windows::Window
{
  Window(AppCore *core, const char *name = "Window");
  virtual ~Window() override;

  virtual void buildUI() override = 0;
  virtual void saveSettings(tsd::serialization::DataNode &thisWindowRoot);
  virtual void loadSettings(tsd::serialization::DataNode &thisWindowRoot);

 protected:
  AppCore *m_core{nullptr};
};

} // namespace tsd_viewer
