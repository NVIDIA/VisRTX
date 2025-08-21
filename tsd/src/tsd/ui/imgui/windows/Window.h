// Copyright 2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// anari_viewer
#include "anari_viewer/windows/Window.h"
// tsd_core
#include "tsd/core/DataTree.hpp"
// tsd_app
#include "tsd/app/Core.h"

namespace tsd::ui::imgui {

class Application;

constexpr float INDENT_AMOUNT = 20.f;

struct Window : public anari_viewer::windows::Window
{
  Window(Application *app, const char *name = "Window");
  virtual ~Window() override;

  virtual void buildUI() override = 0;
  virtual void saveSettings(tsd::core::DataNode &thisWindowRoot);
  virtual void loadSettings(tsd::core::DataNode &thisWindowRoot);

 protected:
  tsd::app::Core *appCore() const;

  Application *m_app{nullptr};
};

} // namespace tsd::ui::imgui
