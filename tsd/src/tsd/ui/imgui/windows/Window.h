// Copyright 2025-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// tsd_core
#include "tsd/core/DataTree.hpp"
// tsd_app
#include "tsd/app/Context.h"
// imgui
#include <imgui.h>
// std
#include <string>

namespace tsd::ui::imgui {

class Application;

constexpr float INDENT_AMOUNT = 20.f;

struct Window
{
  Window(Application *app, const char *name);
  virtual ~Window();

  void renderUI();

  void show();
  void hide();
  void toggleShown();

  bool *visiblePtr();
  const char *name();

  // Interface to override for custom windows //

  virtual void buildUI() = 0;
  virtual void saveSettings(tsd::core::DataNode &thisWindowRoot);
  virtual void loadSettings(tsd::core::DataNode &thisWindowRoot);

 protected:
  virtual int windowFlags() const;
  tsd::app::Context *appContext() const;

  Application *m_app{nullptr};
  std::string m_name;
  bool m_visible{true};
};

} // namespace tsd::ui::imgui
