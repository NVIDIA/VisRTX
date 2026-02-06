// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
// imgui
#include "imgui.h"

#include "tsd/app/Core.h"

namespace tsd::ui::imgui {

class Application;

struct Modal
{
  Modal(Application *app, const char *name);
  virtual ~Modal();

  void renderUI();

  void show();
  void hide();
  bool visible() const;

  const char *name() const;

 protected:
  virtual void buildUI() = 0;
  tsd::app::Core *appCore() const;

  Application *m_app{nullptr};

 private:
  std::string m_name;
  bool m_visible{false};
};

} // namespace tsd::ui::imgui
