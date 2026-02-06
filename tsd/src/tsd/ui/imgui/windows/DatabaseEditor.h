// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Window.h"

namespace tsd::ui::imgui {

struct DatabaseEditor : public Window
{
  DatabaseEditor(Application *app, const char *name = "Database Editor");
  void buildUI() override;
};

} // namespace tsd::ui::imgui
