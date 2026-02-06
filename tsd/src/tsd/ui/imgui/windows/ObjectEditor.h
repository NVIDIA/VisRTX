// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Window.h"

namespace tsd::ui::imgui {

struct ObjectEditor : public Window
{
  ObjectEditor(Application *app, const char *name = "Object Editor");
  void buildUI() override;
};

} // namespace tsd::ui::imgui
