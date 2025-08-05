// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Window.h"

namespace tsd_viewer {

struct ObjectEditor : public Window
{
  ObjectEditor(AppCore *state, const char *name = "Object Editor");
  void buildUI() override;
};

} // namespace tsd_viewer
