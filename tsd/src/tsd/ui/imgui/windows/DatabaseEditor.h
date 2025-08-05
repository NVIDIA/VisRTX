// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Window.h"

namespace tsd_viewer {

struct DatabaseEditor : public Window
{
  DatabaseEditor(AppCore *ctx, const char *name = "Database Editor");
  void buildUI() override;
};

} // namespace tsd_viewer
