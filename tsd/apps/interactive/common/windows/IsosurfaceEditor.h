// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Window.h"

namespace tsd_viewer {

struct IsosurfaceEditor : public Window
{
  IsosurfaceEditor(AppCore *state, const char *name = "Isosurface Editor");
  void buildUI() override;

 private:
  void addIsosurfaceGeometryFromSelected();
};

} // namespace tsd_viewer
