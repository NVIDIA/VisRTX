// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Window.h"

namespace tsd::ui::imgui {

struct IsosurfaceEditor : public Window
{
  IsosurfaceEditor(Application *app, const char *name = "Isosurface Editor");
  void buildUI() override;

 private:
  void addIsosurfaceGeometryFromSelected();
};

} // namespace tsd::ui::imgui
