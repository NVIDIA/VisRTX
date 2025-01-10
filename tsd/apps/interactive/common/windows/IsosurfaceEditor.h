// Copyright 2024 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "AppCore.h"
// anari_viewer
#include "anari_viewer/windows/Window.h"

namespace tsd_viewer {

struct IsosurfaceEditor : public anari_viewer::windows::Window
{
  IsosurfaceEditor(AppCore *state, const char *name = "Isosurface Editor");
  void buildUI() override;

 private:
  void addIsosurfaceGeometryFromSelected();

  AppCore *m_core{nullptr};
};

} // namespace tsd_viewer
