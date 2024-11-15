// Copyright 2024 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "AppContext.h"
// anari_viewer
#include "anari_viewer/windows/Window.h"

namespace tsd_viewer {

struct IsosurfaceEditor : public anari_viewer::windows::Window
{
  IsosurfaceEditor(AppContext *state, const char *name = "Isosurface Editor");
  void buildUI() override;

 private:
  void addIsosurfaceGeometryFromSelected();

  AppContext *m_context{nullptr};
};

} // namespace tsd_viewer
