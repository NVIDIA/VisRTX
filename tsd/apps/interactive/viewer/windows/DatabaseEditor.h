// Copyright 2024 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "AppCore.h"
// anari_viewer
#include "anari_viewer/windows/Window.h"

namespace tsd_viewer {

struct DatabaseEditor : public anari_viewer::windows::Window
{
  DatabaseEditor(AppCore *ctx, const char *name = "Database Editor");
  void buildUI() override;
 private:
  AppCore *m_core{nullptr};
};

} // namespace tsd_viewer
