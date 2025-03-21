// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "AppCore.h"
// anari_viewer
#include "anari_viewer/windows/Window.h"

namespace tsd_viewer {

struct ObjectEditor : public anari_viewer::windows::Window
{
  ObjectEditor(AppCore *state, const char *name = "Object Editor");
  void buildUI() override;
 private:
  AppCore *m_core{nullptr};
};

} // namespace tsd_viewer
