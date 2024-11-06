// Copyright 2024 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "AppContext.h"
// anari_viewer
#include "anari_viewer/windows/Window.h"

namespace tsd_viewer {

struct ObjectEditor : public anari_viewer::windows::Window
{
  ObjectEditor(AppContext *state, const char *name = "Object Editor");
  void buildUI() override;
 private:
  AppContext *m_context{nullptr};
};

} // namespace tsd_viewer
