// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Window.h"

namespace tsd_viewer {

struct CameraPoses : public Window
{
  CameraPoses(AppCore *state, const char *name = "Camera Poses");
  void buildUI() override;
};

} // namespace tsd_viewer
