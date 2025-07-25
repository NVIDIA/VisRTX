// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Modal.h"

namespace tsd_viewer {

struct AppSettingsDialog : public Modal
{
  AppSettingsDialog(AppCore *core);
  ~AppSettingsDialog() override = default;

  void buildUI() override;
  void applySettings();

  private:
  void buildUI_applicationSettings();
  void buildUI_offlineRenderSettings();
};

} // namespace tsd_viewer
