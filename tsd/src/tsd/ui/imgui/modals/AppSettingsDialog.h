// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Modal.h"

namespace tsd::ui::imgui {

struct AppSettingsDialog : public Modal
{
  AppSettingsDialog(Application *app);
  ~AppSettingsDialog() override = default;

  void buildUI() override;
  void applySettings();

 private:
  void buildUI_applicationSettings();
  void buildUI_offlineRenderSettings();

  std::vector<tsd::core::CameraRef> m_menuCameraRefs;
};

} // namespace tsd::ui::imgui
