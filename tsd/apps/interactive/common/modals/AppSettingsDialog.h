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

 private:
  void update();

  float m_fontScale{1.25f};
};

} // namespace tsd_viewer
