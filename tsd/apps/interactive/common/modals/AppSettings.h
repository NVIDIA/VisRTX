// Copyright 2024 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Modal.h"

namespace tsd_viewer {

struct AppSettings : public Modal
{
  AppSettings();
  ~AppSettings() override = default;

  void buildUI() override;

 private:
  void update();

  float m_fontScale{1.25f};
};

} // namespace tsd_viewer
