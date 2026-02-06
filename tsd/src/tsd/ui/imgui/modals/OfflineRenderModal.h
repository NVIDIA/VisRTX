// Copyright 2025-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Modal.h"
// tsd_core
#include "tsd/core/Timer.hpp"

namespace tsd::ui::imgui {

struct OfflineRenderModal : public Modal
{
  OfflineRenderModal(Application *app);
  ~OfflineRenderModal() override;

  void buildUI() override;
  void start();

 private:
  tsd::app::Future m_future;
  std::string m_text;
  tsd::core::Timer m_timer;
  bool m_canceled{false};
};

} // namespace tsd::ui::imgui
