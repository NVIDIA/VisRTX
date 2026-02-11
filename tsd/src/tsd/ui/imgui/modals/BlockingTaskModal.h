// Copyright 2025-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Modal.h"
// tsd_core
#include "tsd/core/TaskQueue.hpp"
#include "tsd/core/Timer.hpp"

namespace tsd::ui::imgui {

struct BlockingTaskModal : public Modal
{
  BlockingTaskModal(Application *app);
  ~BlockingTaskModal() override;

  void buildUI() override;

  void activate(tsd::core::Future &&f, const char *text = "Please Wait");

 private:
  tsd::core::Future m_future;
  std::string m_text;
  tsd::core::Timer m_timer;
};

} // namespace tsd::ui::imgui
