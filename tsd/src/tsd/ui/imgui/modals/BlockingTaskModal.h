// Copyright 2025-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Modal.h"
// tsd_core
#include "tsd/core/TaskQueue.hpp"
#include "tsd/core/Timer.hpp"
// std
#include <deque>
#include <string>
#include <mutex>

namespace tsd::ui::imgui {

struct BlockingTaskModal : public Modal
{
  BlockingTaskModal(Application *app);
  ~BlockingTaskModal() override;

  void buildUI() override;

  void activate(tsd::core::Future &&f, const char *text = "Please Wait");

 private:
  struct RunningTask
  {
    tsd::core::Future future;
    std::string text;
  };

  std::deque<RunningTask> m_tasks;
  tsd::core::Timer m_timer;
  std::mutex m_mutex;
};

} // namespace tsd::ui::imgui
