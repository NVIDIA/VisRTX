// Copyright 2025-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Modal.h"
// tsd_core
#include "tsd/core/Timer.hpp"

namespace tsd::ui::imgui {

struct BlockingTaskModal : public Modal
{
  BlockingTaskModal(Application *app);
  ~BlockingTaskModal() override;

  void buildUI() override;

  template <class F>
  void activate(F &&f, const char *text = "Please Wait");

 private:
  tsd::app::Future m_future;
  std::string m_text;
  tsd::core::Timer m_timer;
};

// Inlined definitions ////////////////////////////////////////////////////////

template <class F>
inline void BlockingTaskModal::activate(F &&f, const char *text)
{
  m_timer.start();
  m_future = appCore()->jobs.queue.enqueue(std::move(f));
  m_text = text;
  this->show();
}

} // namespace tsd::ui::imgui
