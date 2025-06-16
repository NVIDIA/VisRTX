// Copyright 2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../AppCore.h"
#include "Modal.h"

namespace tsd_viewer {

struct BlockingTaskModal : public Modal
{
  BlockingTaskModal(AppCore *ctx);
  ~BlockingTaskModal() override;

  void buildUI() override;

  template <class F>
  void activate(F &&f, const char *text = "Please Wait");

 private:
  AppCore *m_core{nullptr};
  tasking::Future m_future;
  std::string m_text;
};

// Inlined definitions ////////////////////////////////////////////////////////

template <class F>
inline void BlockingTaskModal::activate(F &&f, const char *text)
{
  m_future = m_core->jobs.queue.enqueue(std::move(f));
  m_text = text;
  this->show();
}

} // namespace tsd_viewer
