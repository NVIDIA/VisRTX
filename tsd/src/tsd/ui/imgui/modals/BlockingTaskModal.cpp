// Copyright 2025-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "BlockingTaskModal.h"

namespace tsd::ui::imgui {

BlockingTaskModal::BlockingTaskModal(Application *app)
    : Modal(app, "##blocking_task_modal")
{}

BlockingTaskModal::~BlockingTaskModal() = default;

void BlockingTaskModal::buildUI()
{
  std::lock_guard<std::mutex> lock(m_mutex);

  if (m_tasks.empty()) {
    this->hide();
    return;
  }

  auto *t = &m_tasks.front();
  while (tsd::core::isReady(t->future)) {
    m_tasks.pop_front();
    if (m_tasks.empty()) {
      this->hide();
      return;
    }
    t = &m_tasks.front();
  }

  ImGui::ProgressBar(
      -1.0f * (float)ImGui::GetTime(), ImVec2(0.0f, 0.0f), t->text.c_str());

  m_timer.end();
  ImGui::NewLine();
  ImGui::TextDisabled("elapsed time: %.2fs", m_timer.seconds());
}

void BlockingTaskModal::activate(tsd::core::Future &&f, const char *text)
{
  std::lock_guard<std::mutex> lock(m_mutex);
  if (m_tasks.empty())
    m_timer.start();
  m_tasks.push_back({std::move(f), text});
  this->show();
}

} // namespace tsd::ui::imgui
