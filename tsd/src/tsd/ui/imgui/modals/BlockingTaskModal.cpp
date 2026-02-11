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
  if (tsd::core::isReady(m_future))
    this->hide();

  ImGui::ProgressBar(
      -1.0f * (float)ImGui::GetTime(), ImVec2(0.0f, 0.0f), m_text.c_str());

  m_timer.end();
  ImGui::NewLine();
  ImGui::TextDisabled("elapsed time: %.2fs", m_timer.seconds());
}

void BlockingTaskModal::activate(tsd::core::Future &&f, const char *text)
{
  m_timer.start();
  m_future = std::move(f);
  m_text = text;
  this->show();
}

} // namespace tsd::ui::imgui
