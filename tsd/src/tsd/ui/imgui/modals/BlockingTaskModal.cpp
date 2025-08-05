// Copyright 2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "BlockingTaskModal.h"

namespace tsd::ui::imgui {

BlockingTaskModal::BlockingTaskModal(tsd::app::Core *core)
    : Modal(core, "##"), m_core(core)
{}

BlockingTaskModal::~BlockingTaskModal() = default;

void BlockingTaskModal::buildUI()
{
  if (tsd::app::isReady(m_future))
    this->hide();

  ImGui::ProgressBar(
      -1.0f * (float)ImGui::GetTime(), ImVec2(0.0f, 0.0f), m_text.c_str());

  m_timer.end();
  ImGui::NewLine();
  ImGui::TextDisabled("elapsed time: %.2fs", m_timer.seconds());
}

} // namespace tsd::ui::imgui
