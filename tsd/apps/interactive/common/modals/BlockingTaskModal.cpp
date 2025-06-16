// Copyright 2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "BlockingTaskModal.h"

namespace tsd_viewer {

BlockingTaskModal::BlockingTaskModal(AppCore *core)
    : Modal(core, "##"), m_core(core)
{}

BlockingTaskModal::~BlockingTaskModal() = default;

void BlockingTaskModal::buildUI()
{
  if (tasking::isReady(m_future))
    this->hide();

  ImGui::ProgressBar(
      -1.0f * (float)ImGui::GetTime(), ImVec2(0.0f, 0.0f), m_text.c_str());
}

} // namespace tsd_viewer
