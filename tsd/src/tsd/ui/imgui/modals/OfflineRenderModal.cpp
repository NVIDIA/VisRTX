// Copyright 2025-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "OfflineRenderModal.h"
#include "tsd/ui/imgui/Application.h"
// tsd_core
#include "tsd/core/Logging.hpp"

namespace tsd::ui::imgui {

OfflineRenderModal::OfflineRenderModal(Application *app)
    : Modal(app, "##offline_render_modal")
{}

OfflineRenderModal::~OfflineRenderModal() = default;

void OfflineRenderModal::buildUI()
{
  if (tsd::core::isReady(m_future))
    this->hide();

  ImGui::ProgressBar(
      -1.0f * (float)ImGui::GetTime(), ImVec2(0.0f, 0.0f), "Rendering...");

  m_timer.end();
  ImGui::NewLine();
  ImGui::TextDisabled("elapsed time: %.2fs", m_timer.seconds());

  ImGui::Separator();
  ImGui::BeginDisabled(m_canceled);
  if (ImGui::Button("cancel"))
    m_canceled = true;
  ImGui::EndDisabled();
}

void OfflineRenderModal::start()
{
  m_timer.start();
  m_canceled = false;
  this->show();
  auto *core = appCore();
  m_future = m_app->enqueueTask([this, core]() {
    tsd::app::renderAnimationSequence(*core,
        core->offline.output.outputDirectory,
        core->offline.output.filePrefix,
        [&](int frameIndex, int numFrames) {
          tsd::core::logStatus("[OfflineRenderModal] Rendering frame %d of %d",
              frameIndex + 1,
              numFrames);
          return !m_canceled;
        });
  });
}

} // namespace tsd::ui::imgui
