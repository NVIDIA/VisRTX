// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "ConfirmDiscardDialog.h"

#include "imgui.h"

namespace tsd::scivis_studio {

ConfirmDiscardDialog::ConfirmDiscardDialog(tsd::ui::imgui::Application *app)
    : Modal(app, "Discard Unsaved Changes")
{}

ConfirmDiscardDialog::~ConfirmDiscardDialog() = default;

void ConfirmDiscardDialog::configure(
    std::function<void()> onDiscard, std::function<void()> onCancel)
{
  m_onDiscard = std::move(onDiscard);
  m_onCancel = std::move(onCancel);
}

void ConfirmDiscardDialog::buildUI()
{
  ImGui::TextUnformatted("The current project has unsaved changes.");
  ImGui::Spacing();

  if (ImGui::Button("Cancel")) {
    hide();
    if (m_onCancel)
      m_onCancel();
  }

  ImGui::SameLine();

  if (ImGui::Button("Discard and Continue")) {
    hide();
    if (m_onDiscard)
      m_onDiscard();
  }
}

} // namespace tsd::scivis_studio
