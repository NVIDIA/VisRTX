// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "ConfirmDefaultLayoutDialog.h"

#include "imgui.h"

namespace tsd::scivis_studio {

ConfirmDefaultLayoutDialog::ConfirmDefaultLayoutDialog(
    tsd::ui::imgui::Application *app)
    : Modal(app, "Update Default Layout")
{}

ConfirmDefaultLayoutDialog::~ConfirmDefaultLayoutDialog() = default;

void ConfirmDefaultLayoutDialog::configure(std::function<void()> onConfirm)
{
  m_onConfirm = std::move(onConfirm);
}

void ConfirmDefaultLayoutDialog::buildUI()
{
  ImGui::Dummy(ImVec2(700.f, 0.f));
  ImGui::TextUnformatted("Are you sure?");
  ImGui::Spacing();

  if (ImGui::Button("No"))
    hide();

  ImGui::SameLine();

  if (ImGui::Button("Yes")) {
    hide();
    if (m_onConfirm)
      m_onConfirm();
  }
}

} // namespace tsd::scivis_studio
