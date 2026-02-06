// Copyright 2025-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "Modal.h"
#include "tsd/ui/imgui/Application.h"

namespace tsd::ui::imgui {

Modal::Modal(Application *app, const char *name) : m_app(app), m_name(name) {}

Modal::~Modal() = default;

void Modal::renderUI()
{
  if (!m_visible)
    return;

  ImGuiIO &io = ImGui::GetIO();
  ImGui::SetNextWindowPos(
      ImVec2(io.DisplaySize.x * 0.5f, io.DisplaySize.y * 0.5f),
      ImGuiCond_Always,
      ImVec2(0.5f, 0.5f));

  ImGui::OpenPopup(m_name.c_str());
  if (ImGui::BeginPopupModal(
          m_name.c_str(), &m_visible, ImGuiWindowFlags_AlwaysAutoResize)) {
    buildUI();
    ImGui::EndPopup();
  }
}

void Modal::show()
{
  m_visible = true;
}

void Modal::hide()
{
  m_visible = false;
}

bool Modal::visible() const
{
  return m_visible;
}

const char *Modal::name() const
{
  return m_name.c_str();
}

tsd::app::Core *Modal::appCore() const
{
  return m_app ? m_app->appCore() : nullptr;
}

} // namespace tsd::ui::imgui
