// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
// imgui
#include "imgui.h"

#include "tsd/app/Core.h"

namespace tsd::ui::imgui {

struct Modal
{
  Modal(tsd::app::Core *core, const char *name);
  virtual ~Modal() = default;

  void renderUI();

  void show();
  void hide();
  bool visible() const;

  const char *name() const;

 protected:
  virtual void buildUI() = 0;

  tsd::app::Core *m_core{nullptr};

 private:
  std::string m_name;
  bool m_visible{false};
};

// Inlined definitions ////////////////////////////////////////////////////////

inline Modal::Modal(tsd::app::Core *core, const char *name)
    : m_core(core), m_name(name)
{}

inline void Modal::renderUI()
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

inline void Modal::show()
{
  m_visible = true;
}

inline void Modal::hide()
{
  m_visible = false;
}

inline bool Modal::visible() const
{
  return m_visible;
}

inline const char *Modal::name() const
{
  return m_name.c_str();
}

} // namespace tsd::ui::imgui
