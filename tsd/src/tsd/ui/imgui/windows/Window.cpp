// Copyright 2025-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "Window.h"
#include "tsd/ui/imgui/Application.h"
// imgui
#define IMGUI_DISABLE_INCLUDE_IMCONFIG_H
#include <imgui.h>

namespace tsd::ui::imgui {

Window::Window(Application *app, const char *name)
    : m_app(app), m_name(name)
{}

Window::~Window() = default;

void Window::renderUI()
{
  if (!m_visible)
    return;

  ImGui::SetNextWindowSize(ImVec2(550, 680), ImGuiCond_FirstUseEver);
  ImGui::Begin(m_name.c_str(), &m_visible, windowFlags());
  buildUI();
  ImGui::End();
}

void Window::show()
{
  m_visible = true;
}

void Window::hide()
{
  m_visible = false;
}

void Window::toggleShown()
{
  m_visible = !m_visible;
}

bool *Window::visiblePtr()
{
  return &m_visible;
}

const char *Window::name()
{
  return m_name.c_str();
}

void Window::saveSettings(tsd::core::DataNode &thisWindowRoot)
{
  thisWindowRoot["visible"] = *visiblePtr();
}

void Window::loadSettings(tsd::core::DataNode &thisWindowRoot)
{
  thisWindowRoot["visible"].getValue(ANARI_BOOL, visiblePtr());
}

ImGuiWindowFlags Window::windowFlags() const
{
  return 0;
}

tsd::app::Context *Window::appContext() const
{
  return m_app ? m_app->appContext() : nullptr;
}

} // namespace tsd::ui::imgui
