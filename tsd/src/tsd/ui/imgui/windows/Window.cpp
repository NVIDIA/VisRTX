// Copyright 2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "Window.h"
#include "tsd/ui/imgui/Application.h"
// tsd_app
#include "tsd/app/Core.h"

namespace tsd::ui::imgui {

Window::~Window() = default;

Window::Window(Application *app, const char *name)
    : anari_viewer::windows::Window(app, name, true), m_app(app)
{}

void Window::saveSettings(tsd::core::DataNode &thisWindowRoot)
{
  thisWindowRoot["visible"] = *visiblePtr();
}

void Window::loadSettings(tsd::core::DataNode &thisWindowRoot)
{
  thisWindowRoot["visible"].getValue(ANARI_BOOL, visiblePtr());
}

tsd::app::Core *Window::appCore() const
{
  return m_app ? m_app->appCore() : nullptr;
}

} // namespace tsd::ui::imgui
