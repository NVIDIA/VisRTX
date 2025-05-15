// Copyright 2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "Window.h"
#include "../AppCore.h"

namespace tsd_viewer {

Window::~Window() = default;

Window::Window(AppCore *core, const char *name)
    : anari_viewer::windows::Window(core->application, name, true), m_core(core)
{}

void Window::saveSettings(tsd::serialization::DataNode &thisWindowRoot)
{
  thisWindowRoot["visible"] = *visiblePtr();
}

void Window::loadSettings(tsd::serialization::DataNode &thisWindowRoot)
{
  thisWindowRoot["visible"].getValue(ANARI_BOOL, visiblePtr());
}

} // namespace tsd_viewer
