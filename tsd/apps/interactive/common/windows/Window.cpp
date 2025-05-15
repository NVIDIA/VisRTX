// Copyright 2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "Window.h"
#include "../AppCore.h"

namespace tsd_viewer {

Window::~Window() = default;

Window::Window(AppCore *core, const char *name)
    : anari_viewer::windows::Window(core->application, name, true), m_core(core)
{}

} // namespace tsd_viewer
