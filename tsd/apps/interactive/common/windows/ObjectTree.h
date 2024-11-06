// Copyright 2024 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "AppContext.h"
// anari_viewer
#include "anari_viewer/windows/Window.h"

namespace tsd_viewer {

struct ObjectTree : public anari_viewer::windows::Window
{
  ObjectTree(AppContext *state, const char *name = "Object Tree");
  void buildUI() override;

 private:
  void buildUI_objectContextMenu();

  AppContext *m_context{nullptr};
  size_t m_hoveredNode{tsd::INVALID_INDEX};
  size_t m_contextMenuNode{tsd::INVALID_INDEX};
  bool m_contextMenuVisible{false};
  std::vector<int> m_needToTreePop;
};

} // namespace tsd_viewer
