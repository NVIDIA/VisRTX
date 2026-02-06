// Copyright 2023-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// tsd_ui_imgui
#include <tsd/ui/imgui/windows/Window.h>
// imnodes
#include <imnodes.h>
// viskores_graph
#include <viskores_graph/ExecutionGraph.h>

#include "NodeInfoWindow.h"

namespace tsd::viskores_graph {

namespace graph = viskores::graph;

class NodeEditor : public tsd::ui::imgui::Window
{
 public:
  NodeEditor(tsd::ui::imgui::Application *app,
      graph::ExecutionGraph *graph,
      tsd::viskores_graph::NodeInfoWindow *nodeInfoWindow);

  void buildUI() override;
  void updateNodeSummary();

 private:
  void contextMenu();
  void contextMenuPin();

  void editor_Node(graph::Node *n);

  int m_summarizedNodeID{-1};
  int m_prevNumSelectedNodes{-1};
  int m_pinHoverId{-1};

  bool m_contextMenuVisible{false};
  bool m_contextPinMenuVisible{false};
  graph::ExecutionGraph *m_graph{nullptr};
  graph::TimeStamp m_lastGraphChange{};
  tsd::viskores_graph::NodeInfoWindow *m_nodeInfoWindow{nullptr};
};

} // namespace tsd::viskores_graph
