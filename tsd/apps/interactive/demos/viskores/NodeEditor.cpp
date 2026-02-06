// Copyright 2023-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "NodeEditor.h"
// viskores
#include <viskores/cont/DataSetBuilderUniform.h>
// std
#include <algorithm>
#include <random>

namespace tsd::viskores_graph {

static constexpr uint32_t outLinkIDMask = 0xFFFF0000;
static constexpr uint32_t inLinkIDMask = 0x0000FFFF;

static constexpr float sourceNodeWidth = 220.f;
static constexpr float filterNodeWidth = 300.f;
static constexpr float actorNodeWidth = 300.f;
static constexpr float utilityNodeWidth = 300.f;

namespace colors {
// clang-format off
static constexpr auto Red          = IM_COL32(191, 11, 11, 255);
static constexpr auto RedSelect    = IM_COL32(204, 81, 81, 255);
static constexpr auto Orange       = IM_COL32(191, 109, 11, 255);
static constexpr auto OrangeSelect = IM_COL32(204, 148, 81, 255);
static constexpr auto Green        = IM_COL32(11, 191, 11, 255);
static constexpr auto GreenSelect  = IM_COL32(81, 204, 81, 255);
static constexpr auto Blue         = IM_COL32(11, 109, 191, 255);
static constexpr auto BlueSelect   = IM_COL32(81, 148, 204, 255);
static constexpr auto Purple       = IM_COL32(81, 11, 191, 255);
static constexpr auto PurpleSelect = IM_COL32(191, 81, 204, 255);
// clang-format on
} // namespace colors

static viskores::cont::DataSet makeNoiseVolume_callback()
{
  constexpr int size = 64;

  viskores::cont::DataSetBuilderUniform builder;
  auto dataset = builder.Create(viskores::Id3(size),
      viskores::Vec3f(-1.f),
      viskores::Vec3f(2.f / viskores::Float32(size)));

  viskores::cont::ArrayHandle<viskores::Float32> field;
  field.Allocate(dataset.GetCoordinateSystem().GetNumberOfValues());

  {
    std::mt19937 rng;
    rng.seed(0);
    std::normal_distribution<float> dist(0.f, 10.0f);

    viskores::cont::Token token;

    auto *voxelsBegin = (float *)field.GetBuffers()[0].WritePointerHost(token);
    auto *voxelsEnd = voxelsBegin + field.GetNumberOfValues();

    std::for_each(voxelsBegin, voxelsEnd, [&](auto &v) { v = dist(rng); });
  }

  dataset.AddField(viskores::cont::Field(
      "noise", viskores::cont::Field::Association::Points, field));

  return dataset;
}

static bool fieldSelectorSelection_callback(
    void *_fs, int index, const char **out_text)
{
  const auto *fs = (const graph::FieldSelector *)_fs;
  *out_text = fs->fieldName(index);
  return true;
}

static int editor_GetPinShape(graph::PortType type)
{
  switch (type) {
  case graph::PortType::ACTOR:
    return ImNodesPinShape_TriangleFilled;
  case graph::PortType::DATASET:
    return ImNodesPinShape_CircleFilled;
  default:
    break;
  }

  return ImNodesPinShape_QuadFilled;
}

static int editor_MakeLinkID(int outID, int inID)
{
  return ((outID << 16) & outLinkIDMask) + (inID & inLinkIDMask);
}

static void editor_Spacer()
{
  ImGui::Dummy(ImVec2(10.0f, 10.0f));
}

static void editor_NodeParameter(graph::ExecutionGraph *g, graph::Parameter &p)
{
  if (p.type() == graph::ParameterType::BOOL) {
    auto value = p.valueAs<bool>();
    if (ImGui::Checkbox(p.name(), &value))
      g->scheduleParameterUpdate(p, value);
  } else if (p.type() == graph::ParameterType::BOUNDED_FLOAT) {
    ImGui::BeginDisabled(!p.hasMinMax());
    ImGui::PushItemWidth(150.f);
    auto value = p.valueAs<float>();
    if (ImGui::SliderFloat(
            p.name(), &value, p.minAs<float>(), p.maxAs<float>()))
      g->scheduleParameterUpdate(p, value);
    ImGui::PopItemWidth();
    ImGui::EndDisabled();
  } else if (p.type() == graph::ParameterType::BOUNDED_INT) {
    ImGui::BeginDisabled(!p.hasMinMax());
    ImGui::PushItemWidth(150.f);
    auto value = p.valueAs<int>();
    if (ImGui::SliderInt(p.name(), &value, p.minAs<int>(), p.maxAs<int>()))
      g->scheduleParameterUpdate(p, value);
    ImGui::PopItemWidth();
    ImGui::EndDisabled();
  } else if (p.type() == graph::ParameterType::FLOAT) {
    ImGui::PushItemWidth(150.f);
    auto value = p.valueAs<float>();
    if (ImGui::InputFloat(p.name(), &value))
      g->scheduleParameterUpdate(p, value);
    ImGui::PopItemWidth();
  } else if (p.type() == graph::ParameterType::INT) {
    ImGui::PushItemWidth(150.f);
    auto value = p.valueAs<int>();
    if (ImGui::InputInt(p.name(), &value))
      g->scheduleParameterUpdate(p, value);
    ImGui::PopItemWidth();
  } else if (p.type() == graph::ParameterType::FILENAME) {
    ImGui::PushItemWidth(150.f);
    auto value = p.valueAs<std::string>();
    std::vector<char> buf(value.size() + 1 + 256);
    strcpy(buf.data(), value.c_str());
    buf[value.size()] = '\0';
    if (ImGui::InputText(p.name(),
            buf.data(),
            value.size() + 256,
            ImGuiInputTextFlags_EnterReturnsTrue)) {
      g->scheduleParameterUpdate(p, std::string(buf.data()));
    }
    ImGui::PopItemWidth();
  } else
    ImGui::Text("param: %s", p.name());
}

static void editor_NodeParameters(graph::ExecutionGraph *g, graph::Node *n)
{
  std::for_each(n->parametersBegin(), n->parametersEnd(), [&](auto &p) {
    editor_NodeParameter(g, p);
  });
}

static void editor_NodeTitle(graph::Node *n)
{
  ImNodes::BeginNodeTitleBar();
  ImGui::Text("%s", n->name());
  ImNodes::EndNodeTitleBar();
}

static void editor_NodeInputPorts(graph::Node *n)
{
  std::for_each(n->inputBegin(), n->inputEnd(), [](auto &ip) {
    ImNodes::BeginInputAttribute(ip.id(), editor_GetPinShape(ip.type()));
    ImGui::Text("%s", ip.name());
    ImNodes::EndInputAttribute();
  });
}

static void editor_NodeOutputPorts(graph::Node *node)
{
  std::for_each(node->outputBegin(), node->outputEnd(), [](auto &op) {
    ImNodes::BeginOutputAttribute(op.id(), editor_GetPinShape(op.type()));
    auto indentWidth = 0.f;
    auto *n = op.node();
    if (n->type() == graph::NodeType::SOURCE)
      indentWidth = sourceNodeWidth;
    else if (n->type() == graph::NodeType::FILTER)
      indentWidth = filterNodeWidth;
    else if (n->type() == graph::NodeType::ACTOR)
      indentWidth = actorNodeWidth;
    else if (n->type() == graph::NodeType::UTILITY)
      indentWidth = actorNodeWidth;
    if (indentWidth > 0.f)
      ImGui::Indent(indentWidth - ImGui::CalcTextSize(op.name()).x);
    ImGui::Text("%s", op.name());
    ImNodes::EndOutputAttribute();
  });
}

static void editor_NodeInputPortLinks(graph::Node *n)
{
  std::for_each(n->inputBegin(), n->inputEnd(), [](auto &ip) {
    if (ip.isConnected()) {
      auto *otherPort = ip.other();
      auto inID = ip.id();
      auto outID = otherPort->id();
      ImNodes::Link(editor_MakeLinkID(outID, inID), outID, inID);
    }
  });
}

static bool editor_ActorNodeFields(graph::ActorNode *n)
{
  ImGui::PushItemWidth(200.f);
  const auto &cs = n->cselector();
  int whichField = int(cs.currentField());
  if (ImGui::Combo("field",
          &whichField,
          fieldSelectorSelection_callback,
          (void *)&cs,
          cs.numFields())) {
    n->selector().setCurrentField(whichField);
    return true;
  }
  ImGui::PopItemWidth();
  return false;
}

void NodeEditor::editor_Node(graph::Node *n)
{
  auto type = n->type();

  switch (type) {
  case graph::NodeType::FILTER:
    ImNodes::PushColorStyle(ImNodesCol_TitleBar, colors::Red);
    ImNodes::PushColorStyle(ImNodesCol_TitleBarSelected, colors::RedSelect);
    break;
  case graph::NodeType::SOURCE:
    ImNodes::PushColorStyle(ImNodesCol_TitleBar, colors::Green);
    ImNodes::PushColorStyle(ImNodesCol_TitleBarSelected, colors::GreenSelect);
    break;
  case graph::NodeType::ACTOR:
    ImNodes::PushColorStyle(ImNodesCol_TitleBar, colors::Blue);
    ImNodes::PushColorStyle(ImNodesCol_TitleBarSelected, colors::BlueSelect);
    break;
  case graph::NodeType::MAPPER:
    ImNodes::PushColorStyle(ImNodesCol_TitleBar, colors::Orange);
    ImNodes::PushColorStyle(ImNodesCol_TitleBarSelected, colors::OrangeSelect);
    break;
  case graph::NodeType::UTILITY:
    ImNodes::PushColorStyle(ImNodesCol_TitleBar, colors::Purple);
    ImNodes::PushColorStyle(ImNodesCol_TitleBarSelected, colors::PurpleSelect);
    break;
  default:
    break;
  }

  ImNodes::BeginNode(n->id());
  editor_NodeTitle(n);
  editor_NodeOutputPorts(n);
  editor_Spacer();
  editor_NodeParameters(m_graph, n);
  editor_Spacer();
  if (type == graph::NodeType::ACTOR)
    editor_ActorNodeFields((graph::ActorNode *)n);
  editor_Spacer();
  editor_NodeInputPorts(n);
  ImNodes::EndNode();

  ImNodes::PopColorStyle();
  ImNodes::PopColorStyle();

  editor_NodeInputPortLinks(n);
}

///////////////////////////////////////////////////////////////////////////////
// NodeEditor definitions /////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

NodeEditor::NodeEditor(tsd::ui::imgui::Application *app,
    graph::ExecutionGraph *graph,
    tsd::viskores_graph::NodeInfoWindow *nodeInfoWindow)
    : tsd::ui::imgui::Window(app, "Node Editor"),
      m_graph(graph),
      m_nodeInfoWindow(nodeInfoWindow)
{}

void NodeEditor::buildUI()
{
  ImNodes::BeginNodeEditor();

  updateNodeSummary();

  contextMenu();

  for (size_t i = 0; i < m_graph->getNumberOfNodes(); i++)
    editor_Node(m_graph->getNode(i));

  ImNodes::MiniMap();
  ImNodes::EndNodeEditor();

  {
    if (!m_contextPinMenuVisible && ImNodes::IsPinHovered(&m_pinHoverId)) {
      auto *port = graph::Port::fromID(m_pinHoverId);
      const bool enablePinContextMenu = graph::isInputPortID(m_pinHoverId)
          && port->type() == graph::PortType::DATASET;

      if (enablePinContextMenu
          && ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
        m_contextPinMenuVisible = true;
        ImGui::OpenPopup("pinEditorContextMenu");
      }

      ImGui::BeginTooltip();
      ImGui::Text("type: [%s]", graph::portTypeString(port->type()));
      if (enablePinContextMenu)
        ImGui::Text("(right click to edit active field)");
      ImGui::EndTooltip();
    }
  }

  contextMenuPin();

  int startID, endID;
  if (ImNodes::IsLinkCreated(&startID, &endID)) {
    auto *outPort = graph::OutPort::fromID(startID);
    auto *inPort = graph::InPort::fromID(endID);
    if (outPort && inPort)
      graph::connect(outPort, inPort);
  }

  const bool doDelete =
      ImGui::IsKeyReleased(ImGuiKey_Delete) || ImGui::IsKeyReleased(ImGuiKey_X);

  int numSelected = ImNodes::NumSelectedLinks();
  if (numSelected > 0 && doDelete) {
    static std::vector<int> selectedLinks;
    selectedLinks.resize(numSelected);
    ImNodes::GetSelectedLinks(selectedLinks.data());
    for (const int linkID : selectedLinks)
      graph::InPort::fromID(linkID & inLinkIDMask)->disconnect();
    ImNodes::ClearLinkSelection();
  }

  numSelected = ImNodes::NumSelectedNodes();
  if (numSelected > 0 && doDelete) {
    static std::vector<int> selectedNodes;
    selectedNodes.resize(numSelected);
    ImNodes::GetSelectedNodes(selectedNodes.data());
    for (const int nodeID : selectedNodes)
      m_graph->removeNode(nodeID);
    ImNodes::ClearNodeSelection();
  }
}

void NodeEditor::contextMenu()
{
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(10.f, 10.f));

  ImGuiIO &io = ImGui::GetIO();
  const bool openMenu = ImGui::IsKeyChordPressed(ImGuiMod_Shift | ImGuiKey_A);

  static auto mousePos = ImGui::GetMousePos();

  if (openMenu && ImNodes::IsEditorHovered()) {
    m_contextMenuVisible = true;
    mousePos = ImGui::GetMousePos();
    ImGui::OpenPopup("nodeEditorContextMenu");
  }

  if (ImGui::BeginPopup("nodeEditorContextMenu")) {
    if (ImGui::BeginMenu("add node")) {
      graph::Node *addedNode = nullptr;

      if (ImGui::BeginMenu("source")) {
        if (ImGui::MenuItem("ABC"))
          addedNode = m_graph->addNode<graph::ABCSourceNode>();
        if (ImGui::MenuItem("noise")) {
          auto *node = m_graph->addNode<graph::CallbackSourceNode>(
              makeNoiseVolume_callback);
          addedNode = node;
          node->setName("Noise");
        }
        if (ImGui::MenuItem("random points"))
          addedNode = m_graph->addNode<graph::RandomPointsSourceNode>();
        if (ImGui::MenuItem("tangle"))
          addedNode = m_graph->addNode<graph::TangleSourceNode>();
        if (ImGui::MenuItem("VTK file reader"))
          addedNode = m_graph->addNode<graph::VTKFileReaderNode>();
        if (ImGui::MenuItem("empty")) {
          auto *node = m_graph->addNode<graph::CallbackSourceNode>(
              []() -> viskores::cont::DataSet { return {}; });
          addedNode = node;
          node->setName("Empty");
        }
        ImGui::EndMenu();
      }

      ImGui::Separator();

      if (ImGui::BeginMenu("filter")) {
        if (ImGui::MenuItem("cell average"))
          addedNode = m_graph->addNode<graph::CellAverageNode>();
        if (ImGui::MenuItem("clean grid"))
          addedNode = m_graph->addNode<graph::CleanGridNode>();
        if (ImGui::MenuItem("contour"))
          addedNode = m_graph->addNode<graph::ContourNode>();
        if (ImGui::MenuItem("gradient"))
          addedNode = m_graph->addNode<graph::GradientNode>();
        if (ImGui::MenuItem("point average"))
          addedNode = m_graph->addNode<graph::PointAverageNode>();
        if (ImGui::MenuItem("probe"))
          addedNode = m_graph->addNode<graph::ProbeNode>();
        if (ImGui::MenuItem("slice"))
          addedNode = m_graph->addNode<graph::SliceNode>();
        if (ImGui::MenuItem("streamline"))
          addedNode = m_graph->addNode<graph::StreamlineNode>();
        if (ImGui::MenuItem("surface normals"))
          addedNode = m_graph->addNode<graph::SurfaceNormalsNode>();
        if (ImGui::MenuItem("tetrahedralize"))
          addedNode = m_graph->addNode<graph::TetrahedralizeNode>();
        if (ImGui::MenuItem("tube"))
          addedNode = m_graph->addNode<graph::TubeNode>();
        if (ImGui::MenuItem("vector magnitude"))
          addedNode = m_graph->addNode<graph::VectorMagnitudeNode>();
        if (ImGui::MenuItem("vertex clustering"))
          addedNode = m_graph->addNode<graph::VertexClusteringNode>();
        ImGui::EndMenu();
      }

      if (ImGui::BeginMenu("utility")) {
        if (ImGui::MenuItem("dataset to components"))
          addedNode = m_graph->addNode<graph::DataSetToComponentsNode>();
        if (ImGui::MenuItem("components to dataset"))
          addedNode = m_graph->addNode<graph::ComponentsToDataSetNode>();
        if (ImGui::MenuItem("VTK file writer"))
          addedNode = m_graph->addNode<graph::VTKFileWriterNode>();
        ImGui::EndMenu();
      }

      ImGui::Separator();

      if (ImGui::MenuItem("actor"))
        addedNode = m_graph->addNode<graph::ActorNode>();

      ImGui::Separator();

      if (ImGui::BeginMenu("mapper")) {
        if (ImGui::MenuItem("glyphs"))
          addedNode = m_graph->addNode<graph::GlyphMapperNode>();
        if (ImGui::MenuItem("points"))
          addedNode = m_graph->addNode<graph::PointMapperNode>();
        if (ImGui::MenuItem("triangles"))
          addedNode = m_graph->addNode<graph::TriangleMapperNode>();
        if (ImGui::MenuItem("volume"))
          addedNode = m_graph->addNode<graph::VolumeMapperNode>();
        ImGui::EndMenu();
      }

      if (addedNode != nullptr)
        ImNodes::SetNodeScreenSpacePos(addedNode->id(), mousePos);

      ImGui::EndMenu();
    }

    if (!ImGui::IsPopupOpen("nodeEditorContextMenu"))
      m_contextMenuVisible = false;

    ImGui::EndPopup();
  }

  ImGui::PopStyleVar(1);
}

void NodeEditor::contextMenuPin()
{
  if (ImGui::BeginPopup("pinEditorContextMenu")) {
    if (!graph::isInputPortID(m_pinHoverId)) {
      ImGui::Text("LOGIC ERROR: incorrect pin type activated this menu");
    } else {
      auto *p = graph::InPort::fromID(m_pinHoverId);
      const auto *cs = p->cselector();
      int whichField = cs->currentField();
      if (ImGui::Combo("field",
              &whichField,
              fieldSelectorSelection_callback,
              (void *)cs,
              cs->numFields())) {
        p->selector()->setCurrentField(whichField);
      }
    }

    if (!ImGui::IsPopupOpen("pinEditorContextMenu"))
      m_contextPinMenuVisible = false;

    ImGui::EndPopup();
  }
}

void NodeEditor::updateNodeSummary()
{
  if (ImNodes::NumSelectedNodes() == 0) {
    m_prevNumSelectedNodes = -1;
    return;
  }

  static std::vector<int> selectedNodes;
  selectedNodes.resize(ImNodes::NumSelectedNodes());

  ImNodes::GetSelectedNodes(selectedNodes.data());
  const int selectedNode = selectedNodes[0];

  if (m_lastGraphChange >= m_graph->lastChange()
      && selectedNode == m_prevNumSelectedNodes)
    return;

  m_lastGraphChange.renew();

  auto *node = graph::Node::fromID(selectedNode);
  if (node) {
    m_nodeInfoWindow->setText(node->summary());
    m_prevNumSelectedNodes = selectedNode;
  } else {
    m_nodeInfoWindow->setText("<no summary>");
    m_prevNumSelectedNodes = -1;
  }
}

} // namespace tsd::viskores_graph
