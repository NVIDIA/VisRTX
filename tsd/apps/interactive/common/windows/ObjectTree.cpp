// Copyright 2024 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "ObjectTree.h"
#include "tsd_ui.h"

namespace tsd_viewer {

static ImGuiTreeNodeFlags base_flags = ImGuiTreeNodeFlags_OpenOnArrow
    | ImGuiTreeNodeFlags_OpenOnDoubleClick | ImGuiTreeNodeFlags_SpanAvailWidth;

// ObjectTree definitions /////////////////////////////////////////////////////

ObjectTree::ObjectTree(AppContext *state, const char *name)
    : anari_viewer::windows::Window(name, true), m_context(state)
{}

void ObjectTree::buildUI()
{
  ImGui::BeginDisabled(!m_context->tsd.sceneLoadComplete);

  auto &ctx = m_context->tsd.ctx;
  auto &tree = ctx.tree;

  if (ImGui::Button("clear scene"))
    m_context->tsd.ctx.removeAllObjects();

  ImGui::Separator();

  if (!m_contextMenuVisible)
    m_contextMenuNode = tsd::INVALID_INDEX;
  m_hoveredNode = tsd::INVALID_INDEX;

  const ImGuiTableFlags flags =
      ImGuiTableFlags_RowBg | ImGuiTableFlags_Resizable;
  if (ImGui::BeginTable("objects", 1, flags)) {
    ImGui::TableSetupColumn("objects");

    const auto &style = ImGui::GetStyle();

    m_needToTreePop.resize(tree.size());
    auto onNodeEntryBuildUI = [&](auto &node, int level) {
      ImGui::TableNextRow();
      ImGui::TableSetColumnIndex(0);

      ImGuiTreeNodeFlags node_flags = base_flags;

      tsd::Object *obj = ctx.getObject(node->value);

      const bool enabled = node->enabled;
      const bool selected = obj && m_context->tsd.selectedObject == obj;
      if (selected) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.f, 1.f, 0.f, 1.f));
        node_flags |= ImGuiTreeNodeFlags_Selected;
      } else if (!enabled) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.3f, 0.3f, 0.3f, 1.f));
      }

      const char *nameText = "<unhandled UI node type>";
      if (!node->name.empty())
        nameText = node->name.c_str();
      else {
        switch (node->value.type()) {
        case ANARI_FLOAT32_MAT4:
          nameText = "xfm";
          break;
        case ANARI_SURFACE:
          nameText = obj ? obj->name().c_str() : "UNABLE TO FIND SURFACE";
          break;
        case ANARI_VOLUME:
          nameText = obj ? obj->name().c_str() : "UNABLE TO FIND VOLUME";
          break;
        case ANARI_LIGHT:
          nameText = obj ? obj->name().c_str() : "UNABLE TO FIND LIGHT";
          break;
        case ANARI_STRING:
          nameText = node->value.getCStr();
          break;
        default:
          nameText = anari::toString(node->value.type());
          break;
        }
      }

      const char *typeText = "[-]";
      switch (node->value.type()) {
      case ANARI_FLOAT32_MAT4:
        typeText = "[T]";
        break;
      case ANARI_SURFACE:
        typeText = "[S]";
        break;
      case ANARI_VOLUME:
        typeText = "[V]";
        break;
      case ANARI_LIGHT:
        typeText = "[L]";
        break;
      default:
        break;
      }

      if (node.isLeaf()) {
        node_flags |=
            ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;
      } else {
        ImGui::SetNextItemOpen(true, ImGuiCond_FirstUseEver);
      }

      bool open =
          ImGui::TreeNodeEx(&node, node_flags, "%s %s", typeText, nameText);

      m_needToTreePop[node.index()] = open && !node.isLeaf();

      if (ImGui::IsItemHovered())
        m_hoveredNode = node.index();

      if (ImGui::IsItemClicked() && m_contextMenuNode == tsd::INVALID_INDEX)
        m_context->setSelectedObject(obj);

      if (selected || !enabled)
        ImGui::PopStyleColor(1);

      return open && enabled;
    };

    int nodeIndex = 0;
    auto onNodeExitTreePop = [&](auto &node, int level) {
      if (m_needToTreePop[node.index()])
        ImGui::TreePop();
    };

    tree.traverse(tree.root(), onNodeEntryBuildUI, onNodeExitTreePop);

    ImGui::EndTable();
  }

  ImGui::EndDisabled();

  if (ImGui::IsWindowHovered()) {
    if (ImGui::IsMouseClicked(ImGuiMouseButton_Right)
        && m_hoveredNode != tsd::INVALID_INDEX) {
      m_contextMenuVisible = true;
      m_contextMenuNode = m_hoveredNode;
      ImGui::OpenPopup("ObjectTree_contextMenu");
    } else if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)
        && m_hoveredNode == tsd::INVALID_INDEX) {
      m_context->setSelectedObject(nullptr);
    }
  }

  buildUI_objectContextMenu();
}

void ObjectTree::buildUI_objectContextMenu()
{
  auto &tsd_ctx = m_context->tsd.ctx;
  auto &tree = tsd_ctx.tree;

  if (ImGui::BeginPopup("ObjectTree_contextMenu")) {
    if (ImGui::Checkbox("visible", &(*tree.at(m_contextMenuNode))->enabled))
      tsd_ctx.signalInstanceTreeChange();

    if (ImGui::BeginMenu("add")) {
      if (ImGui::BeginMenu("light")) {
        if (ImGui::MenuItem("directional")) {
          auto l =
              tsd_ctx.createObject<tsd::Light>(tsd::tokens::light::directional);
          l->setName("directional light");
          tsd_ctx.addInstancedObject(tree.at(m_contextMenuNode),
              tsd::utility::Any(ANARI_LIGHT, l.index()),
              "directional light");
          m_contextMenuNode = tsd::INVALID_INDEX;
          m_context->setSelectedObject(nullptr);
        }

        if (ImGui::MenuItem("point")) {
          auto l = tsd_ctx.createObject<tsd::Light>(tsd::tokens::light::point);
          l->setName("point light");
          tsd_ctx.addInstancedObject(tree.at(m_contextMenuNode),
              tsd::utility::Any(ANARI_LIGHT, l.index()),
              "point light");
          m_contextMenuNode = tsd::INVALID_INDEX;
          m_context->setSelectedObject(nullptr);
        }

        if (ImGui::MenuItem("quad")) {
          auto l = tsd_ctx.createObject<tsd::Light>(tsd::tokens::light::quad);
          l->setName("quad light");
          tsd_ctx.addInstancedObject(tree.at(m_contextMenuNode),
              tsd::utility::Any(ANARI_LIGHT, l.index()),
              "quad light");
          m_contextMenuNode = tsd::INVALID_INDEX;
          m_context->setSelectedObject(nullptr);
        }

        ImGui::EndMenu();
      }

      ImGui::EndMenu();
    }

    ImGui::Separator();

    if (ImGui::MenuItem("delete")) {
      if (m_contextMenuNode != tsd::INVALID_INDEX) {
        tsd_ctx.removeInstancedObject(tree.at(m_contextMenuNode));
        m_contextMenuNode = tsd::INVALID_INDEX;
        m_context->setSelectedObject(nullptr);
      }
    }

    ImGui::EndPopup();
  }

  if (!ImGui::IsPopupOpen("ObjectTree_contextMenu"))
    m_contextMenuVisible = false;
}

} // namespace tsd_viewer