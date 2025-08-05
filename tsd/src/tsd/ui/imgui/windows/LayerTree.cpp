// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "LayerTree.h"
// tsd_io
#include "tsd/io/procedural.hpp"
// tsd_app
#include "tsd/app/Core.h"
// tsd_ui_imgui
#include "tsd/ui/imgui/modals/ImportFileDialog.h"
#include "tsd/ui/imgui/tsd_ui_imgui.h"

namespace tsd::ui::imgui {

static std::string s_newLayerName;

static bool UI_layerName_callback(void *l, int index, const char **out_text)
{
  const auto &layers = *(const tsd::core::LayerMap *)l;
  *out_text = layers.at_index(index).first.c_str();
  return true;
}

// LayerTree definitions //////////////////////////////////////////////////////

LayerTree::LayerTree(Application *app, const char *name) : Window(app, name) {}

void LayerTree::buildUI()
{
  if (!appCore()->tsd.sceneLoadComplete) {
    ImGui::Text("PLEASE WAIT...LOADING SCENE");
    return;
  }

  buildUI_layerHeader();
  ImGui::Separator();
  buildUI_tree();
  buildUI_activateObjectContextMenu();
  buildUI_buildObjectContextMenu();
  buildUI_buildNewLayerContextMenu();
}

void LayerTree::buildUI_layerHeader()
{
  auto &ctx = appCore()->tsd.ctx;
  const auto &layers = ctx.layers();

  ImGui::SetNextItemWidth(-1.0f);
  ImGui::Combo("##layer",
      &m_layerIdx,
      UI_layerName_callback,
      (void *)&layers,
      layers.size());

  if (ImGui::Button("clear")) {
    appCore()->clearSelected();
    appCore()->tsd.ctx.removeAllObjects();
  }

  ImGui::SameLine();

  if (ImGui::Button("new")) {
    s_newLayerName.clear();
    ImGui::OpenPopup("LayerTree_contextMenu_newLayer");
  }

  ImGui::SameLine();

  ImGui::BeginDisabled(m_layerIdx == 0);
  if (ImGui::Button("delete")) {
    ctx.removeLayer(layers.at_index(m_layerIdx).first);
    m_layerIdx--;
  }
  ImGui::EndDisabled();
}

void LayerTree::buildUI_tree()
{
  auto &ctx = appCore()->tsd.ctx;
  auto &layer = *ctx.layer(m_layerIdx);

  if (!m_menuVisible)
    m_menuNode = TSD_INVALID_INDEX;
  m_hoveredNode = TSD_INVALID_INDEX;

  const ImGuiTableFlags flags =
      ImGuiTableFlags_RowBg | ImGuiTableFlags_Resizable;
  if (ImGui::BeginTable("objects", 1, flags)) {
    ImGui::TableSetupColumn("objects");

    const auto &style = ImGui::GetStyle();

    // to track if children are also disabled:
    const void *firstDisabledNode = nullptr;

    m_needToTreePop.clear();
    m_needToTreePop.resize(layer.capacity(), false);
    auto onNodeEntryBuildUI = [&](auto &node, int level) {
      if (level == 0)
        return true;

      ImGui::TableNextRow();
      ImGui::TableSetColumnIndex(0);

      ImGuiTreeNodeFlags node_flags = ImGuiTreeNodeFlags_OpenOnArrow
          | ImGuiTreeNodeFlags_OpenOnDoubleClick
          | ImGuiTreeNodeFlags_SpanAvailWidth;

      tsd::core::Object *obj = ctx.getObject(node->value);

      const bool firstDisabled = firstDisabledNode == nullptr && !node->enabled;
      if (firstDisabled) {
        firstDisabledNode = &node;
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.3f, 0.3f, 0.3f, 1.f));
      }

      const bool selected = (obj && appCore()->tsd.selectedObject == obj)
          || (appCore()->tsd.selectedNode
              && node == *appCore()->tsd.selectedNode);
      if (selected) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.f, 1.f, 0.f, 1.f));
        node_flags |= ImGuiTreeNodeFlags_Selected;
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

      if (ImGui::IsItemHovered()) {
        m_hoveredNode = node.index();
        if (node->isObject()) {
          ImGui::SetTooltip("object: %s[%zu]",
              anari::toString(node->value.type()),
              node->value.getAsObjectIndex());
        } else if (node->isTransform())
          ImGui::SetTooltip("transform: ANARI_FLOAT32_MAT4");
      }

      if (ImGui::IsItemClicked() && m_menuNode == TSD_INVALID_INDEX)
        appCore()->setSelectedNode(node);

      if (selected)
        ImGui::PopStyleColor(1);

      return open;
    };

    auto onNodeExitTreePop = [&](auto &node, int level) {
      if (level == 0)
        return;
      if (&node == firstDisabledNode) {
        firstDisabledNode = nullptr;
        ImGui::PopStyleColor(1);
      }
      if (m_needToTreePop[node.index()])
        ImGui::TreePop();
    };

    layer.traverse(layer.root(), onNodeEntryBuildUI, onNodeExitTreePop);

    ImGui::EndTable();
  }
}

void LayerTree::buildUI_activateObjectContextMenu()
{
  if (ImGui::IsWindowHovered()) {
    if (ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
      m_menuVisible = true;
      m_menuNode = m_hoveredNode;
      ImGui::OpenPopup("LayerTree_contextMenu_object");
    } else if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)
        && m_hoveredNode == TSD_INVALID_INDEX) {
      appCore()->clearSelected();
    }
  }
}

void LayerTree::buildUI_buildObjectContextMenu()
{
  auto &ctx = appCore()->tsd.ctx;
  auto &layer = *ctx.layer(m_layerIdx);
  const bool nodeSelected = m_menuNode != TSD_INVALID_INDEX;
  auto menuNode = nodeSelected ? layer.at(m_menuNode) : layer.root();

  bool clearSelectedNode = false;

  if (ImGui::BeginPopup("LayerTree_contextMenu_object")) {
    if (nodeSelected && ImGui::Checkbox("visible", &(*menuNode)->enabled))
      ctx.signalLayerChange(&layer);

    if (nodeSelected && ImGui::BeginMenu("rename")) {
      ImGui::InputText("##edit_node_name", &(*menuNode)->name);
      ImGui::EndMenu();
    }

    if (ImGui::BeginMenu("add")) {
      if (ImGui::MenuItem("transform")) {
        ctx.insertChildTransformNode(menuNode);
        clearSelectedNode = true;
      }

      ImGui::Separator();

      if (ImGui::MenuItem("imported file"))
        appCore()->windows.importDialog->show();

      ImGui::Separator();

      if (ImGui::BeginMenu("existing object")) {
#define OBJECT_UI_MENU_ITEM(text, type)                                        \
  if (ctx.numberOfObjects(type) > 0 && ImGui::BeginMenu(text)) {               \
    if (auto i = tsd::ui::buildUI_objects_menulist(ctx, type);                 \
        i != TSD_INVALID_INDEX)                                                \
      ctx.insertChildObjectNode(menuNode, type, i);                            \
    ImGui::EndMenu();                                                          \
  }
        OBJECT_UI_MENU_ITEM("surface", ANARI_SURFACE);
        OBJECT_UI_MENU_ITEM("volume", ANARI_VOLUME);
        OBJECT_UI_MENU_ITEM("light", ANARI_LIGHT);
        ImGui::EndMenu();
      }

      ImGui::Separator();

      if (ImGui::BeginMenu("procedural")) {
        if (ImGui::MenuItem("cylinders")) {
          tsd::io::generate_cylinders(ctx, menuNode);
          clearSelectedNode = true;
        }

        if (ImGui::MenuItem("material orb")) {
          tsd::io::generate_material_orb(ctx, menuNode);
          clearSelectedNode = true;
        }

        if (ImGui::MenuItem("monkey")) {
          tsd::io::generate_monkey(ctx, menuNode);
          clearSelectedNode = true;
        }

        if (ImGui::MenuItem("randomSpheres")) {
          tsd::io::generate_randomSpheres(ctx, menuNode);
          clearSelectedNode = true;
        }

        if (ImGui::MenuItem("rtow")) {
          tsd::io::generate_rtow(ctx, menuNode);
          clearSelectedNode = true;
        }

        if (ImGui::MenuItem("noise volume")) {
          tsd::io::generate_noiseVolume(ctx, menuNode);
          clearSelectedNode = true;
        }

        ImGui::EndMenu();
      }

      ImGui::Separator();

      if (ImGui::BeginMenu("light")) {
        if (ImGui::MenuItem("directional")) {
          ctx.insertNewChildObjectNode<tsd::core::Light>(menuNode,
              tsd::core::tokens::light::directional,
              "directional light");
          clearSelectedNode = true;
        }

        if (ImGui::MenuItem("point")) {
          ctx.insertNewChildObjectNode<tsd::core::Light>(
              menuNode, tsd::core::tokens::light::point, "point light");
          clearSelectedNode = true;
        }

        if (ImGui::MenuItem("quad")) {
          ctx.insertNewChildObjectNode<tsd::core::Light>(
              menuNode, tsd::core::tokens::light::quad, "quad light");
          clearSelectedNode = true;
        }

        if (ImGui::MenuItem("spot")) {
          ctx.insertNewChildObjectNode<tsd::core::Light>(
              menuNode, tsd::core::tokens::light::spot, "spot light");
          clearSelectedNode = true;
        }

        if (ImGui::BeginMenu("hdri")) {
          if (ImGui::MenuItem("simple dome")) {
            tsd::io::generate_hdri_dome(ctx, menuNode);
            clearSelectedNode = true;
          }

          if (ImGui::MenuItem("test image")) {
            tsd::io::generate_hdri_test_image(ctx, menuNode);
            clearSelectedNode = true;
          }
          ImGui::EndMenu();
        }

        ImGui::EndMenu();
      }

      ImGui::EndMenu();
    }

    if (nodeSelected) {
      ImGui::Separator();

      if (ImGui::MenuItem("delete selected")) {
        if (m_menuNode != TSD_INVALID_INDEX) {
          ctx.removeInstancedObject(layer.at(m_menuNode));
          m_menuNode = TSD_INVALID_INDEX;
          appCore()->clearSelected();
        }
      }
    }

    ImGui::EndPopup();

    if (clearSelectedNode) {
      m_menuNode = TSD_INVALID_INDEX;
      appCore()->clearSelected();
    }
  }

  if (!ImGui::IsPopupOpen("LayerTree_contextMenu_object"))
    m_menuVisible = false;
}

void LayerTree::buildUI_buildNewLayerContextMenu()
{
  if (ImGui::BeginPopup("LayerTree_contextMenu_newLayer")) {
    ImGui::InputText("layer name", &s_newLayerName);

    ImGui::Separator();

    ImGuiIO &io = ImGui::GetIO();
    if ((ImGui::Button("ok") || ImGui::IsKeyDown(ImGuiKey_Enter))
        && !s_newLayerName.empty()) {
      auto &ctx = appCore()->tsd.ctx;
      tsd::core::Token layerName = s_newLayerName.c_str();
      ctx.addLayer(layerName);

      auto &layers = ctx.layers();
      for (int i = 0; i < int(layers.size()); i++) {
        if (layers.at_index(i).first == layerName) {
          m_layerIdx = i;
          break;
        }
      }

      ImGui::CloseCurrentPopup();
    }

    ImGui::SameLine();

    if (ImGui::Button("cancel"))
      ImGui::CloseCurrentPopup();

    ImGui::EndPopup();
  }
}

} // namespace tsd::ui::imgui
