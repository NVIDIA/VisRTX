// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "LayerTree.h"

// tsd_core
#include "tsd/core/scene/objects/Volume.hpp"
// tsd_io
#include "tsd/io/procedural.hpp"
// tsd_app
#include "tsd/app/Core.h"
// tsd_ui_imgui
#include "tsd/ui/imgui/modals/ExportNanoVDBFileDialog.h"
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
  buildUI_activateObjectSceneMenu();
  buildUI_objectSceneMenu();
  buildUI_newLayerSceneMenu();
  buildUI_setActiveLayersSceneMenus();
}

void LayerTree::setEnableAddRemoveLayers(bool enable)
{
  m_enableAddRemove = enable;
}

void LayerTree::buildUI_layerHeader()
{
  auto &scene = appCore()->tsd.scene;
  const auto &layers = scene.layers();

  if (scene.numberOfLayers() == 0) {
    ImGui::Text("No layers in scene");
    ImGui::BeginDisabled(!m_enableAddRemove);
    if (ImGui::Button("new")) {
      s_newLayerName.clear();
      ImGui::OpenPopup("LayerTree_contextMenu_newLayer");
    }
    ImGui::EndDisabled();
    return;
  }

  ImGui::SetNextItemWidth(-1.0f);
  ImGui::Combo("##layer",
      &m_layerIdx,
      UI_layerName_callback,
      (void *)&layers,
      layers.size());

  if (ImGui::IsItemHovered()) {
    ImGui::SetTooltip("right-click to set layer visibility");
    if (ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
      ImGui::OpenPopup("LayerTree_contextMenu_setActiveLayers");
      m_activeLayerMenuTriggered = true;
    }
  }

  if (ImGui::Button("clear")) {
    appCore()->clearSelected();
    appCore()->tsd.scene.removeAllObjects();
  }

  ImGui::SameLine();

  ImGui::BeginDisabled(!m_enableAddRemove);
  if (ImGui::Button("new")) {
    s_newLayerName.clear();
    ImGui::OpenPopup("LayerTree_contextMenu_newLayer");
  }
  ImGui::EndDisabled();

  ImGui::SameLine();

  ImGui::BeginDisabled(!m_enableAddRemove || m_layerIdx == 0);
  if (ImGui::Button("delete")) {
    auto to_delete = layers.at_index(m_layerIdx);
    scene.removeLayer(to_delete.first);
    m_layerIdx--;
  }
  ImGui::EndDisabled();
}

std::vector<tsd::core::LayerNodeRef> LayerTree::computeSelectionRange(
    tsd::core::Layer &layer,
    const tsd::core::LayerNodeRef &anchor,
    const tsd::core::LayerNodeRef &target)
{
  std::vector<tsd::core::LayerNodeRef> range;

  if (!anchor.valid() || !target.valid()) {
    return range;
  }

  bool foundFirst = false;
  bool foundSecond = false;

  layer.traverse(layer.root(), [&](auto &node, int level) {
    auto nodeRef = layer.at(node.index());

    // Check if this is either the anchor or target
    if (nodeRef == anchor || nodeRef == target) {
      if (!foundFirst) {
        foundFirst = true;
        range.push_back(nodeRef);
      } else {
        foundSecond = true;
        range.push_back(nodeRef);
        return false;
      }
    } else if (foundFirst && !foundSecond) {
      // We're between the two boundaries
      range.push_back(nodeRef);
    }
    
    return true;
  });

  if (!foundFirst || !foundSecond) {
    return std::vector<tsd::core::LayerNodeRef>();
  }

  return range;
}

void LayerTree::buildUI_tree()
{
  auto &scene = appCore()->tsd.scene;

  if (scene.numberOfLayers() == 0)
    return;

  auto &layer = *scene.layer(m_layerIdx);

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

      tsd::core::Object *obj = node->getObject();

      const bool firstDisabled =
          firstDisabledNode == nullptr && !node->isEnabled();
      if (firstDisabled) {
        firstDisabledNode = &node;
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.3f, 0.3f, 0.3f, 1.f));
      }

      auto selectedNodeRef = appCore()->getSelected();
      auto currentNodeRef = layer.at(node.index());

      // Check if this node is in the selection set
      const bool isSelectedNode = appCore()->isSelected(currentNodeRef);

      // Check if any selected node's object matches this node's object
      bool sameObject = false;
      if (obj) {
        const auto &selectedNodes = appCore()->getSelectedNodes();
        for (const auto &selected : selectedNodes) {
          if (selected.valid() && (*selected)->getObject() == obj) {
            sameObject = true;
            break;
          }
        }
      }

      const bool strongHighlight = isSelectedNode;
      const bool lightHighlight = !isSelectedNode && sameObject;

      if (strongHighlight) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.f, 1.f, 0.f, 1.f));
        node_flags |= ImGuiTreeNodeFlags_Selected;
      } else if (lightHighlight) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.f, 1.f, 0.f, 0.75f));
      }

      const char *nameText = "<unhandled UI node type>";
      if (!node->name().empty())
        nameText = node->name().c_str();
      else {
        switch (node->type()) {
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
        default:
          nameText = anari::toString(node->type());
          break;
        }
      }

      const char *typeText = "[-]";
      switch (node->type()) {
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
              anari::toString(node->type()),
              node->getObjectIndex());
        } else if (node->isTransform())
          ImGui::SetTooltip("transform: ANARI_FLOAT32_MAT4");
      }

      if (ImGui::IsItemClicked() && m_menuNode == TSD_INVALID_INDEX) {
        auto clickedNode = layer.at(node.index());

        ImGuiIO &io = ImGui::GetIO();
        bool ctrlPressed = io.KeyCtrl;
        bool shiftPressed = io.KeyShift;
        
        if (ctrlPressed) {
          // Toggle selection
          if (appCore()->isSelected(clickedNode)) {
            appCore()->removeFromSelection(clickedNode);
          } else {
            appCore()->addToSelection(clickedNode);
          }
          m_anchorNode = clickedNode;
        } else if (shiftPressed) {
          // Range selection
          if (m_anchorNode.valid()) {
            auto rangeNodes =
                computeSelectionRange(layer, m_anchorNode, clickedNode);
            if (!rangeNodes.empty()) {
              appCore()->setSelected(rangeNodes);
            }
          } else {
            appCore()->addToSelection(clickedNode);
            m_anchorNode = clickedNode;
          }
        } else {
          // Replace selection
          appCore()->setSelected(clickedNode);
          m_anchorNode = clickedNode;
        }
      }

      if (strongHighlight || lightHighlight)
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

void LayerTree::buildUI_activateObjectSceneMenu()
{
  if (!m_activeLayerMenuTriggered && ImGui::IsWindowHovered()) {
    // Check for Escape key to clear selection
    if (ImGui::IsKeyPressed(ImGuiKey_Escape, false)) {
      appCore()->clearSelected();
    }

    // Check for Delete key to delete selected nodes
    if (ImGui::IsKeyPressed(ImGuiKey_Delete, false)) {
      auto &scene = appCore()->tsd.scene;
      auto parentOnlyNodes = appCore()->getParentOnlySelectedNodes();

      if (!parentOnlyNodes.empty()) {
        for (const auto &node : parentOnlyNodes) {
          if (node.valid()) {
            scene.removeInstancedObject(node);
          }
        }
        appCore()->clearSelected();
      }
    }

    if (ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
      m_menuVisible = true;
      m_menuNode = m_hoveredNode;
      ImGui::OpenPopup("LayerTree_contextMenu_object");
    } else if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)
        && m_hoveredNode == TSD_INVALID_INDEX) {
      ImGuiIO &io = ImGui::GetIO();
      // Only clear selection on left click if no modifiers are pressed
      if (!io.KeyCtrl && !io.KeyShift) {
        appCore()->clearSelected();
      }
    }
  }
}

void LayerTree::buildUI_objectSceneMenu()
{
  auto &scene = appCore()->tsd.scene;

  if (scene.numberOfLayers() == 0)
    return;

  auto &layer = *scene.layer(m_layerIdx);
  const bool nodeSelected = m_menuNode != TSD_INVALID_INDEX;
  auto menuNode = nodeSelected ? layer.at(m_menuNode) : layer.root();

  bool clearSelectedNode = false;

  if (ImGui::BeginPopup("LayerTree_contextMenu_object")) {
    bool enabled = (*menuNode)->isEnabled();
    if (nodeSelected && ImGui::Checkbox("visible", &enabled)) {
      (*menuNode)->setEnabled(enabled);
      scene.signalLayerChange(&layer);
    }

    if (nodeSelected && ImGui::MenuItem("show all")) {
      layer.traverse(menuNode, [&](auto &n, int) {
        n->setEnabled(true);
        return true;
      });
      scene.signalLayerChange(&layer);
    }

    if (nodeSelected && ImGui::MenuItem("hide all")) {
      layer.traverse(menuNode, [&](auto &n, int) {
        n->setEnabled(false);
        return true;
      });
      scene.signalLayerChange(&layer);
    }

    ImGui::Separator();

    if (nodeSelected && ImGui::BeginMenu("rename")) {
      ImGui::InputText("##edit_node_name", &(*menuNode)->name());
      ImGui::EndMenu();
    }

    if (ImGui::BeginMenu("add")) {
      if (ImGui::MenuItem("transform")) {
        scene.insertChildTransformNode(menuNode);
        clearSelectedNode = true;
      }

      ImGui::Separator();

      if (ImGui::BeginMenu("new object")) {
        if (ImGui::BeginMenu("light")) {
          if (ImGui::MenuItem("directional")) {
            scene.insertNewChildObjectNode<tsd::core::Light>(menuNode,
                tsd::core::tokens::light::directional,
                "directional light");
            clearSelectedNode = true;
          }

          if (ImGui::MenuItem("point")) {
            scene.insertNewChildObjectNode<tsd::core::Light>(
                menuNode, tsd::core::tokens::light::point, "point light");
            clearSelectedNode = true;
          }

          if (ImGui::MenuItem("quad")) {
            scene.insertNewChildObjectNode<tsd::core::Light>(
                menuNode, tsd::core::tokens::light::quad, "quad light");
            clearSelectedNode = true;
          }

          if (ImGui::MenuItem("spot")) {
            scene.insertNewChildObjectNode<tsd::core::Light>(
                menuNode, tsd::core::tokens::light::spot, "spot light");
            clearSelectedNode = true;
          }

          if (ImGui::MenuItem("ring")) {
            scene.insertNewChildObjectNode<tsd::core::Light>(
                menuNode, tsd::core::tokens::light::ring, "ring light");
            clearSelectedNode = true;
          }

          if (ImGui::BeginMenu("hdri")) {
            if (ImGui::MenuItem("simple dome")) {
              tsd::io::generate_hdri_dome(scene, menuNode);
              clearSelectedNode = true;
            }

            if (ImGui::MenuItem("test image")) {
              tsd::io::generate_hdri_test_image(scene, menuNode);
              clearSelectedNode = true;
            }
            ImGui::EndMenu(); // "hdri"
          }

          ImGui::EndMenu(); // "light"
        }

        if (ImGui::BeginMenu("surface")) {
          tsd::core::GeometryRef g;
#define OBJECT_UI_MENU_ITEM(text, subtype)                                     \
  if (ImGui::MenuItem(text)) {                                                 \
    g = scene.createObject<tsd::core::Geometry>(                               \
        tsd::core::tokens::geometry::subtype);                                 \
  }
          OBJECT_UI_MENU_ITEM("cone", cone);
          OBJECT_UI_MENU_ITEM("curve", curve);
          OBJECT_UI_MENU_ITEM("cylinder", cylinder);
          OBJECT_UI_MENU_ITEM("isosurface", isosurface);
          OBJECT_UI_MENU_ITEM("neural", neural);
          OBJECT_UI_MENU_ITEM("quad", quad);
          OBJECT_UI_MENU_ITEM("sphere", sphere);
          OBJECT_UI_MENU_ITEM("triangle", triangle);
#undef OBJECT_UI_MENU_ITEM
          if (g) {
            auto s = scene.createSurface("", g, scene.defaultMaterial());
            scene.insertChildObjectNode(menuNode, s, "surface");
            clearSelectedNode = true;
          }

          ImGui::EndMenu(); // "surface"
        }

        ImGui::EndMenu(); // "new object"
      }

      if (ImGui::BeginMenu("existing object")) {
#define OBJECT_UI_MENU_ITEM(text, type)                                        \
  if (scene.numberOfObjects(type) > 0 && ImGui::BeginMenu(text)) {             \
    auto t = type;                                                             \
    if (auto i = tsd::ui::buildUI_objects_menulist(scene, t);                  \
        i != TSD_INVALID_INDEX)                                                \
      scene.insertChildObjectNode(menuNode, t, i);                             \
    ImGui::EndMenu();                                                          \
  }
        OBJECT_UI_MENU_ITEM("light", ANARI_LIGHT);
        OBJECT_UI_MENU_ITEM("surface", ANARI_SURFACE);
        OBJECT_UI_MENU_ITEM("volume", ANARI_VOLUME);
        ImGui::EndMenu();
      }

      ImGui::Separator();

      if (ImGui::MenuItem("import..."))
        appCore()->windows.importDialog->show();

      ImGui::Separator();

      if (ImGui::BeginMenu("procedural")) {
        if (ImGui::MenuItem("cylinders")) {
          tsd::io::generate_cylinders(scene, menuNode);
          clearSelectedNode = true;
        }

        if (ImGui::MenuItem("material orb")) {
          tsd::io::generate_material_orb(scene, menuNode);
          clearSelectedNode = true;
        }

        if (ImGui::MenuItem("monkey")) {
          tsd::io::generate_monkey(scene, menuNode);
          clearSelectedNode = true;
        }

        if (ImGui::MenuItem("randomSpheres")) {
          tsd::io::generate_randomSpheres(scene, menuNode);
          clearSelectedNode = true;
        }

        if (ImGui::MenuItem("rtow")) {
          tsd::io::generate_rtow(scene, menuNode);
          clearSelectedNode = true;
        }

        if (ImGui::MenuItem("noise volume")) {
          tsd::io::generate_noiseVolume(scene, menuNode);
          clearSelectedNode = true;
        }

        if (ImGui::MenuItem("sphere set volume")) {
          tsd::io::generate_sphereSetVolume(scene, menuNode);
          clearSelectedNode = true;
        }

        ImGui::EndMenu();
      }

      ImGui::EndMenu();
    }

    if (nodeSelected) {
      if ((*menuNode)->isObject()
          && (*menuNode)->getObject()->subtype()
              == core::tokens::volume::transferFunction1D) {
        auto tf1D = (*menuNode)->getObject();
        auto spatialFieldObject = tf1D->parameterValueAsObject("value");
        if (spatialFieldObject
            && spatialFieldObject->subtype()
                == core::tokens::volume::structuredRegular) {
          ImGui::Separator();
          if (ImGui::MenuItem("export to NanoVDB")) {
            appCore()->windows.exportNanoVDBDialog->show();
          }
        }
      }
      ImGui::Separator();

      if (ImGui::MenuItem("delete selected")) {
        auto parentOnlyNodes = appCore()->getParentOnlySelectedNodes();

        if (!parentOnlyNodes.empty()) {
          for (const auto &node : parentOnlyNodes) {
            if (node.valid()) {
              scene.removeInstancedObject(node);
            }
          }
          m_menuNode = TSD_INVALID_INDEX;
          appCore()->clearSelected();
        } else if (m_menuNode != TSD_INVALID_INDEX) {
          // Fallback: delete the menu node if nothing is selected
          scene.removeInstancedObject(layer.at(m_menuNode));
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

void LayerTree::buildUI_newLayerSceneMenu()
{
  if (ImGui::BeginPopup("LayerTree_contextMenu_newLayer")) {
    ImGui::InputText("layer name", &s_newLayerName);

    ImGui::Separator();

    ImGuiIO &io = ImGui::GetIO();
    if ((ImGui::Button("ok") || ImGui::IsKeyDown(ImGuiKey_Enter))
        && !s_newLayerName.empty()) {
      auto &scene = appCore()->tsd.scene;
      tsd::core::Token layerName = s_newLayerName.c_str();
      auto *newLayer = scene.addLayer(layerName);

      auto &layers = scene.layers();
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

void LayerTree::buildUI_setActiveLayersSceneMenus()
{
  if (ImGui::BeginPopup("LayerTree_contextMenu_setActiveLayers")) {
    auto &scene = appCore()->tsd.scene;

    if (ImGui::Button("show all"))
      scene.setAllLayersActive();

    for (auto &ls : scene.layers()) {
      ImGui::PushID(ls.second.ptr.get());
      // Make sure at least one layer is always active
      ImGui::BeginDisabled(
          scene.numberOfActiveLayers() < 2 && ls.second.active);

      bool active = ls.second.active;
      if (ImGui::Checkbox(ls.first.c_str(), &active))
        scene.setLayerActive(ls.first, active);

      ImGui::SameLine();
      ImGui::Text("|");
      ImGui::SameLine();

      if (ImGui::Button("o"))
        scene.setOnlyLayerActive(ls.first);

      ImGui::EndDisabled();
      ImGui::PopID();
    }
    ImGui::EndPopup();
  } else {
    m_activeLayerMenuTriggered = false;
  }
}

} // namespace tsd::ui::imgui
