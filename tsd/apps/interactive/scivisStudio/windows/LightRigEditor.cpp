// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "LightRigEditor.h"

#include "imgui.h"
#include "tsd/app/Context.h"

#include <algorithm>
#include <array>
#include <cfloat>
#include <cstring>
#include <string>
#include <vector>

namespace tsd::scivis_studio {


namespace {

struct LightTypeOption
{
  const char *label;
  const char *subtype;
};

constexpr std::array<LightTypeOption, 5> LIGHT_TYPES = {
    {{"Directional", "directional"},
        {"Point", "point"},
        {"Quad", "quad"},
        {"Spot", "spot"},
        {"Ring", "ring"}}};

std::vector<tsd::scene::LayerNodeRef> lightNodes(tsd::scene::LayerNodeRef root)
{
  std::vector<tsd::scene::LayerNodeRef> nodes;
  if (!root)
    return nodes;

  auto *layer = (*root)->layer();
  layer->traverse(root, [&](auto &node, int level) {
    if (level > 0 && node->isObject() && node->type() == ANARI_LIGHT)
      nodes.push_back(layer->at(node.index()));
    return true;
  });
  return nodes;
}

std::string lightName(tsd::scene::LayerNodeRef node)
{
  if (!node)
    return "<missing>";

  auto *object = (*node)->getObject();
  std::string label = (*node)->name();
  if (label.empty() && object)
    label = object->name();
  if (label.empty())
    label = "Light";
  return label;
}

std::string lightSubtype(tsd::scene::LayerNodeRef node)
{
  if (!node)
    return "";

  auto *object = (*node)->getObject();
  if (object)
    return object->subtype().str();
  return "";
}

} // namespace

LightRigEditor::LightRigEditor(
    tsd::ui::imgui::Application *app, ProjectContext *projectContext)
    : Window(app, "Light Rig"), m_projectContext(projectContext)
{}

LightRigEditor::~LightRigEditor() = default;

bool LightRigEditor::inputText(
    const char *label, std::string &value, size_t capacity)
{
  std::vector<char> buffer(capacity, '\0');
  std::strncpy(buffer.data(), value.c_str(), buffer.size() - 1);
  if (ImGui::InputText(label, buffer.data(), buffer.size())) {
    value = buffer.data();
    return true;
  }
  return false;
}

void LightRigEditor::buildUI_addLight(LightRig &rig)
{
  if (ImGui::Button("Add Light"))
    ImGui::OpenPopup("Add Light");

  if (ImGui::BeginPopup("Add Light")) {
    for (const auto &type : LIGHT_TYPES) {
      if (ImGui::MenuItem(type.label)) {
        auto node = m_projectContext->addLightToRig(rig, type.subtype);
        if (node)
          appContext()->setSelected(node);
      }
    }
    ImGui::EndPopup();
  }
}

void LightRigEditor::buildUI_lightList(LightRig &rig)
{
  auto root = m_projectContext->resolveLightRigRoot(rig);
  auto nodes = lightNodes(root);
  if (m_selectedLight >= static_cast<int>(nodes.size()))
    m_selectedLight = nodes.empty() ? -1 : 0;

  ImGui::SeparatorText("Lights");
  buildUI_addLight(rig);

  if (nodes.empty()) {
    ImGui::TextDisabled("No lights");
  } else {
    const ImGuiTableFlags flags = ImGuiTableFlags_Borders
        | ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp;
    if (ImGui::BeginTable("lights", 3, flags)) {
      ImGui::TableSetupColumn(
          "", ImGuiTableColumnFlags_WidthFixed, ImGui::GetFrameHeight());
      ImGui::TableSetupColumn("Name");
      ImGui::TableSetupColumn("Type");
      ImGui::TableHeadersRow();

      for (int i = 0; i < static_cast<int>(nodes.size()); ++i) {
        const bool selected = i == m_selectedLight;
        ImGui::PushID(i);
        ImGui::TableNextRow();
        if (selected) {
          const ImU32 selectedColor = ImGui::GetColorU32(ImGuiCol_Header);
          ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg0, selectedColor);
        }

        ImGui::TableNextColumn();
        if (ImGui::RadioButton("##selected", selected)) {
          m_selectedLight = i;
          appContext()->setSelected(nodes[i]);
        }

        ImGui::TableNextColumn();
        const auto name = lightName(nodes[i]);
        ImGui::TextUnformatted(name.c_str());
        if (selected)
          ImGui::SetItemDefaultFocus();

        ImGui::TableNextColumn();
        const auto subtype = lightSubtype(nodes[i]);
        ImGui::TextUnformatted(subtype.c_str());
        ImGui::PopID();
      }

      ImGui::EndTable();
    }
  }

  const bool hasSelection =
      m_selectedLight >= 0 && m_selectedLight < static_cast<int>(nodes.size());
  auto selectedNode =
      hasSelection ? nodes[m_selectedLight] : tsd::scene::LayerNodeRef{};
  ImGui::BeginDisabled(!hasSelection);
  if (ImGui::Button("Rename Selected") && hasSelection) {
    auto *object = (*selectedNode)->getObject();
    m_renameLightName = (*selectedNode)->name();
    if (m_renameLightName.empty() && object)
      m_renameLightName = object->name();
    ImGui::OpenPopup("Rename Light");
  }
  ImGui::SameLine();
  if (ImGui::Button("Remove Selected") && hasSelection) {
    m_projectContext->removeLightFromRig(rig, nodes[m_selectedLight]);
    m_selectedLight = -1;
  }
  ImGui::EndDisabled();

  ImGui::SetNextWindowSize(ImVec2(800.f, 0.f), ImGuiCond_Appearing);
  if (ImGui::BeginPopupModal(
          "Rename Light", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
    ImGui::Text("Light Name");
    ImGui::SetNextItemWidth(-FLT_MIN);
    if (ImGui::IsWindowAppearing())
      ImGui::SetKeyboardFocusHere();
    inputText("##lightName", m_renameLightName);

    ImGui::BeginDisabled(m_renameLightName.empty() || !selectedNode);
    if (ImGui::Button("Rename") && selectedNode) {
      (*selectedNode)->name() = m_renameLightName;
      if (auto *object = (*selectedNode)->getObject())
        object->setName(m_renameLightName);
      m_projectContext->project().markDirty();
      ImGui::CloseCurrentPopup();
    }
    ImGui::EndDisabled();
    ImGui::SameLine();
    if (ImGui::Button("Cancel"))
      ImGui::CloseCurrentPopup();
    ImGui::EndPopup();
  }
}

void LightRigEditor::buildUI()
{
  if (!m_projectContext)
    return;

  auto &project = m_projectContext->project();
  auto &rigs = project.lightRigs;

  if (ImGui::Button("Add Rig")) {
    if (auto *rig = m_projectContext->createLightRig())
      m_selectedRig = static_cast<int>(rigs.size()) - 1;
  }

  if (rigs.empty()) {
    ImGui::TextDisabled("No light rigs");
    return;
  }

  if (m_selectedRig >= static_cast<int>(rigs.size()))
    m_selectedRig = 0;

  const char *preview = rigs[m_selectedRig].name.c_str();
  if (ImGui::BeginCombo("Rig", preview)) {
    for (int i = 0; i < static_cast<int>(rigs.size()); ++i) {
      const bool selected = i == m_selectedRig;
      if (ImGui::Selectable(rigs[i].name.c_str(), selected)) {
        m_selectedRig = i;
        m_selectedLight = 0;
      }
      if (selected)
        ImGui::SetItemDefaultFocus();
    }
    ImGui::EndCombo();
  }

  auto &rig = rigs[m_selectedRig];
  if (inputText("Name", rig.name))
    project.markDirty();

  auto *shot = project::activeShot(project);
  const bool activeShotUsesRig = shot && shot->lightRigId == rig.id;
  ImGui::BeginDisabled(!shot || activeShotUsesRig);
  if (ImGui::Button("Use for Active Shot") && shot) {
    shot->lightRigId = rig.id;
    project.markDirty();
    m_projectContext->applyActiveShot();
  }
  ImGui::EndDisabled();

  ImGui::SameLine();
  if (ImGui::Button("Remove Rig")) {
    if (m_projectContext->shotUseCount(rig.id) > 0) {
      m_pendingDeleteRig = rig.id;
      ImGui::OpenPopup("Delete Light Rig?");
    } else {
      m_projectContext->removeLightRig(rig.id);
      m_selectedRig = 0;
      return;
    }
  }

  if (ImGui::BeginPopupModal(
          "Delete Light Rig?", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
    auto *pending = project::findLightRig(project, m_pendingDeleteRig);
    const int useCount = m_projectContext->shotUseCount(m_pendingDeleteRig);
    ImGui::Text("Delete '%s' and clear %d shot reference%s?",
        pending ? pending->name.c_str() : m_pendingDeleteRig.c_str(),
        useCount,
        useCount == 1 ? "" : "s");
    if (ImGui::Button("Delete")) {
      m_projectContext->removeLightRig(m_pendingDeleteRig);
      m_pendingDeleteRig.clear();
      m_selectedRig = 0;
      ImGui::CloseCurrentPopup();
    }
    ImGui::SameLine();
    if (ImGui::Button("Cancel")) {
      m_pendingDeleteRig.clear();
      ImGui::CloseCurrentPopup();
    }
    ImGui::EndPopup();
  }

  buildUI_lightList(rig);
}

} // namespace tsd::scivis_studio
