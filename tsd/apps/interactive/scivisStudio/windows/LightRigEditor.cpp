// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "LightRigEditor.h"

#include "imgui.h"
#include "tsd/app/Context.h"
#include "tsd/ui/imgui/Application.h"

#include <algorithm>
#include <array>
#include <cfloat>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

namespace tsd::scivis_studio {

namespace {

// Default rig Archives to a .tsd extension when the user didn't supply one.
std::string withTsdExtension(const std::string &path)
{
  std::filesystem::path p(path);
  if (p.extension().empty())
    p.replace_extension(".tsd");
  return p.string();
}

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

int selectedLightIndex(const std::vector<tsd::scene::LayerNodeRef> &nodes,
    tsd::scene::LayerNodeRef selectedNode)
{
  if (!selectedNode)
    return -1;

  auto itr = std::find(nodes.begin(), nodes.end(), selectedNode);
  if (itr == nodes.end())
    return -1;

  return static_cast<int>(std::distance(nodes.begin(), itr));
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

void LightRigEditor::buildUI_nameField(LightRig &rig)
{
  if (m_nameBufferRig != rig.id) {
    m_nameBufferRig = rig.id;
    m_nameBuffer = rig.name;
    m_nameError.clear();
  }

  std::vector<char> buffer(512, '\0');
  std::strncpy(buffer.data(), m_nameBuffer.c_str(), buffer.size() - 1);
  const bool entered = ImGui::InputText("Name",
      buffer.data(),
      buffer.size(),
      ImGuiInputTextFlags_EnterReturnsTrue);
  m_nameBuffer = buffer.data();

  if (entered || ImGui::IsItemDeactivatedAfterEdit()) {
    std::string error;
    if (m_nameBuffer == rig.name)
      m_nameError.clear();
    else if (m_projectContext->renameLightRig(rig.id, m_nameBuffer, &error))
      m_nameError.clear();
    else {
      m_nameError = error;
      m_nameBuffer = rig.name; // reject: restore the last valid name
    }
  }

  if (!m_nameError.empty())
    ImGui::TextColored(
        ImVec4(1.f, 0.4f, 0.4f, 1.f), "Invalid name: %s", m_nameError.c_str());
}

void LightRigEditor::syncSelectionToActiveShot()
{
  auto &project = m_projectContext->project();
  auto *shot = project::activeShot(project);
  const auto activeShotId = shot ? shot->id : ShotID{};
  const auto activeLightRigId = shot ? shot->lightRigId : LightRigID{};
  if (activeShotId == m_lastActiveShotId
      && activeLightRigId == m_lastActiveShotLightRigId)
    return;

  m_lastActiveShotId = activeShotId;
  m_lastActiveShotLightRigId = activeLightRigId;
  if (!shot)
    return;

  auto itr = std::find_if(project.lightRigs.begin(),
      project.lightRigs.end(),
      [&](const LightRig &rig) { return rig.id == shot->lightRigId; });
  if (itr == project.lightRigs.end())
    return;

  m_selectedRig =
      static_cast<int>(std::distance(project.lightRigs.begin(), itr));
  m_selectedLight = -1;
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
  auto *ctx = appContext();
  m_selectedLight = selectedLightIndex(
      nodes, ctx ? ctx->getFirstSelected() : tsd::scene::LayerNodeRef{});

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
    if (ctx)
      ctx->removeFromSelection(selectedNode);
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

void LightRigEditor::pollPendingFileIO()
{
  if (m_pendingFileIO == PendingFileIO::None || m_pendingFilename.empty())
    return;

  const auto request = m_pendingFileIO;
  const auto filename = m_pendingFilename;
  const auto rigToSave = m_pendingSaveRig;
  m_pendingFileIO = PendingFileIO::None;
  m_pendingFilename.clear();
  m_pendingSaveRig.clear();

  auto &project = m_projectContext->project();
  std::string error;
  if (request == PendingFileIO::Load) {
    if (m_projectContext->loadLightRigArchive(filename, &error)) {
      m_selectedRig = static_cast<int>(project.lightRigs.size()) - 1;
      m_selectedLight = -1;
    } else {
      m_ioError = error;
      ImGui::OpenPopup("Light Rig Archive Error");
    }
  } else if (request == PendingFileIO::Save) {
    if (!m_projectContext->saveLightRigArchive(
            rigToSave, withTsdExtension(filename), &error)) {
      m_ioError = error;
      ImGui::OpenPopup("Light Rig Archive Error");
    }
  }
}

void LightRigEditor::buildUI()
{
  if (!m_projectContext)
    return;

  pollPendingFileIO();
  syncSelectionToActiveShot();

  auto &project = m_projectContext->project();
  auto &rigs = project.lightRigs;

  if (ImGui::Button("Add Rig")) {
    if (auto *rig = m_projectContext->createLightRig())
      m_selectedRig = static_cast<int>(rigs.size()) - 1;
  }

  ImGui::SameLine();
  ImGui::BeginDisabled(rigs.empty());
  if (ImGui::Button("Clone Rig")) {
    if (m_selectedRig >= static_cast<int>(rigs.size()))
      m_selectedRig = 0;

    const auto sourceRigId = rigs[m_selectedRig].id;
    if (m_projectContext->cloneLightRig(sourceRigId)) {
      m_selectedRig = static_cast<int>(rigs.size()) - 1;
      m_selectedLight = -1;
    }
  }
  ImGui::EndDisabled();

  ImGui::SameLine();
  if (ImGui::Button("Load Archive...")) {
    m_pendingFileIO = PendingFileIO::Load;
    m_pendingFilename.clear();
    m_app->getFilenameFromDialog(
        m_pendingFilename, tsd::ui::imgui::FileDialogMode::OpenFile);
  }

  if (rigs.empty()) {
    buildUI_ioError();
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
        m_selectedLight = -1;
      }
      if (selected)
        ImGui::SetItemDefaultFocus();
    }
    ImGui::EndCombo();
  }

  auto &rig = rigs[m_selectedRig];
  buildUI_nameField(rig);

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
  if (ImGui::Button("Save Archive...")) {
    m_pendingFileIO = PendingFileIO::Save;
    m_pendingSaveRig = rig.id;
    m_pendingFilename.clear();
    m_app->getFilenameFromDialog(
        m_pendingFilename, tsd::ui::imgui::FileDialogMode::SaveFile);
  }

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
    auto *pending = light_rig::findLightRig(project, m_pendingDeleteRig);
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

  buildUI_ioError();
}

void LightRigEditor::buildUI_ioError()
{
  ImGui::SetNextWindowSize(ImVec2(500.f, 0.f), ImGuiCond_Appearing);
  if (ImGui::BeginPopupModal("Light Rig Archive Error",
          nullptr,
          ImGuiWindowFlags_AlwaysAutoResize)) {
    ImGui::TextWrapped("%s", m_ioError.c_str());
    ImGui::Spacing();
    if (ImGui::Button("OK") || ImGui::IsKeyPressed(ImGuiKey_Escape)) {
      m_ioError.clear();
      ImGui::CloseCurrentPopup();
    }
    ImGui::EndPopup();
  }
}

} // namespace tsd::scivis_studio
