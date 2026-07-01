// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "CameraRigEditor.h"

#include "tsd/rendering/view/ManipulatorToTSD.hpp"
#include "tsd/scene/objects/Camera.hpp"
#include "tsd/ui/imgui/Application.h"
#include "tsd/ui/imgui/tsd_ui_imgui.h"

#include "imgui.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
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

int cameraInterpolationIndex(CameraInterpolation interpolation)
{
  switch (interpolation) {
  case CameraInterpolation::Hold:
    return 0;
  case CameraInterpolation::Linear:
    return 1;
  case CameraInterpolation::EaseOut:
    return 2;
  case CameraInterpolation::EaseIn:
    return 3;
  case CameraInterpolation::EaseOutIn:
    return 4;
  }
  return 1;
}

CameraInterpolation cameraInterpolationFromIndex(int index)
{
  switch (index) {
  case 0:
    return CameraInterpolation::Hold;
  case 2:
    return CameraInterpolation::EaseOut;
  case 3:
    return CameraInterpolation::EaseIn;
  case 4:
    return CameraInterpolation::EaseOutIn;
  default:
    return CameraInterpolation::Linear;
  }
}

} // namespace

CameraRigEditor::CameraRigEditor(
    tsd::ui::imgui::Application *app, ProjectContext *projectContext)
    : Window(app, "Camera Rig"), m_projectContext(projectContext)
{}

CameraRigEditor::~CameraRigEditor() = default;

bool CameraRigEditor::inputText(
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

void CameraRigEditor::buildUI_nameField(CameraRig &rig)
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
    else if (m_projectContext->renameCameraRig(rig.id, m_nameBuffer, &error))
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

void CameraRigEditor::syncSelectionToActiveShot()
{
  auto &project = m_projectContext->project();
  auto *shot = project::activeShot(project);
  const auto activeShotId = shot ? shot->id : ShotID{};
  if (activeShotId == m_lastActiveShotId)
    return;

  m_lastActiveShotId = activeShotId;
  if (!shot)
    return;

  auto itr = std::find_if(project.cameraRigs.begin(),
      project.cameraRigs.end(),
      [&](const CameraRig &rig) { return rig.id == shot->cameraRigId; });
  if (itr == project.cameraRigs.end())
    return;

  m_selectedRig =
      static_cast<int>(std::distance(project.cameraRigs.begin(), itr));
  m_selectedKeyframe = -1;
}

void CameraRigEditor::buildUI_rigControls()
{
  auto &project = m_projectContext->project();

  if (ImGui::Button("Add Rig")) {
    if (auto *rig = m_projectContext->createCameraRig()) {
      (void)rig;
      m_selectedRig = static_cast<int>(project.cameraRigs.size()) - 1;
      m_selectedKeyframe = -1;
    }
  }

  ImGui::SameLine();
  ImGui::BeginDisabled(project.cameraRigs.empty());
  if (ImGui::Button("Clone Rig")) {
    if (m_selectedRig >= static_cast<int>(project.cameraRigs.size()))
      m_selectedRig = 0;

    auto clone = project.cameraRigs[m_selectedRig];
    clone.id = camera_rig::nextCameraRigId(project);
    clone.name = clone.name.empty() ? "Camera Rig Copy" : clone.name + " Copy";
    project.cameraRigs.push_back(clone);
    m_selectedRig = static_cast<int>(project.cameraRigs.size()) - 1;
    m_selectedKeyframe = -1;
    project.markDirty();
  }
  ImGui::EndDisabled();

  ImGui::SameLine();
  if (ImGui::Button("Load Archive...")) {
    m_pendingFileIO = PendingFileIO::Load;
    m_pendingFilename.clear();
    m_app->getFilenameFromDialog(
        m_pendingFilename, tsd::ui::imgui::FileDialogMode::OpenFile);
  }

  if (project.cameraRigs.empty()) {
    buildUI_ioError();
    ImGui::TextDisabled("No camera rigs");
    return;
  }

  if (m_selectedRig >= static_cast<int>(project.cameraRigs.size()))
    m_selectedRig = 0;

  const char *preview = project.cameraRigs[m_selectedRig].name.c_str();
  if (ImGui::BeginCombo("Rig", preview)) {
    for (int i = 0; i < static_cast<int>(project.cameraRigs.size()); ++i) {
      const bool selected = i == m_selectedRig;
      if (ImGui::Selectable(project.cameraRigs[i].name.c_str(), selected)) {
        m_selectedRig = i;
        m_selectedKeyframe = -1;
      }
      if (selected)
        ImGui::SetItemDefaultFocus();
    }
    ImGui::EndCombo();
  }

  auto &cameraRig = project.cameraRigs[m_selectedRig];
  buildUI_nameField(cameraRig);

  auto *shot = project::activeShot(project);
  const bool activeShotUsesRig = shot && shot->cameraRigId == cameraRig.id;
  ImGui::BeginDisabled(!shot || activeShotUsesRig);
  if (ImGui::Button("Use for Active Shot") && shot) {
    shot->cameraRigId = cameraRig.id;
    project.markDirty();
    m_projectContext->applyActiveShot();
  }
  ImGui::EndDisabled();

  ImGui::SameLine();
  if (ImGui::Button("Save Archive...")) {
    m_pendingFileIO = PendingFileIO::Save;
    m_pendingSaveRig = cameraRig.id;
    m_pendingFilename.clear();
    m_app->getFilenameFromDialog(
        m_pendingFilename, tsd::ui::imgui::FileDialogMode::SaveFile);
  }

  ImGui::SameLine();
  if (ImGui::Button("Remove Rig")) {
    if (m_projectContext->cameraRigUseCount(cameraRig.id) > 0) {
      m_pendingDeleteRig = cameraRig.id;
      ImGui::OpenPopup("Delete Camera Rig?");
    } else {
      m_projectContext->removeCameraRig(cameraRig.id);
      m_selectedRig = 0;
      return;
    }
  }

  if (ImGui::BeginPopupModal(
          "Delete Camera Rig?", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
    auto *pending = camera_rig::findCameraRig(project, m_pendingDeleteRig);
    const int useCount =
        m_projectContext->cameraRigUseCount(m_pendingDeleteRig);
    ImGui::Text("Delete '%s' and clear %d shot reference%s?",
        pending ? pending->name.c_str() : m_pendingDeleteRig.c_str(),
        useCount,
        useCount == 1 ? "" : "s");
    if (ImGui::Button("Delete")) {
      m_projectContext->removeCameraRig(m_pendingDeleteRig);
      m_pendingDeleteRig.clear();
      m_selectedRig = 0;
      m_selectedKeyframe = -1;
      ImGui::CloseCurrentPopup();
    }
    ImGui::SameLine();
    if (ImGui::Button("Cancel")) {
      m_pendingDeleteRig.clear();
      ImGui::CloseCurrentPopup();
    }
    ImGui::EndPopup();
  }

  buildUI_ioError();
}

void CameraRigEditor::buildUI_ioError()
{
  ImGui::SetNextWindowSize(ImVec2(500.f, 0.f), ImGuiCond_Appearing);
  if (ImGui::BeginPopupModal("Camera Rig Archive Error",
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

bool CameraRigEditor::buildUI_poseEditor(
    const char *label, ManipulatorState &state)
{
  if (!ImGui::CollapsingHeader(label, ImGuiTreeNodeFlags_DefaultOpen))
    return false;

  bool changed = false;
  auto &pose = state.orbit;

  ImGui::PushID(label);

  ImGui::SetNextItemWidth(-1.f);
  changed |=
      ImGui::DragFloat3("Look At", &pose.lookat.x, 0.01f, 0.f, 0.f, "%.3f");

  ImGui::SetNextItemWidth(-1.f);
  changed |=
      ImGui::SliderFloat("Azimuth", &pose.azeldist.x, 0.f, 360.f, "%.3f");

  ImGui::SetNextItemWidth(-1.f);
  changed |=
      ImGui::SliderFloat("Elevation", &pose.azeldist.y, 0.f, 360.f, "%.3f");

  ImGui::SetNextItemWidth(-1.f);
  changed |=
      ImGui::DragFloat("Distance", &pose.azeldist.z, 0.01f, 0.f, 0.f, "%.3f");

  bool hasFixedDistance = std::isfinite(pose.fixedDist);
  if (ImGui::Checkbox("Fixed Distance", &hasFixedDistance)) {
    pose.fixedDist = hasFixedDistance ? pose.azeldist.z : tsd::math::inf;
    changed = true;
  }

  auto fixedDistance =
      std::isfinite(pose.fixedDist) ? pose.fixedDist : pose.azeldist.z;
  ImGui::BeginDisabled(!hasFixedDistance);
  ImGui::SetNextItemWidth(-1.f);
  if (ImGui::DragFloat(
          "Fixed Distance Value", &fixedDistance, 0.01f, 0.f, 0.f, "%.3f")) {
    pose.fixedDist = fixedDistance;
    changed = true;
  }
  ImGui::EndDisabled();

  auto upAxis = pose.upAxis;
  if (ImGui::Combo("Up", &upAxis, "+x\0+y\0+z\0-x\0-y\0-z\0\0")) {
    pose.upAxis = upAxis;
    changed = true;
  }

  auto mode = pose.mode;
  if (ImGui::Combo("Mode", &mode, "Orbit\0Look\0\0")) {
    pose.mode = mode;
    changed = true;
  }

  ImGui::PopID();
  return changed;
}

void CameraRigEditor::buildUI_keyframes(CameraRig &cameraRig)
{
  auto &project = m_projectContext->project();
  auto *shot = project::activeShot(project);
  auto *ctx = m_projectContext->appContext();
  auto &rig = cameraRig;

  ImGui::BeginDisabled(!ctx);
  if (ImGui::Button("Set View")) {
    rig.current =
        camera_rig::manipulatorStateFromManipulator(ctx->view.manipulator);
    project.markDirty();
  }
  tsd::ui::tooltipForPreviousItem("Set Rig View From Viewport");
  ImGui::EndDisabled();

  ImGui::SameLine();

  ImGui::BeginDisabled(!ctx || !shot);
  if (ImGui::Button("Capture")) {
    CameraKeyframe keyframe;
    keyframe.frame = shot->currentFrame;
    keyframe.name = "Frame " + std::to_string(shot->currentFrame);
    keyframe.manipulator =
        camera_rig::manipulatorStateFromManipulator(ctx->view.manipulator);
    rig.keyframes.push_back(std::move(keyframe));
    camera_rig::sortKeyframes(rig);
    m_selectedKeyframe = static_cast<int>(rig.keyframes.size()) - 1;
    project.markDirty();
  }
  tsd::ui::tooltipForPreviousItem("Capture Keyframe At Current Frame");
  ImGui::EndDisabled();

  ImGui::SameLine();

  if (m_selectedKeyframe >= static_cast<int>(rig.keyframes.size()))
    m_selectedKeyframe = rig.keyframes.empty() ? -1 : 0;

  bool hasSelection = m_selectedKeyframe >= 0
      && m_selectedKeyframe < static_cast<int>(rig.keyframes.size());

  ImGui::BeginDisabled(!ctx || !hasSelection);
  if (ImGui::Button("Update")) {
    rig.keyframes[m_selectedKeyframe].manipulator =
        camera_rig::manipulatorStateFromManipulator(ctx->view.manipulator);
    project.markDirty();
  }
  tsd::ui::tooltipForPreviousItem("Update Selected From Viewport");
  ImGui::EndDisabled();

  ImGui::SameLine();
  ImGui::BeginDisabled(!shot || !hasSelection);
  if (ImGui::Button("Jump")) {
    shot->currentFrame = rig.keyframes[m_selectedKeyframe].frame;
    if (ctx)
      ctx->tsd.animationMgr.setAnimationFrame(shot->currentFrame);
    else
      m_projectContext->applyActiveShot();
  }
  tsd::ui::tooltipForPreviousItem("Jump Viewport To Keyframe");
  ImGui::EndDisabled();

  ImGui::SameLine();
  ImGui::BeginDisabled(!hasSelection);
  if (ImGui::Button("Delete")) {
    rig.keyframes.erase(rig.keyframes.begin() + m_selectedKeyframe);
    m_selectedKeyframe = -1;
    hasSelection = false;
    project.markDirty();
  }
  tsd::ui::tooltipForPreviousItem("Delete Keyframe");
  ImGui::EndDisabled();

  if (ImGui::BeginTable(
          "keyframes", 5, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
    ImGui::TableSetupColumn(
        "", ImGuiTableColumnFlags_WidthFixed, ImGui::GetFrameHeight());
    ImGui::TableSetupColumn("Frame");
    ImGui::TableSetupColumn("Name");
    ImGui::TableSetupColumn("Interpolation");
    ImGui::TableSetupColumn("Pose");
    ImGui::TableHeadersRow();

    for (int i = 0; i < static_cast<int>(rig.keyframes.size()); ++i) {
      auto &keyframe = rig.keyframes[i];
      ImGui::PushID(i);
      ImGui::TableNextRow();
      if (m_selectedKeyframe == i) {
        const ImU32 selectedColor = ImGui::GetColorU32(ImGuiCol_Header);
        ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg0, selectedColor);
      }

      ImGui::TableNextColumn();
      if (ImGui::RadioButton("##selected", m_selectedKeyframe == i))
        m_selectedKeyframe = i;
      tsd::ui::tooltipForPreviousItem(
          "Select Keyframe; Double-Click To Jump Viewport To Keyframe");
      if (ImGui::IsItemHovered()
          && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
        m_selectedKeyframe = i;
        if (shot) {
          shot->currentFrame = keyframe.frame;
          if (ctx)
            ctx->tsd.animationMgr.setAnimationFrame(shot->currentFrame);
          else
            m_projectContext->applyActiveShot();
        }
      }

      ImGui::TableNextColumn();
      if (ImGui::InputInt("##frame", &keyframe.frame)) {
        camera_rig::sortKeyframes(rig);
        project.markDirty();
      }

      ImGui::TableNextColumn();
      char name[256]{};
      std::snprintf(name, sizeof(name), "%s", keyframe.name.c_str());
      if (ImGui::InputText("##name", name, sizeof(name))) {
        keyframe.name = name;
        project.markDirty();
      }

      ImGui::TableNextColumn();
      int interpolation =
          cameraInterpolationIndex(keyframe.interpolationToNext);
      const char *items[] = {
          "Hold", "Linear", "Ease Out", "Ease In", "Ease Out + In"};
      if (ImGui::Combo("##interp", &interpolation, items, 5)) {
        keyframe.interpolationToNext =
            cameraInterpolationFromIndex(interpolation);
        project.markDirty();
      }

      ImGui::TableNextColumn();
      const auto &pose = keyframe.manipulator.orbit;
      ImGui::Text(
          "%.2f %.2f %.2f", pose.azeldist.x, pose.azeldist.y, pose.azeldist.z);
      ImGui::PopID();
    }

    ImGui::EndTable();
  }

  if (hasSelection) {
    ImGui::Separator();
    auto &keyframe = rig.keyframes[m_selectedKeyframe];
    const auto label = keyframe.name.empty()
        ? ("Selected Keyframe Pose: frame " + std::to_string(keyframe.frame))
        : ("Selected Keyframe Pose: " + keyframe.name);
    if (buildUI_poseEditor(label.c_str(), keyframe.manipulator))
      project.markDirty();
  }

  if (hasSelection && ImGui::IsWindowHovered()
      && ImGui::IsMouseClicked(ImGuiMouseButton_Left)
      && !ImGui::IsAnyItemHovered()) {
    m_selectedKeyframe = -1;
  }
}

void CameraRigEditor::pollPendingFileIO()
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
    if (m_projectContext->loadCameraRigArchive(filename, &error)) {
      m_selectedRig = static_cast<int>(project.cameraRigs.size()) - 1;
      m_selectedKeyframe = -1;
    } else {
      m_ioError = error;
      ImGui::OpenPopup("Camera Rig Archive Error");
    }
  } else if (request == PendingFileIO::Save) {
    if (!m_projectContext->saveCameraRigArchive(
            rigToSave, withTsdExtension(filename), &error)) {
      m_ioError = error;
      ImGui::OpenPopup("Camera Rig Archive Error");
    }
  }
}

void CameraRigEditor::buildUI()
{
  if (!m_projectContext)
    return;

  pollPendingFileIO();
  syncSelectionToActiveShot();
  buildUI_rigControls();

  auto &project = m_projectContext->project();
  if (project.cameraRigs.empty())
    return;

  if (m_selectedRig >= static_cast<int>(project.cameraRigs.size()))
    m_selectedRig = 0;

  buildUI_keyframes(project.cameraRigs[m_selectedRig]);
}

} // namespace tsd::scivis_studio
