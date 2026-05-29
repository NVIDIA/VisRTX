// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "CameraRigEditor.h"

#include "tsd/rendering/view/ManipulatorToTSD.hpp"
#include "tsd/scene/objects/Camera.hpp"
#include "tsd/ui/imgui/tsd_ui_imgui.h"

#include "imgui.h"

#include <algorithm>

namespace tsd::scivis_studio {


namespace {

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

void CameraRigEditor::buildUI()
{
  if (!m_projectContext)
    return;

  auto &project = m_projectContext->project();
  auto *shot = project::activeShot(project);
  if (!shot) {
    ImGui::TextDisabled("No active shot");
    return;
  }

  auto *ctx = m_projectContext->appContext();
  auto &rig = shot->cameraRig;

  if (ImGui::Button("Set View")) {
    rig.current = shot_camera_rig::manipulatorStateFromManipulator(ctx->view.manipulator);
    project.markDirty();
  }
  tsd::ui::tooltipForPreviousItem("Set Rig View From Viewport");

  ImGui::SameLine();

  if (ImGui::Button("Capture")) {
    CameraKeyframe keyframe;
    keyframe.frame = shot->currentFrame;
    keyframe.name = "Frame " + std::to_string(shot->currentFrame);
    keyframe.manipulator =
        shot_camera_rig::manipulatorStateFromManipulator(ctx->view.manipulator);
    rig.keyframes.push_back(std::move(keyframe));
    shot_camera_rig::sortKeyframes(rig);
    m_selectedKeyframe = static_cast<int>(rig.keyframes.size()) - 1;
    project.markDirty();
  }
  tsd::ui::tooltipForPreviousItem("Capture Keyframe At Current Frame");

  ImGui::SameLine();

  if (m_selectedKeyframe >= static_cast<int>(rig.keyframes.size()))
    m_selectedKeyframe = rig.keyframes.empty() ? -1 : 0;

  const bool hasSelection = m_selectedKeyframe >= 0
      && m_selectedKeyframe < static_cast<int>(rig.keyframes.size());

  ImGui::BeginDisabled(!hasSelection);
  if (ImGui::Button("Update")) {
    rig.keyframes[m_selectedKeyframe].manipulator =
        shot_camera_rig::manipulatorStateFromManipulator(ctx->view.manipulator);
    project.markDirty();
  }
  tsd::ui::tooltipForPreviousItem("Update Selected From Viewport");
  ImGui::SameLine();
  if (ImGui::Button("Jump")) {
    shot->currentFrame = rig.keyframes[m_selectedKeyframe].frame;
    if (ctx)
      ctx->tsd.animationMgr.setAnimationFrame(shot->currentFrame);
    else
      m_projectContext->applyActiveShot();
  }
  tsd::ui::tooltipForPreviousItem("Jump Viewport To Keyframe");
  ImGui::SameLine();
  if (ImGui::Button("Delete")) {
    rig.keyframes.erase(rig.keyframes.begin() + m_selectedKeyframe);
    m_selectedKeyframe = -1;
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
        shot->currentFrame = keyframe.frame;
        if (ctx)
          ctx->tsd.animationMgr.setAnimationFrame(shot->currentFrame);
        else
          m_projectContext->applyActiveShot();
      }

      ImGui::TableNextColumn();
      if (ImGui::InputInt("##frame", &keyframe.frame)) {
        shot_camera_rig::sortKeyframes(rig);
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

  if (hasSelection && ImGui::IsWindowHovered()
      && ImGui::IsMouseClicked(ImGuiMouseButton_Left)
      && !ImGui::IsAnyItemHovered()) {
    m_selectedKeyframe = -1;
  }
}

} // namespace tsd::scivis_studio
