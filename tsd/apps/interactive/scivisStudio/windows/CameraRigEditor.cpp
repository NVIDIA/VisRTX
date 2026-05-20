// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "CameraRigEditor.h"

#include "tsd/rendering/view/ManipulatorToTSD.hpp"
#include "tsd/scene/objects/Camera.hpp"

#include "imgui.h"

#include <algorithm>

namespace tsd::scivis_studio {

static void tooltipForPreviousItem(const char *text)
{
  if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
    ImGui::SetTooltip("%s", text);
}

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
  auto *shot = activeShot(project);
  if (!shot) {
    ImGui::TextDisabled("No active shot");
    return;
  }

  auto *ctx = m_projectContext->appContext();
  auto &rig = shot->cameraRig;

  if (ImGui::Button("Set View")) {
    rig.current = manipulatorStateFromManipulator(ctx->view.manipulator);
    project.markDirty();
  }
  tooltipForPreviousItem("Set Rig View From Viewport");

  ImGui::SameLine();

  if (ImGui::Button("Capture")) {
    CameraKeyframe keyframe;
    keyframe.frame = shot->currentFrame;
    keyframe.name = "Frame " + std::to_string(shot->currentFrame);
    keyframe.manipulator =
        manipulatorStateFromManipulator(ctx->view.manipulator);
    rig.keyframes.push_back(std::move(keyframe));
    sortKeyframes(rig);
    m_selectedKeyframe = static_cast<int>(rig.keyframes.size()) - 1;
    project.markDirty();
  }
  tooltipForPreviousItem("Capture Keyframe At Current Frame");

  ImGui::SameLine();

  if (m_selectedKeyframe >= static_cast<int>(rig.keyframes.size()))
    m_selectedKeyframe = rig.keyframes.empty() ? -1 : 0;

  const bool hasSelection = m_selectedKeyframe >= 0
      && m_selectedKeyframe < static_cast<int>(rig.keyframes.size());

  ImGui::BeginDisabled(!hasSelection);
  if (ImGui::Button("Update")) {
    rig.keyframes[m_selectedKeyframe].manipulator =
        manipulatorStateFromManipulator(ctx->view.manipulator);
    project.markDirty();
  }
  tooltipForPreviousItem("Update Selected From Viewport");
  ImGui::SameLine();
  if (ImGui::Button("Jump")) {
    shot->currentFrame = rig.keyframes[m_selectedKeyframe].frame;
    if (ctx)
      ctx->tsd.animationMgr.setAnimationFrame(shot->currentFrame);
    else
      m_projectContext->applyActiveShot();
  }
  tooltipForPreviousItem("Jump Viewport To Keyframe");
  ImGui::SameLine();
  if (ImGui::Button("Delete")) {
    rig.keyframes.erase(rig.keyframes.begin() + m_selectedKeyframe);
    m_selectedKeyframe = -1;
    project.markDirty();
  }
  tooltipForPreviousItem("Delete Keyframe");
  ImGui::EndDisabled();

  if (ImGui::BeginTable(
          "keyframes", 4, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
    ImGui::TableSetupColumn("Frame");
    ImGui::TableSetupColumn("Name");
    ImGui::TableSetupColumn("Interpolation");
    ImGui::TableSetupColumn("Pose");
    ImGui::TableHeadersRow();

    for (int i = 0; i < static_cast<int>(rig.keyframes.size()); ++i) {
      auto &keyframe = rig.keyframes[i];
      ImGui::PushID(i);
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      if (ImGui::Selectable("##select",
              m_selectedKeyframe == i,
              ImGuiSelectableFlags_SpanAllColumns))
        m_selectedKeyframe = i;
      ImGui::SameLine();
      if (ImGui::InputInt("##frame", &keyframe.frame)) {
        sortKeyframes(rig);
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
          keyframe.interpolationToNext == CameraInterpolation::Hold ? 0 : 1;
      const char *items[] = {"Hold", "Linear"};
      if (ImGui::Combo("##interp", &interpolation, items, 2)) {
        keyframe.interpolationToNext = interpolation == 0
            ? CameraInterpolation::Hold
            : CameraInterpolation::Linear;
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
}

} // namespace tsd::scivis_studio
