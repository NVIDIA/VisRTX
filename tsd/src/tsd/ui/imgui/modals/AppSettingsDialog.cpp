// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "AppSettingsDialog.h"
// tsd_ui
#include "tsd/ui/imgui/Application.h"
#include "tsd/ui/imgui/tsd_ui_imgui.h"

namespace tsd::ui::imgui {

AppSettingsDialog::AppSettingsDialog(Application *app)
    : Modal(app, "AppSettings")
{
  auto *core = appCore();
  if (core->offline.renderer.activeRenderer < 0)
    core->setOfflineRenderingLibrary(core->commandLine.libraryList[0]);
}

void AppSettingsDialog::buildUI()
{
  buildUI_applicationSettings();
  ImGui::Separator();
  buildUI_offlineRenderSettings();
  ImGui::Separator();
  ImGui::NewLine();

  if (ImGui::Button("close") || ImGui::IsKeyDown(ImGuiKey_Escape))
    this->hide();
}

void AppSettingsDialog::applySettings()
{
  const auto *config = m_app->uiConfig();

  ImGuiIO &io = ImGui::GetIO();
  io.FontGlobalScale = config->fontScale;

  ImGuiStyle &style = ImGui::GetStyle();
  style.WindowRounding = config->rounding;
  style.ChildRounding = config->rounding;
  style.FrameRounding = config->rounding;
  style.ScrollbarRounding = config->rounding;
  style.GrabRounding = config->rounding;
  style.PopupRounding = config->rounding;
}

void AppSettingsDialog::buildUI_applicationSettings()
{
  auto *core = appCore();

  ImGui::Text("Application Settings:");
  ImGui::Indent(tsd::ui::INDENT_AMOUNT);

  bool doUpdate = false;

  auto *config = m_app->uiConfig();

  doUpdate |=
      ImGui::DragFloat("font size", &config->fontScale, 0.01f, 0.5f, 4.f);

  doUpdate |= ImGui::DragFloat("rounding", &config->rounding, 0.01f, 0.f, 12.f);

  bool useFlat = core->anari.useFlatRenderIndex();
  if (ImGui::Checkbox("use flat render index", &useFlat))
    core->anari.setUseFlatRenderIndex(useFlat);

  if (ImGui::IsItemHovered())
    ImGui::SetTooltip("Check this option to bypass instancing of objects.");

  if (doUpdate)
    applySettings();

  ImGui::Unindent(tsd::ui::INDENT_AMOUNT);
}

void AppSettingsDialog::buildUI_offlineRenderSettings()
{
  auto *core = appCore();

  ImGui::Text("Offline Render Settings (tsdRender):");
  ImGui::Indent(tsd::ui::INDENT_AMOUNT);

  // Output //

  ImGui::Text("Output:");
  ImGui::InputText("##outputDirectory",
      &core->offline.output.outputDirectory,
      ImGuiInputTextFlags_EnterReturnsTrue);
  ImGui::SameLine();
  ImGui::Text("output directory");

  ImGui::InputText("##filePrefix",
      &core->offline.output.filePrefix,
      ImGuiInputTextFlags_EnterReturnsTrue);
  ImGui::SameLine();
  ImGui::Text("file prefix");

  // Frame //

  ImGui::Text("==== Frame ====");
  ImGui::DragInt("##width", (int *)&core->offline.frame.width, 1, 10, 10000);
  ImGui::SameLine();
  ImGui::Text("x");
  ImGui::SameLine();
  ImGui::DragInt("##height", (int *)&core->offline.frame.height, 1, 10, 10000);
  ImGui::SameLine();
  ImGui::Text("size");

  ImGui::DragInt("samples",
      (int *)&core->offline.frame.samples,
      1,
      1,
      std::numeric_limits<int>::max());

  auto fixupStartEndFrame = [&]() {
    auto &num = core->offline.frame.numFrames;
    auto &start = core->offline.frame.startFrame;
    auto &end = core->offline.frame.endFrame;
    if (end >= num)
      end = num - 1;
    if (start > end)
      end = start;
    if (end < start)
      start = end;
  };

  bool doFix = false;

  doFix |= ImGui::DragInt("total animation frame count",
      (int *)&core->offline.frame.numFrames,
      1,
      1,
      std::numeric_limits<int>::max());

  if (ImGui::IsItemHovered())
    ImGui::SetTooltip("Number of total frames for [0.0-1.0] animation time.");

  ImGui::DragInt("frameIncrement",
      (int *)&core->offline.frame.frameIncrement,
      1,
      1,
      std::max(1, core->offline.frame.numFrames / 2));

  if (ImGui::IsItemHovered())
    ImGui::SetTooltip("Render every {N} frames");

  ImGui::Checkbox("render subset", &core->offline.frame.renderSubset);

  ImGui::BeginDisabled(!core->offline.frame.renderSubset);

  doFix |= ImGui::DragInt("start frame offset",
      (int *)&core->offline.frame.startFrame,
      1,
      0,
      core->offline.frame.numFrames - 1);

  if (ImGui::IsItemHovered())
    ImGui::SetTooltip("Offset into total frame count (when rendering subset)");

  doFix |= ImGui::DragInt("end frame offset",
      (int *)&core->offline.frame.endFrame,
      1,
      core->offline.frame.startFrame,
      core->offline.frame.numFrames - 1);

  if (ImGui::IsItemHovered()) {
    ImGui::SetTooltip(
        "Stop at this frame (when rendering subset),"
        " -1 means go to last frame in full animation");
  }

  ImGui::EndDisabled();

  if (doFix)
    fixupStartEndFrame();

  // Camera //

  ImGui::Separator();
  ImGui::Text("==== Camera ====");
  ImGui::DragFloat("apertureRadius",
      &core->offline.camera.apertureRadius,
      1,
      0.f,
      std::numeric_limits<float>::max());
  ImGui::DragFloat("focusDistance",
      &core->offline.camera.focusDistance,
      1,
      0.f,
      std::numeric_limits<float>::max());

  // Build camera list
  std::vector<std::string> cameraNames = {"<none>"};
  m_menuCameraRefs.resize(1);
  m_menuCameraRefs[0] = {};
  int currentSelection = 0;

  const auto &cameraDB = appCore()->tsd.scene.objectDB().camera;
  tsd::core::foreach_item_const(cameraDB, [&](const auto *cam) {
    if (cam) {
      cameraNames.push_back(cam->name());
      m_menuCameraRefs.push_back(cam->self());
      if (core->offline.camera.cameraIndex == cam->index()) {
        currentSelection = static_cast<int>(cameraNames.size() - 1);
      }
    }
  });

  if (ImGui::Combo(
          "Select",
          &currentSelection,
          [](void *data, int idx, const char **out) {
            auto *names = (std::vector<std::string> *)data;
            *out = (*names)[idx].c_str();
            return true;
          },
          &cameraNames,
          static_cast<int>(cameraNames.size()))) {
    if (currentSelection == 0)
      core->offline.camera.cameraIndex = TSD_INVALID_INDEX;
    else {
      core->offline.camera.cameraIndex =
          m_menuCameraRefs[currentSelection].index();
    }
  }

  // Renderer //

  ImGui::Separator();
  ImGui::Text("==== Renderer ====");

  if (ImGui::InputText("##ANARI library",
          &core->offline.renderer.libraryName,
          ImGuiInputTextFlags_EnterReturnsTrue)) {
    core->setOfflineRenderingLibrary(core->offline.renderer.libraryName);
  }

  ImGui::SameLine();
  if (ImGui::BeginCombo("##library_combo", "", ImGuiComboFlags_NoPreview)) {
    for (size_t n = 0; n < core->commandLine.libraryList.size(); n++) {
      if (ImGui::Selectable(core->commandLine.libraryList[n].c_str(), false))
        core->setOfflineRenderingLibrary(core->commandLine.libraryList[n]);
    }
    ImGui::EndCombo();
  }

  ImGui::SameLine();
  ImGui::Text("ANARI library");

  auto comboGetRendererSubtype = [](void *data, int n) -> const char * {
    auto &renderers = *(std::vector<tsd::core::Object> *)data;
    return renderers[n].name().c_str();
  };

  ImGui::Combo("renderer",
      &core->offline.renderer.activeRenderer,
      comboGetRendererSubtype,
      &core->offline.renderer.rendererObjects,
      core->offline.renderer.rendererObjects.size());

  {
    ImGui::Indent(tsd::ui::INDENT_AMOUNT);
    auto &activeRenderer = core->offline.renderer.activeRenderer;
    tsd::ui::buildUI_object(
        core->offline.renderer.rendererObjects[activeRenderer],
        core->tsd.scene,
        false);
    ImGui::Unindent(tsd::ui::INDENT_AMOUNT);
  }

  // AOV Visualization //

  ImGui::Separator();
  ImGui::Text("==== AOV Visualization ====");

  const char *aovItems[] = {"none",
      "depth",
      "albedo",
      "normal",
      "edges",
      "object ID",
      "primitive ID",
      "instance ID"};
  int aovIdx = static_cast<int>(core->offline.aov.aovType);
  if (ImGui::Combo("AOV type", &aovIdx, aovItems, IM_ARRAYSIZE(aovItems))) {
    core->offline.aov.aovType = static_cast<tsd::rendering::AOVType>(aovIdx);
  }

  ImGui::BeginDisabled(
      core->offline.aov.aovType != tsd::rendering::AOVType::DEPTH);
  ImGui::DragFloat("depth min",
      &core->offline.aov.depthMin,
      0.1f,
      0.f,
      core->offline.aov.depthMax);
  ImGui::DragFloat("depth max",
      &core->offline.aov.depthMax,
      0.1f,
      core->offline.aov.depthMin,
      1e20f);
  ImGui::EndDisabled();

  ImGui::BeginDisabled(
      core->offline.aov.aovType != tsd::rendering::AOVType::EDGES);
  ImGui::DragFloat(
      "edge threshold", &core->offline.aov.edgeThreshold, 0.01f, 0.f, 1.f);
  ImGui::Checkbox("invert edges", &core->offline.aov.edgeInvert);
  ImGui::EndDisabled();

  ImGui::Unindent(tsd::ui::INDENT_AMOUNT);
}

} // namespace tsd::ui::imgui
