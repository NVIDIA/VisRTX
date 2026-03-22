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
  auto *ctx = appContext();
  const auto &libraryList = ctx->anari.libraryList();
  if (ctx->offline.renderer.activeRenderer < 0)
    ctx->setOfflineRenderingLibrary(libraryList[0]);
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
  auto *ctx = appContext();

  ImGui::Text("Application Settings:");
  ImGui::Indent(tsd::ui::INDENT_AMOUNT);

  bool doUpdate = false;

  auto *config = m_app->uiConfig();

  doUpdate |=
      ImGui::DragFloat("font size", &config->fontScale, 0.01f, 0.5f, 4.f);

  doUpdate |= ImGui::DragFloat("rounding", &config->rounding, 0.01f, 0.f, 12.f);

  auto kind = ctx->anari.renderIndexKind();

  if (ImGui::RadioButton(
          "all layers", kind == tsd::app::RenderIndexKind::ALL_LAYERS))
    ctx->anari.setRenderIndexKind(tsd::app::RenderIndexKind::ALL_LAYERS);
  if (ImGui::IsItemHovered())
    ImGui::SetTooltip("Full render index with instancing support.");
  ImGui::SameLine();
  if (ImGui::RadioButton("flat", kind == tsd::app::RenderIndexKind::FLAT))
    ctx->anari.setRenderIndexKind(tsd::app::RenderIndexKind::FLAT);
  if (ImGui::IsItemHovered())
    ImGui::SetTooltip("Bypass instancing of objects.");

  if (doUpdate)
    applySettings();

  ImGui::Unindent(tsd::ui::INDENT_AMOUNT);
}

void AppSettingsDialog::buildUI_offlineRenderSettings()
{
  auto *ctx = appContext();

  ImGui::Text("Offline Render Settings (tsdRender):");
  ImGui::Indent(tsd::ui::INDENT_AMOUNT);

  // Output //

  ImGui::Text("Output:");
  ImGui::InputText("##outputDirectory",
      &ctx->offline.output.outputDirectory,
      ImGuiInputTextFlags_EnterReturnsTrue);
  ImGui::SameLine();
  ImGui::Text("output directory");

  ImGui::InputText("##filePrefix",
      &ctx->offline.output.filePrefix,
      ImGuiInputTextFlags_EnterReturnsTrue);
  ImGui::SameLine();
  ImGui::Text("file prefix");

  // Frame //

  ImGui::Text("==== Frame ====");
  ImGui::DragInt("##width", (int *)&ctx->offline.frame.width, 1, 10, 10000);
  ImGui::SameLine();
  ImGui::Text("x");
  ImGui::SameLine();
  ImGui::DragInt("##height", (int *)&ctx->offline.frame.height, 1, 10, 10000);
  ImGui::SameLine();
  ImGui::Text("size");

  ImGui::DragInt("samples",
      (int *)&ctx->offline.frame.samples,
      1,
      1,
      std::numeric_limits<int>::max());

  auto fixupStartEndFrame = [&]() {
    auto &num = ctx->offline.frame.numFrames;
    auto &start = ctx->offline.frame.startFrame;
    auto &end = ctx->offline.frame.endFrame;
    if (end >= num)
      end = num - 1;
    if (start > end)
      end = start;
    if (end < start)
      start = end;
  };

  bool doFix = false;

  doFix |= ImGui::DragInt("total animation frame count",
      (int *)&ctx->offline.frame.numFrames,
      1,
      1,
      std::numeric_limits<int>::max());

  if (ImGui::IsItemHovered())
    ImGui::SetTooltip("Number of total frames for [0.0-1.0] animation time.");

  ImGui::DragInt("frameIncrement",
      (int *)&ctx->offline.frame.frameIncrement,
      1,
      1,
      std::max(1, ctx->offline.frame.numFrames / 2));

  if (ImGui::IsItemHovered())
    ImGui::SetTooltip("Render every {N} frames");

  ImGui::Checkbox("render subset", &ctx->offline.frame.renderSubset);

  ImGui::BeginDisabled(!ctx->offline.frame.renderSubset);

  doFix |= ImGui::DragInt("start frame offset",
      (int *)&ctx->offline.frame.startFrame,
      1,
      0,
      ctx->offline.frame.numFrames - 1);

  if (ImGui::IsItemHovered())
    ImGui::SetTooltip("Offset into total frame count (when rendering subset)");

  doFix |= ImGui::DragInt("end frame offset",
      (int *)&ctx->offline.frame.endFrame,
      1,
      ctx->offline.frame.startFrame,
      ctx->offline.frame.numFrames - 1);

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
      &ctx->offline.camera.apertureRadius,
      1,
      0.f,
      std::numeric_limits<float>::max());
  ImGui::DragFloat("focusDistance",
      &ctx->offline.camera.focusDistance,
      1,
      0.f,
      std::numeric_limits<float>::max());

  // Build camera list
  std::vector<std::string> cameraNames = {"<none>"};
  m_menuCameraRefs.resize(1);
  m_menuCameraRefs[0] = {};
  int currentSelection = 0;

  const auto &cameraDB = appContext()->tsd.scene.objectDB().camera;
  tsd::core::foreach_item_const(cameraDB, [&](const auto *cam) {
    if (cam) {
      cameraNames.push_back(cam->name());
      m_menuCameraRefs.push_back(cam->self());
      if (ctx->offline.camera.cameraIndex == cam->index()) {
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
      ctx->offline.camera.cameraIndex = TSD_INVALID_INDEX;
    else {
      ctx->offline.camera.cameraIndex =
          m_menuCameraRefs[currentSelection].index();
    }
  }

  // Renderer //

  ImGui::Separator();
  ImGui::Text("==== Renderer ====");

  if (ImGui::InputText("##ANARI library",
          &ctx->offline.renderer.libraryName,
          ImGuiInputTextFlags_EnterReturnsTrue)) {
    ctx->setOfflineRenderingLibrary(ctx->offline.renderer.libraryName);
  }

  ImGui::SameLine();
  const auto &libraryList = appContext()->anari.libraryList();
  if (ImGui::BeginCombo("##library_combo", "", ImGuiComboFlags_NoPreview)) {
    for (size_t n = 0; n < libraryList.size(); n++) {
      if (ImGui::Selectable(libraryList[n].c_str(), false))
        ctx->setOfflineRenderingLibrary(libraryList[n]);
    }
    ImGui::EndCombo();
  }

  ImGui::SameLine();
  ImGui::Text("ANARI library");

  auto comboGetRendererSubtype = [](void *data, int n) -> const char * {
    auto &renderers = *(std::vector<tsd::scene::Object> *)data;
    return renderers[n].name().c_str();
  };

  ImGui::Combo("renderer",
      &ctx->offline.renderer.activeRenderer,
      comboGetRendererSubtype,
      &ctx->offline.renderer.rendererObjects,
      ctx->offline.renderer.rendererObjects.size());

  {
    ImGui::Indent(tsd::ui::INDENT_AMOUNT);
    auto &activeRenderer = ctx->offline.renderer.activeRenderer;
    tsd::ui::buildUI_object(
        ctx->offline.renderer.rendererObjects[activeRenderer],
        ctx->tsd.scene,
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
  int aovIdx = static_cast<int>(ctx->offline.aov.aovType);
  if (ImGui::Combo("AOV type", &aovIdx, aovItems, IM_ARRAYSIZE(aovItems))) {
    ctx->offline.aov.aovType = static_cast<tsd::rendering::AOVType>(aovIdx);
  }

  ImGui::BeginDisabled(
      ctx->offline.aov.aovType != tsd::rendering::AOVType::DEPTH);
  ImGui::DragFloat("depth min",
      &ctx->offline.aov.depthMin,
      0.1f,
      0.f,
      ctx->offline.aov.depthMax);
  ImGui::DragFloat("depth max",
      &ctx->offline.aov.depthMax,
      0.1f,
      ctx->offline.aov.depthMin,
      1e20f);
  ImGui::EndDisabled();

  ImGui::BeginDisabled(
      ctx->offline.aov.aovType != tsd::rendering::AOVType::EDGES);
  ImGui::Checkbox("invert edges", &ctx->offline.aov.edgeInvert);
  ImGui::EndDisabled();

  ImGui::Unindent(tsd::ui::INDENT_AMOUNT);
}

} // namespace tsd::ui::imgui
