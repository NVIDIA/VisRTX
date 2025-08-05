// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "AppSettingsDialog.h"
// tsd_ui
#include "tsd/ui/imgui/tsd_ui_imgui.h"

namespace tsd::ui::imgui {

AppSettingsDialog::AppSettingsDialog(tsd::app::Core *core)
    : Modal(core, "AppSettings")
{
  if (core->offline.renderer.activeRenderer < 0)
    core->setOfflineRenderingLibrary(core->commandLine.libraryList[0]);
}

void AppSettingsDialog::buildUI()
{
  buildUI_applicationSettings();
  ImGui::NewLine();
  ImGui::Separator();
  buildUI_offlineRenderSettings();
  ImGui::Separator();
  ImGui::NewLine();

  if (ImGui::Button("close") || ImGui::IsKeyDown(ImGuiKey_Escape))
    this->hide();
}

void AppSettingsDialog::applySettings()
{
  ImGuiIO &io = ImGui::GetIO();
  io.FontGlobalScale = m_core->windows.fontScale;

  ImGuiStyle &style = ImGui::GetStyle();
  style.WindowRounding = m_core->windows.uiRounding;
  style.ChildRounding = m_core->windows.uiRounding;
  style.FrameRounding = m_core->windows.uiRounding;
  style.ScrollbarRounding = m_core->windows.uiRounding;
  style.GrabRounding = m_core->windows.uiRounding;
  style.PopupRounding = m_core->windows.uiRounding;
}

void AppSettingsDialog::buildUI_applicationSettings()
{
  ImGui::Text("Application Settings:");
  ImGui::Indent(tsd::ui::INDENT_AMOUNT);

  bool doUpdate = false;

  doUpdate |= ImGui::DragFloat(
      "font size", &m_core->windows.fontScale, 0.01f, 0.5f, 4.f);

  doUpdate |= ImGui::DragFloat(
      "rounding", &m_core->windows.uiRounding, 0.01f, 0.f, 12.f);

  if (doUpdate)
    applySettings();

  ImGui::Unindent(tsd::ui::INDENT_AMOUNT);
}

void AppSettingsDialog::buildUI_offlineRenderSettings()
{
  ImGui::Text("Offline Render Settings (tsdRender):");
  ImGui::Indent(tsd::ui::INDENT_AMOUNT);

  // Frame //

  ImGui::Text("Frame:");
  ImGui::DragInt("##width", (int *)&m_core->offline.frame.width, 1, 10, 10000);
  ImGui::SameLine();
  ImGui::Text("x");
  ImGui::SameLine();
  ImGui::DragInt(
      "##height", (int *)&m_core->offline.frame.height, 1, 10, 10000);
  ImGui::SameLine();
  ImGui::Text("size");

  ImGui::DragInt("samples",
      (int *)&m_core->offline.frame.samples,
      1,
      1,
      std::numeric_limits<int>::max());

  // Depth of Field //

  ImGui::Separator();
  ImGui::Text("Depth-of-Field:");
  ImGui::DragFloat("apertureRadius",
      &m_core->offline.camera.apertureRadius,
      1,
      0.f,
      std::numeric_limits<float>::max());
  ImGui::DragFloat("focusDistance",
      &m_core->offline.camera.focusDistance,
      1,
      0.f,
      std::numeric_limits<float>::max());

  // Renderer //

  ImGui::Separator();
  ImGui::Text("Renderer:");

  if (ImGui::InputText("##ANARI library",
          &m_core->offline.renderer.libraryName,
          ImGuiInputTextFlags_EnterReturnsTrue)) {
    m_core->setOfflineRenderingLibrary(m_core->offline.renderer.libraryName);
  }

  ImGui::SameLine();
  if (ImGui::BeginCombo("##library_combo", "", ImGuiComboFlags_NoPreview)) {
    for (size_t n = 0; n < m_core->commandLine.libraryList.size(); n++) {
      if (ImGui::Selectable(m_core->commandLine.libraryList[n].c_str(), false))
        m_core->setOfflineRenderingLibrary(m_core->commandLine.libraryList[n]);
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
      &m_core->offline.renderer.activeRenderer,
      comboGetRendererSubtype,
      &m_core->offline.renderer.rendererObjects,
      m_core->offline.renderer.rendererObjects.size());

  {
    ImGui::Indent(tsd::ui::INDENT_AMOUNT);
    auto &activeRenderer = m_core->offline.renderer.activeRenderer;
    tsd::ui::buildUI_object(
        m_core->offline.renderer.rendererObjects[activeRenderer],
        m_core->tsd.ctx,
        false);
    ImGui::Unindent(tsd::ui::INDENT_AMOUNT);
  }

  ImGui::Unindent(tsd::ui::INDENT_AMOUNT);
}

} // namespace tsd::ui::imgui
