// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "ExportNanoVDBFileDialog.h"

// tsd
#include "tsd/core/Logging.hpp"

// tsd_io
#include "tsd/io/serialization.hpp"
#include "tsd/io/serialization.hpp"

//tsd ui
#include "tsd/ui/imgui/Application.h"
#include "tsd/ui/imgui/modals/BlockingTaskModal.h"

// imgui
#include "imgui.h"

namespace tsd::ui::imgui {

ExportNanoVDBFileDialog::ExportNanoVDBFileDialog(Application *app)
    : Modal(app, "ExportVDBFileDialog")
{}

ExportNanoVDBFileDialog::~ExportNanoVDBFileDialog() = default;

void ExportNanoVDBFileDialog::buildUI()
{
  constexpr int MAX_LENGTH = 2000;
  m_filename.reserve(MAX_LENGTH);

  static std::string outPath;
  if (ImGui::Button("...")) {
    outPath.clear();
    m_app->getFilenameFromDialog(outPath, true);
  }

  if (!outPath.empty()) {
    m_filename = outPath;
    outPath.clear();
  }

  ImGui::SameLine();

  auto text_cb = [](ImGuiInputTextCallbackData *cbd) {
    auto &fname = *(std::string *)cbd->UserData;
    fname.resize(cbd->BufTextLen);
    return 0;
  };

  ImGui::InputText("##filename",
      m_filename.data(),
      MAX_LENGTH,
      ImGuiInputTextFlags_CallbackEdit,
      text_cb,
      &m_filename);

  //////////

  ImGui::NewLine();

  // Undefined value parameters
  ImGui::Checkbox("Enable undefined value", &m_enableUndefinedValue);
  
  if (m_enableUndefinedValue) {
    ImGui::InputFloat("Undefined value", &m_undefinedValue);
  }

  ImGui::NewLine();

  ImGuiIO &io = ImGui::GetIO();
  if (ImGui::Button("cancel") || ImGui::IsKeyDown(ImGuiKey_Escape))
    this->hide();

  ImGui::SameLine();

  if (ImGui::Button("export")) {
    this->hide();

    auto doExport = [&]() {
      auto *core = appCore();
      auto selectedObject = core->tsd.selectedObject;
      if (!selectedObject) {
        core::logError("[ExportVDBFileDialog] No object selected for VDB export.");
        return;
      }

      if (selectedObject->subtype() != core::tokens::volume::transferFunction1D) {
        core::logError(
            "[ExportVDBFileDialog] Selected object is not a transfer function 1D.");
        return;
      }
      auto spatialFieldObject = selectedObject->parameterValueAsObject("value");

      if (!spatialFieldObject || !spatialFieldObject->subtype() == core::tokens::volume::structuredRegular) {
        core::logError(
            "[ExportVDBFileDialog] Selected TransferFunction1D does not reference a structured regular volume.");
        return;
      }

      io::export_StructuredRegularVolumeToVDB(
          static_cast<const core::SpatialField*>(spatialFieldObject),
          m_filename.c_str(),
          m_enableUndefinedValue,
          m_undefinedValue);
    };

    if (!appCore()->windows.taskModal)
      doExport();
    else {
      appCore()->windows.taskModal->activate(
          doExport, "Please Wait: Importing Data...");
    }
  }
}

} // namespace tsd::ui::imgui
