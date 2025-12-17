// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "ImportFileDialog.h"
#include "tsd/ui/imgui/Application.h"
#include "tsd/ui/imgui/modals/BlockingTaskModal.h"
// SDL
#include <SDL3/SDL_dialog.h>
// tsd_io
#include "tsd/io/importers.hpp"

namespace tsd::ui::imgui {

ImportFileDialog::ImportFileDialog(Application *app)
    : Modal(app, "ImportFileDialog")
{}

ImportFileDialog::~ImportFileDialog() = default;

void ImportFileDialog::buildUI()
{
  constexpr int MAX_LENGTH = 2000;
  m_filename.reserve(MAX_LENGTH);

  const char *importers[] = {
      "AGX",
      "ASSIMP",
      "ASSIMP_FLAT",
      "AXYZ",
      "DLAF",
      "E57XYZ",
      "GLTF",
      "HDRI",
      "HSMESH",
      "NBODY",
      "OBJ",
      "PDB",
      "PLY",
      "POINTSBIN_MULTIFILE",
      "PT (neural)",
      "SILO",
      "SMESH",
      "SMESH_ANIMATION",
      "SWC",
      "TRK",
      "USD",
      "USD2",
      "XYZDP",
      "VOLUME",
      "TSD",
  };

  ImGui::Combo(
      "importer type", &m_selectedFileType, importers, std::size(importers));

  static std::string outPath;
  if (ImGui::Button("...")) {
    outPath.clear();
    m_app->getFilenameFromDialog(outPath);
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

  ImGuiIO &io = ImGui::GetIO();
  if (ImGui::Button("cancel") || ImGui::IsKeyDown(ImGuiKey_Escape))
    this->hide();

  ImGui::SameLine();

  if (ImGui::Button("import")) {
    this->hide();

    auto doLoad = [&]() {
      auto *core = appCore();
      auto &scene = core->tsd.scene;
      auto *layer = core->tsd.scene.defaultLayer();
      auto importRoot = core->getSelected();
      if (!importRoot.valid())
        importRoot = layer->root();
      app::ImportFile file{
          static_cast<app::ImporterType>(m_selectedFileType), m_filename};
      core->importFile(file, importRoot);
      scene.signalLayerChange(layer);
    };

    if (!appCore()->windows.taskModal)
      doLoad();
    else {
      appCore()->windows.taskModal->activate(
          doLoad, "Please Wait: Importing Data...");
    }
  }
}

} // namespace tsd::ui::imgui
