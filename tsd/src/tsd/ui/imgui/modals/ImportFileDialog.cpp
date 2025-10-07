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
      "ASSIMP",
      "ASSIMP_FLAT",
      "AXYZ",
      "DLAF",
      "E57XYZ",
      "NBODY",
      "PLY",
      "OBJ",
      "USD",
      "HDRI",
      "VOLUME",
      "SWC",
      "PDB",
      "TRK",
      "XYZDP",
      "HSMESH",
      "NEURAL",
      "GLTF",
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
      auto importRoot = core->tsd.selectedNode;
      if (!importRoot)
        importRoot = layer->root();

      auto selectedFileType =
          static_cast<app::ImporterType>(m_selectedFileType);
      if (selectedFileType == app::ImporterType::PLY)
        tsd::io::import_PLY(scene, m_filename.c_str(), importRoot);
      else if (selectedFileType == app::ImporterType::OBJ)
        tsd::io::import_OBJ(scene, m_filename.c_str(), importRoot);
      else if (selectedFileType == app::ImporterType::USD)
        tsd::io::import_USD(scene, m_filename.c_str(), importRoot);
      else if (selectedFileType == app::ImporterType::ASSIMP)
        tsd::io::import_ASSIMP(scene, m_filename.c_str(), importRoot, false);
      else if (selectedFileType == app::ImporterType::ASSIMP_FLAT)
        tsd::io::import_ASSIMP(scene, m_filename.c_str(), importRoot, true);
      else if (selectedFileType == app::ImporterType::DLAF)
        tsd::io::import_DLAF(scene, m_filename.c_str(), importRoot);
      else if (selectedFileType == app::ImporterType::E57XYZ)
        tsd::io::import_E57XYZ(scene, m_filename.c_str());
      else if (selectedFileType == app::ImporterType::NBODY)
        tsd::io::import_NBODY(scene, m_filename.c_str(), importRoot);
      else if (selectedFileType == app::ImporterType::HDRI)
        tsd::io::import_HDRI(scene, m_filename.c_str(), importRoot);
      else if (selectedFileType == app::ImporterType::SWC)
        tsd::io::import_SWC(scene, m_filename.c_str(), importRoot);
      else if (selectedFileType == app::ImporterType::PDB)
        tsd::io::import_PDB(scene, m_filename.c_str(), importRoot);
      else if (selectedFileType == app::ImporterType::TRK)
        tsd::io::import_TRK(scene, m_filename.c_str(), importRoot);
      else if (selectedFileType == app::ImporterType::XYZDP)
        tsd::io::import_XYZDP(scene, m_filename.c_str());
      else if (selectedFileType == app::ImporterType::HSMESH)
        tsd::io::import_HSMESH(scene, m_filename.c_str(), importRoot);
      else if (selectedFileType == app::ImporterType::VOLUME)
        tsd::io::import_volume(scene, m_filename.c_str());
      else if (selectedFileType == app::ImporterType::NEURAL)
        tsd::io::import_PT(scene, m_filename.c_str(), importRoot);
      else if (selectedFileType == app::ImporterType::GLTF)
        tsd::io::import_GLTF(scene, m_filename.c_str(), importRoot);
      else if (selectedFileType == app::ImporterType::AXYZ)
        tsd::io::import_AXYZ(scene, m_filename.c_str(), importRoot);
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
