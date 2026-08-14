// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "ImportFileDialog.h"
#include "tsd/ui/imgui/Application.h"
#include "tsd/ui/imgui/modals/BlockingTaskModal.h"
// SDL
#include <SDL3/SDL_dialog.h>
// tsd_io
#include "tsd/io/importers.hpp"
// tsd_core
#include "tsd/core/Logging.hpp"

namespace tsd::ui::imgui {

namespace {

// The label shown in the combo, paired with the type it selects. Pairing them
// is what keeps the two in step: the combo index used to be cast straight to
// ImporterType, so every entry after the first gap named one importer and ran
// another.
struct ImporterChoice
{
  const char *name;
  tsd::io::ImporterType type;
};

constexpr ImporterChoice IMPORTERS[] = {
    {"AGX", tsd::io::ImporterType::AGX},
    {"ASSIMP", tsd::io::ImporterType::ASSIMP},
    {"ASSIMP_FLAT", tsd::io::ImporterType::ASSIMP_FLAT},
    {"AXYZ", tsd::io::ImporterType::AXYZ},
    {"DLAF", tsd::io::ImporterType::DLAF},
    {"E57XYZ", tsd::io::ImporterType::E57XYZ},
    {"ENSIGHT", tsd::io::ImporterType::ENSIGHT},
    {"GLTF", tsd::io::ImporterType::GLTF},
    {"HDRI", tsd::io::ImporterType::HDRI},
    {"HSMESH", tsd::io::ImporterType::HSMESH},
    {"NBODY", tsd::io::ImporterType::NBODY},
    {"OBJ", tsd::io::ImporterType::OBJ},
    {"PDB", tsd::io::ImporterType::PDB},
    {"PBRT", tsd::io::ImporterType::PBRT},
    {"PLY", tsd::io::ImporterType::PLY},
    {"POINTSBIN_MULTIFILE", tsd::io::ImporterType::POINTSBIN_MULTIFILE},
    {"PT (neural)", tsd::io::ImporterType::PT},
    {"SILO", tsd::io::ImporterType::SILO},
    {"SMESH", tsd::io::ImporterType::SMESH},
    {"SMESH_ANIMATION", tsd::io::ImporterType::SMESH_ANIMATION},
    {"SWC", tsd::io::ImporterType::SWC},
    {"SWC_SDF", tsd::io::ImporterType::SWC_SDF},
    {"TRK", tsd::io::ImporterType::TRK},
    {"USD", tsd::io::ImporterType::USD},
    {"USD_MTLX", tsd::io::ImporterType::USD_MTLX},
    {"VTP", tsd::io::ImporterType::VTP},
    {"VTU", tsd::io::ImporterType::VTU},
    {"XYZDP", tsd::io::ImporterType::XYZDP},
    {"VOLUME", tsd::io::ImporterType::VOLUME},
    {"VOLUME_ANIMATION", tsd::io::ImporterType::VOLUME_ANIMATION},
};

} // namespace

ImportFileDialog::ImportFileDialog(Application *app)
    : Modal(app, "ImportFileDialog")
{}

ImportFileDialog::~ImportFileDialog() = default;

void ImportFileDialog::buildUI()
{
  constexpr int MAX_LENGTH = 2000;
  m_filename.reserve(MAX_LENGTH);

  const char *importerNames[std::size(IMPORTERS)] = {};
  for (size_t i = 0; i < std::size(IMPORTERS); i++)
    importerNames[i] = IMPORTERS[i].name;

  ImGui::Combo("importer type",
      &m_selectedFileType,
      importerNames,
      std::size(importerNames));

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
      auto *ctx = appContext();
      auto &scene = ctx->tsd.scene;
      auto *layer = ctx->tsd.scene.defaultLayer();
      auto importRoot = ctx->getFirstSelected();
      if (!importRoot.valid())
        importRoot = layer->root();
      tsd::io::ImportFile file{IMPORTERS[m_selectedFileType].type, m_filename};
      tsd::io::import_file(scene, ctx->tsd.animationMgr, file, importRoot);
      scene.signalLayerStructureChanged(layer);
    };

    m_app->showTaskModal(doLoad, "Please Wait: Importing Data...");
  }
}

} // namespace tsd::ui::imgui
