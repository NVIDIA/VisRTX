// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "ImportFileDialog.h"
// SDL
#include <SDL3/SDL_dialog.h>

namespace tsd_viewer {

ImportFileDialog::ImportFileDialog(AppCore *core)
    : Modal(core, "ImportFileDialog"), m_core(core)
{}

ImportFileDialog::~ImportFileDialog() = default;

void ImportFileDialog::buildUI()
{
  constexpr int MAX_LENGTH = 2000;
  m_filename.reserve(MAX_LENGTH);

  bool update = false;

  const char *importers[] = {"ASSIMP",
      "ASSIMP_FLAT",
      "DLAF",
      "NBODY",
      "PLY",
      "OBJ",
      "HDRI",
      "VOLUME",
      "SWC",
      "PDB",
      "XYZDP"};

  ImGui::Combo("importer type", &m_selectedFileType, importers, 10);

  static std::string outPath;
  if (ImGui::Button("...")) {
    outPath.clear();
    m_core->getFilenameFromDialog(outPath);
  }

  if (!outPath.empty()) {
    m_filename = outPath;
    update = true;
    outPath.clear();
  }

  ImGui::SameLine();

  auto text_cb = [](ImGuiInputTextCallbackData *cbd) {
    auto &fname = *(std::string *)cbd->UserData;
    fname.resize(cbd->BufTextLen);
    return 0;
  };

  update |= ImGui::InputText("##filename",
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
    auto &ctx = m_core->tsd.ctx;
    auto *layer = m_core->tsd.ctx.defaultLayer();
    auto importRoot = m_core->tsd.selectedNode;
    if (!importRoot)
      importRoot = layer->root();

    auto selectedFileType =
        static_cast<tsd_viewer::ImporterType>(m_selectedFileType);
    if (selectedFileType == ImporterType::PLY)
      tsd::import_PLY(ctx, m_filename.c_str(), importRoot);
    else if (selectedFileType == ImporterType::OBJ)
      tsd::import_OBJ(ctx, m_filename.c_str(), importRoot);
    else if (selectedFileType == ImporterType::ASSIMP)
      tsd::import_ASSIMP(ctx, m_filename.c_str(), importRoot, false);
    else if (selectedFileType == ImporterType::ASSIMP_FLAT)
      tsd::import_ASSIMP(ctx, m_filename.c_str(), importRoot, true);
    else if (selectedFileType == ImporterType::DLAF)
      tsd::import_DLAF(ctx, m_filename.c_str(), importRoot);
    else if (selectedFileType == ImporterType::NBODY)
      tsd::import_NBODY(ctx, m_filename.c_str(), importRoot);
    else if (selectedFileType == ImporterType::HDRI)
      tsd::import_HDRI(ctx, m_filename.c_str(), importRoot);
    else if (selectedFileType == ImporterType::SWC)
      tsd::import_SWC(ctx, m_filename.c_str(), importRoot);
    else if (selectedFileType == ImporterType::PDB)
      tsd::import_PDB(ctx, m_filename.c_str(), importRoot);
    else if (selectedFileType == ImporterType::XYZDP)
      tsd::import_XYZDP(ctx, m_filename.c_str());
    else if (selectedFileType == ImporterType::VOLUME)
      tsd::import_volume(ctx, m_filename.c_str());
    ctx.signalLayerChange(layer);

    this->hide();
  }
}

} // namespace tsd_viewer
