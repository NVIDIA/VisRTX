// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "ImportFileDialog.h"
#include "BlockingTaskModal.h"
// SDL
#include <SDL3/SDL_dialog.h>
// tsd_io
#include "tsd/io/importers.hpp"

namespace tsd::ui::imgui {

ImportFileDialog::ImportFileDialog(tsd::app::Core *core)
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
      "E57XYZ",
      "NBODY",
      "PLY",
      "OBJ",
      "USD",
      "HDRI",
      "VOLUME",
      "SWC",
      "PDB",
      "XYZDP",
      "HSMESH",
      "NEURAL"};

  ImGui::Combo("importer type", &m_selectedFileType, importers, 15);

  static std::string outPath;
  if (ImGui::Button("...")) {
    outPath.clear();
#if 0
    m_core->getFilenameFromDialog(outPath);
#endif
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
    this->hide();

    auto doLoad = [&]() {
      auto &ctx = m_core->tsd.ctx;
      auto *layer = m_core->tsd.ctx.defaultLayer();
      auto importRoot = m_core->tsd.selectedNode;
      if (!importRoot)
        importRoot = layer->root();

      auto selectedFileType =
          static_cast<app::ImporterType>(m_selectedFileType);
      if (selectedFileType == app::ImporterType::PLY)
        tsd::io::import_PLY(ctx, m_filename.c_str(), importRoot);
      else if (selectedFileType == app::ImporterType::OBJ)
        tsd::io::import_OBJ(ctx, m_filename.c_str(), importRoot);
      else if (selectedFileType == app::ImporterType::USD)
        tsd::io::import_USD(ctx, m_filename.c_str(), importRoot);
      else if (selectedFileType == app::ImporterType::ASSIMP)
        tsd::io::import_ASSIMP(ctx, m_filename.c_str(), importRoot, false);
      else if (selectedFileType == app::ImporterType::ASSIMP_FLAT)
        tsd::io::import_ASSIMP(ctx, m_filename.c_str(), importRoot, true);
      else if (selectedFileType == app::ImporterType::DLAF)
        tsd::io::import_DLAF(ctx, m_filename.c_str(), importRoot);
      else if (selectedFileType == app::ImporterType::E57XYZ)
        tsd::io::import_E57XYZ(ctx, m_filename.c_str());
      else if (selectedFileType == app::ImporterType::NBODY)
        tsd::io::import_NBODY(ctx, m_filename.c_str(), importRoot);
      else if (selectedFileType == app::ImporterType::HDRI)
        tsd::io::import_HDRI(ctx, m_filename.c_str(), importRoot);
      else if (selectedFileType == app::ImporterType::SWC)
        tsd::io::import_SWC(ctx, m_filename.c_str(), importRoot);
      else if (selectedFileType == app::ImporterType::PDB)
        tsd::io::import_PDB(ctx, m_filename.c_str(), importRoot);
      else if (selectedFileType == app::ImporterType::XYZDP)
        tsd::io::import_XYZDP(ctx, m_filename.c_str());
      else if (selectedFileType == app::ImporterType::HSMESH)
        tsd::io::import_HSMESH(ctx, m_filename.c_str(), importRoot);
      else if (selectedFileType == app::ImporterType::VOLUME)
        tsd::io::import_volume(ctx, m_filename.c_str());
      else if (selectedFileType == app::ImporterType::NEURAL)
        tsd::io::import_PT(ctx, m_filename.c_str(), importRoot);
      ctx.signalLayerChange(layer);
    };

    if (!m_core->windows.taskModal)
      doLoad();
    else {
      m_core->windows.taskModal->activate(
          doLoad, "Please Wait: Importing Data...");
    }
  }
}

} // namespace tsd::ui::imgui
