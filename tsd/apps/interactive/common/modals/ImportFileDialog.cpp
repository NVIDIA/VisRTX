// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "ImportFileDialog.h"
// glfw
#include <GLFW/glfw3.h>
// nfd (from anari::anari_viewer)
#include "nfd.h"

namespace tsd_viewer {

ImportFileDialog::ImportFileDialog(AppCore *ctx)
    : Modal("ImportFileDialog"), m_core(ctx)
{
  NFD_Init();
}

ImportFileDialog::~ImportFileDialog()
{
  NFD_Quit();
}

void ImportFileDialog::buildUI()
{
  constexpr int MAX_LENGTH = 2000;
  m_filename.reserve(MAX_LENGTH);

  bool update = false;

  const char *importers[] = {
      "ASSIMP", "ASSIMP_FLAT", "DLAF", "NBODY", "PLY", "OBJ", "HDRI"};

  ImGui::Combo("importer type", &m_selectedFileType, importers, 7);

  if (ImGui::Button("...")) {
    nfdchar_t *outPath = nullptr;
    nfdfilteritem_t filterItem[3] = {
        {"All Supported Files", "gltf,glb,obj,dlaf,nbody,ply,hdri"},
        {"glTF Files", "gltf,glb"},
        {"OBJ Files", "obj"}};
    nfdresult_t result = NFD_OpenDialog(&outPath, filterItem, 3, nullptr);
    if (result == NFD_OKAY) {
      m_filename = std::string(outPath).c_str();
      update = true;
      NFD_FreePath(outPath);
    } else {
      tsd::logWarning("NFD Error: %s\n", NFD_GetError());
    }
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
  if (ImGui::Button("cancel") || io.KeysDown[GLFW_KEY_ESCAPE])
    this->hide();

  ImGui::SameLine();

  if (ImGui::Button("import")) {
    auto &ctx = m_core->tsd.ctx;
    auto importRoot = m_core->tsd.selectedNode;
    if (!importRoot)
      importRoot = m_core->tsd.ctx.tree.root();

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

    ctx.signalInstanceTreeChange();

    this->hide();
  }
}

} // namespace tsd_viewer
