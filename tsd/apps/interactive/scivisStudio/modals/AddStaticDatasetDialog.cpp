// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "AddStaticDatasetDialog.h"

#include "tsd/core/Logging.hpp"
#include "tsd/ui/imgui/Application.h"

#include "imgui.h"

#include <array>
#include <cstring>
#include <filesystem>
#include <optional>

namespace tsd::scivis_studio {

namespace {

struct DatasetSourceChoice
{
  const char *name;
  std::optional<tsd::io::ImporterType> importer;
};

constexpr std::array<DatasetSourceChoice, 26> SOURCES = {{
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
    {"PLY", tsd::io::ImporterType::PLY},
    {"POINTSBIN_MULTIFILE", tsd::io::ImporterType::POINTSBIN_MULTIFILE},
    {"PT", tsd::io::ImporterType::PT},
    {"SILO", tsd::io::ImporterType::SILO},
    {"SMESH", tsd::io::ImporterType::SMESH},
    {"SWC", tsd::io::ImporterType::SWC},
    {"TRK", tsd::io::ImporterType::TRK},
    {"USD", tsd::io::ImporterType::USD},
    {"VTP", tsd::io::ImporterType::VTP},
    {"VTU", tsd::io::ImporterType::VTU},
    {"XYZDP", tsd::io::ImporterType::XYZDP},
    {"VOLUME", tsd::io::ImporterType::VOLUME},
    {"TSD Dataset Archive", std::nullopt},
}};

template <size_t N>
void copyToInputBuffer(std::array<char, N> &buffer, const std::string &value)
{
  buffer.fill('\0');
  std::strncpy(buffer.data(), value.c_str(), buffer.size() - 1);
}

} // namespace

AddStaticDatasetDialog::AddStaticDatasetDialog(
    tsd::ui::imgui::Application *app, ProjectContext *projectContext)
    : Modal(app, "Add Static Dataset"), m_projectContext(projectContext)
{}

AddStaticDatasetDialog::~AddStaticDatasetDialog() = default;

void AddStaticDatasetDialog::buildUI()
{
  ImGui::InputText("Name", m_name.data(), m_name.size());

  if (!m_browsedSourcePath.empty()) {
    copyToInputBuffer(m_sourcePath, m_browsedSourcePath);
    m_browsedSourcePath.clear();
  }

  if (ImGui::Button("...##datasetSource")) {
    m_browsedSourcePath.clear();
    m_app->getFilenameFromDialog(
        m_browsedSourcePath, tsd::ui::imgui::FileDialogMode::OpenFile);
  }
  ImGui::SameLine();
  ImGui::InputText("Source Path", m_sourcePath.data(), m_sourcePath.size());

  const char *preview = SOURCES[m_selectedSource].name;
  if (ImGui::BeginCombo("Source", preview)) {
    for (int i = 0; i < static_cast<int>(SOURCES.size()); ++i) {
      const bool selected = i == m_selectedSource;
      if (ImGui::Selectable(SOURCES[i].name, selected))
        m_selectedSource = i;
      if (selected)
        ImGui::SetItemDefaultFocus();
    }
    ImGui::EndCombo();
  }

  ImGui::Spacing();
  if (ImGui::Button("Cancel") || ImGui::IsKeyPressed(ImGuiKey_Escape)) {
    hide();
    return;
  }

  ImGui::SameLine();
  const auto sourceChoice = SOURCES[m_selectedSource];
  const bool loadArchive = !sourceChoice.importer.has_value();
  if (ImGui::Button(loadArchive ? "Load" : "Import")) {
    const std::string name = m_name.data();
    const std::filesystem::path sourcePath = m_sourcePath.data();
    if (sourcePath.empty()) {
      tsd::core::logWarning("[SciVisStudio] Dataset source path is empty");
      return;
    }

    hide();
    m_app->showTaskModal(
        [ctx = m_projectContext, name, sourcePath, sourceChoice]() {
          if (!ctx)
            return;
          if (sourceChoice.importer) {
            ctx->addStaticDataset(name, sourcePath, *sourceChoice.importer);
          } else {
            std::string error;
            auto *dataset = ctx->loadDatasetArchive(sourcePath, &error);
            if (!dataset) {
              tsd::core::logWarning(
                  "[SciVisStudio] Failed to load Dataset Archive: %s",
                  error.c_str());
            } else if (!name.empty()
                && !ctx->renameDataset(dataset->id, name, &error)) {
              tsd::core::logWarning(
                  "[SciVisStudio] Failed to rename loaded Dataset Archive: %s",
                  error.c_str());
            }
          }
        },
        loadArchive ? "Loading Dataset Archive..." : "Importing Dataset...");
  }
}

} // namespace tsd::scivis_studio
