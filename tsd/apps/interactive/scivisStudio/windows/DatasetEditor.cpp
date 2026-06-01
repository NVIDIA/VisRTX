// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "DatasetEditor.h"

#include "imgui.h"

#include <filesystem>

namespace tsd::scivis_studio {

DatasetEditor::DatasetEditor(
    tsd::ui::imgui::Application *app, ProjectContext *projectContext)
    : Window(app, "Dataset Editor"), m_projectContext(projectContext)
{}

DatasetEditor::~DatasetEditor() = default;

void DatasetEditor::buildUI()
{
  if (!m_projectContext)
    return;

  auto &datasets = m_projectContext->project().datasets;
  if (datasets.empty()) {
    ImGui::TextDisabled("No dataset selected");
    return;
  }

  if (m_selectedDataset >= static_cast<int>(datasets.size()))
    m_selectedDataset = 0;

  const char *preview = datasets[m_selectedDataset].name.c_str();
  if (ImGui::BeginCombo("Dataset", preview)) {
    for (int i = 0; i < static_cast<int>(datasets.size()); ++i) {
      const bool selected = i == m_selectedDataset;
      if (ImGui::Selectable(datasets[i].name.c_str(), selected))
        m_selectedDataset = i;
      if (selected)
        ImGui::SetItemDefaultFocus();
    }
    ImGui::EndCombo();
  }

  auto &dataset = datasets[m_selectedDataset];
  ImGui::Text("ID: %s", dataset.id.c_str());
  ImGui::Text("Status: %s", dataset::toString(dataset.status));
  ImGui::Text("Source kind: %s", dataset::toString(dataset.sourceKind));
  ImGui::Text("Importer: %s", dataset.importerType.c_str());
  ImGui::TextWrapped("Path: %s", dataset.source.absolutePath.c_str());
  if (dataset.sourceKind == DatasetSourceKind::TimeSeries) {
    ImGui::Text("Frames: %zu", dataset.sourceFiles.size());
    ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_SpanAvailWidth;
    if (dataset.sourceFiles.size() <= 12)
      flags |= ImGuiTreeNodeFlags_DefaultOpen;
    const auto label =
        "Source Files (" + std::to_string(dataset.sourceFiles.size()) + ")";
    if (ImGui::TreeNodeEx(label.c_str(), flags)) {
      for (size_t i = 0; i < dataset.sourceFiles.size(); ++i) {
        const auto &sourceFile = dataset.sourceFiles[i];
        const auto path = m_projectContext->resolveSourceFilePath(sourceFile);
        const bool regular =
            m_projectContext->sourceFileIsRegular(sourceFile);
        const auto filename = path.empty()
            ? std::filesystem::path(sourceFile.absolutePath).filename().string()
            : path.filename().string();
        const auto row = std::to_string(i) + "  " + filename;
        if (!regular)
          ImGui::PushStyleColor(
              ImGuiCol_Text, ImVec4(1.f, 0.35f, 0.25f, 1.f));
        ImGui::TextUnformatted(row.c_str());
        if (ImGui::IsItemHovered())
          ImGui::SetTooltip("%s", path.string().c_str());
        if (!regular)
          ImGui::PopStyleColor();
      }
      ImGui::TreePop();
    }
  }
  ImGui::Text("Root: %s/%zu",
      dataset.rootNode.layerName.c_str(),
      dataset.rootNode.nodeIndex);
}

} // namespace tsd::scivis_studio
