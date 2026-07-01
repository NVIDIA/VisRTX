// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "DatasetEditor.h"

#include "tsd/ui/imgui/Application.h"
#include "tsd/ui/imgui/windows/Viewport.h"

#include "imgui.h"

#include <cstring>
#include <exception>
#include <filesystem>
#include <utility>
#include <vector>

namespace tsd::scivis_studio {

DatasetEditor::DatasetEditor(tsd::ui::imgui::Application *app,
    ProjectContext *projectContext,
    tsd::ui::imgui::Viewport *viewport)
    : Window(app, "Dataset Editor"),
      m_projectContext(projectContext),
      m_viewport(viewport)
{}

DatasetEditor::~DatasetEditor() = default;

namespace {

std::string withTsdExtension(const std::string &path)
{
  std::filesystem::path file(path);
  if (file.extension().empty())
    file.replace_extension(".tsd");
  return file.string();
}

} // namespace

void DatasetEditor::pollPendingFileIO()
{
  if (m_pendingFileIO == PendingFileIO::None || m_pendingFilename.empty())
    return;

  const auto request = m_pendingFileIO;
  const auto filename = m_pendingFilename;
  const auto exportDataset = m_pendingExportDataset;
  m_pendingFileIO = PendingFileIO::None;
  m_pendingFilename.clear();
  m_pendingExportDataset.clear();

  std::string error;
  if (request == PendingFileIO::Import) {
    if (m_projectContext->importDataset(filename, &error)) {
      m_selectedDataset =
          static_cast<int>(m_projectContext->project().datasets.size()) - 1;
    } else
      m_ioError = error;
  } else if (!m_projectContext->exportDataset(
                 exportDataset, withTsdExtension(filename), &error))
    m_ioError = error;

  if (!m_ioError.empty())
    ImGui::OpenPopup("Dataset IO Error");
}

void DatasetEditor::pollPendingReimport()
{
  if (!m_pendingReimport
      || !m_pendingReimport->complete.load(std::memory_order_acquire))
    return;

  if (!m_pendingReimport->error.empty())
    m_ioError = std::move(m_pendingReimport->error);
  m_pendingReimport.reset();
  if (m_viewport)
    m_viewport->setRenderingEnabled(true);
}

void DatasetEditor::buildErrorPopup()
{
  ImGui::SetNextWindowSize(ImVec2(520.f, 0.f), ImGuiCond_Appearing);
  if (ImGui::BeginPopupModal(
          "Dataset IO Error", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
    ImGui::TextWrapped("%s", m_ioError.c_str());
    if (ImGui::Button("OK") || ImGui::IsKeyPressed(ImGuiKey_Escape)) {
      m_ioError.clear();
      ImGui::CloseCurrentPopup();
    }
    ImGui::EndPopup();
  }
}

void DatasetEditor::buildDiscoveryReview()
{
  ImGui::SetNextWindowSize(ImVec2(760.f, 0.f), ImGuiCond_Appearing);
  if (!ImGui::BeginPopupModal("Review Discovered Datasets",
          nullptr,
          ImGuiWindowFlags_AlwaysAutoResize))
    return;

  ImGui::TextWrapped(
      "Select the standalone dataset files to incorporate into this project.");
  if (m_candidates.empty())
    ImGui::TextDisabled("No unlisted dataset assets were found.");

  for (size_t i = 0; i < m_candidates.size(); ++i) {
    ImGui::PushID(static_cast<int>(i));
    bool selected = m_candidateSelected[i] != 0;
    if (ImGui::Checkbox("##selected", &selected))
      m_candidateSelected[i] = selected;
    ImGui::SameLine();
    ImGui::TextUnformatted(m_candidates[i].file.filename().string().c_str());
    ImGui::SameLine();
    std::vector<char> name(256, '\0');
    std::strncpy(name.data(), m_candidateNames[i].c_str(), name.size() - 1);
    ImGui::SetNextItemWidth(260.f);
    if (ImGui::InputText("##name", name.data(), name.size()))
      m_candidateNames[i] = name.data();
    ImGui::PopID();
  }

  if (ImGui::Button("Incorporate Selected")) {
    for (size_t i = 0; i < m_candidates.size(); ++i) {
      if (!m_candidateSelected[i])
        continue;
      std::string error;
      if (!m_projectContext->incorporateDatasetCandidate(
              m_candidates[i], m_candidateNames[i], &error)) {
        m_ioError += (m_ioError.empty() ? "" : "\n")
            + m_candidates[i].file.filename().string() + ": " + error;
      }
    }
    m_candidates.clear();
    m_candidateSelected.clear();
    m_candidateNames.clear();
    ImGui::CloseCurrentPopup();
  }
  ImGui::SameLine();
  if (ImGui::Button("Cancel") || ImGui::IsKeyPressed(ImGuiKey_Escape)) {
    m_candidates.clear();
    m_candidateSelected.clear();
    m_candidateNames.clear();
    ImGui::CloseCurrentPopup();
  }
  ImGui::EndPopup();
}

void DatasetEditor::buildUI()
{
  if (!m_projectContext)
    return;

  pollPendingFileIO();
  pollPendingReimport();

  if (ImGui::Button("Import...")) {
    m_pendingFileIO = PendingFileIO::Import;
    m_pendingFilename.clear();
    m_app->getFilenameFromDialog(
        m_pendingFilename, tsd::ui::imgui::FileDialogMode::OpenFile);
  }
  ImGui::SameLine();
  if (ImGui::Button("Discover...")) {
    m_candidates = m_projectContext->discoverDatasetCandidates();
    m_candidateSelected.assign(m_candidates.size(), 1);
    m_candidateNames.clear();
    for (const auto &candidate : m_candidates)
      m_candidateNames.push_back(candidate.proposedName);
    ImGui::OpenPopup("Review Discovered Datasets");
  }

  buildDiscoveryReview();
  if (!m_ioError.empty())
    ImGui::OpenPopup("Dataset IO Error");
  buildErrorPopup();

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
  if (m_nameBufferDataset != dataset.id) {
    m_nameBufferDataset = dataset.id;
    m_nameBuffer = dataset.name;
    m_nameError.clear();
  }
  std::vector<char> name(256, '\0');
  std::strncpy(name.data(), m_nameBuffer.c_str(), name.size() - 1);
  const bool submitted = ImGui::InputText(
      "Name", name.data(), name.size(), ImGuiInputTextFlags_EnterReturnsTrue);
  m_nameBuffer = name.data();
  if (submitted || ImGui::IsItemDeactivatedAfterEdit()) {
    std::string error;
    if (m_nameBuffer == dataset.name)
      m_nameError.clear();
    else if (m_projectContext->renameDataset(dataset.id, m_nameBuffer, &error))
      m_nameError.clear();
    else {
      m_nameError = error;
      m_nameBuffer = dataset.name;
    }
  }
  if (!m_nameError.empty())
    ImGui::TextColored(ImVec4(1.f, 0.4f, 0.4f, 1.f), "%s", m_nameError.c_str());

  ImGui::Text("ID: %s", dataset.id.c_str());
  ImGui::Text("Status: %s", dataset::toString(dataset.status));
  ImGui::Text("Source kind: %s", dataset::toString(dataset.sourceKind));
  ImGui::Text("Importer: %s", dataset.importerType.c_str());
  ImGui::TextWrapped("Source: %s", dataset.source.sourcePath.c_str());
  if (dataset.sourceKind == DatasetSourceKind::FileAnimation) {
    ImGui::Text("Frames: %zu", dataset.sourceFiles.size());
    ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_SpanAvailWidth;
    if (dataset.sourceFiles.size() <= 12)
      flags |= ImGuiTreeNodeFlags_DefaultOpen;
    const auto label =
        "Source Files (" + std::to_string(dataset.sourceFiles.size()) + ")";
    if (ImGui::TreeNodeEx(label.c_str(), flags)) {
      for (size_t i = 0; i < dataset.sourceFiles.size(); ++i) {
        const auto &sourceFile = dataset.sourceFiles[i];
        const auto path = std::filesystem::path(sourceFile.path);
        const auto filename = path.filename().string();
        const auto row = std::to_string(i) + "  " + filename;
        ImGui::TextUnformatted(row.c_str());
        if (ImGui::IsItemHovered())
          ImGui::SetTooltip("%s", path.string().c_str());
      }
      ImGui::TreePop();
    }
  }
  ImGui::Text("Root: %s/%zu",
      dataset.rootNode.layerName.c_str(),
      dataset.rootNode.nodeIndex);

  ImGui::BeginDisabled(dataset.status != DatasetStatus::Available);
  if (ImGui::Button("Export...")) {
    m_pendingFileIO = PendingFileIO::Export;
    m_pendingExportDataset = dataset.id;
    m_pendingFilename.clear();
    m_app->getFilenameFromDialog(
        m_pendingFilename, tsd::ui::imgui::FileDialogMode::SaveFile);
  }
  ImGui::EndDisabled();
  ImGui::SameLine();
  ImGui::BeginDisabled(dataset.sourceKind != DatasetSourceKind::Static
      || dataset.source.sourcePath.empty());
  if (ImGui::Button("Reimport")) {
    const auto datasetId = dataset.id;
    m_pendingReimport = std::make_shared<ReimportResult>();
    if (m_viewport)
      m_viewport->setRenderingEnabled(false);
    m_app->showTaskModal(
        [ctx = m_projectContext, datasetId, result = m_pendingReimport]() {
          try {
            ctx->reimportStaticDataset(datasetId, &result->error);
          } catch (const std::exception &e) {
            result->error = std::string("dataset reimport failed: ") + e.what();
          } catch (...) {
            result->error = "dataset reimport failed";
          }
          result->complete.store(true, std::memory_order_release);
        },
        "Reimporting Dataset...");
  }
  ImGui::EndDisabled();
  ImGui::SameLine();
  if (ImGui::Button("Remove...")) {
    m_pendingRemoveDataset = dataset.id;
    m_keepRemovedAsset = false;
    ImGui::OpenPopup("Remove Dataset?");
  }

  if (ImGui::BeginPopupModal(
          "Remove Dataset?", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
    ImGui::TextWrapped(
        "Remove '%s' from the inventory and every shot?", dataset.name.c_str());
    ImGui::Checkbox("Keep managed asset file", &m_keepRemovedAsset);
    if (ImGui::Button("Remove")) {
      std::string error;
      if (!m_projectContext->removeDataset(
              m_pendingRemoveDataset, m_keepRemovedAsset, &error))
        m_ioError = error;
      m_pendingRemoveDataset.clear();
      m_selectedDataset = 0;
      ImGui::CloseCurrentPopup();
    }
    ImGui::SameLine();
    if (ImGui::Button("Cancel")) {
      m_pendingRemoveDataset.clear();
      ImGui::CloseCurrentPopup();
    }
    ImGui::EndPopup();
  }
}

} // namespace tsd::scivis_studio
