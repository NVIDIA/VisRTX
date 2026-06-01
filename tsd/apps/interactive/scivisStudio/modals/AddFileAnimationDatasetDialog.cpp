// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "AddFileAnimationDatasetDialog.h"

#include "tsd/core/Logging.hpp"
#include "tsd/ui/imgui/Application.h"

#include "imgui.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <set>

namespace tsd::scivis_studio {

namespace {

template <size_t N>
void copyToInputBuffer(std::array<char, N> &buffer, const std::string &value)
{
  buffer.fill('\0');
  std::strncpy(buffer.data(), value.c_str(), buffer.size() - 1);
}

int naturalCompareString(const std::string &a, const std::string &b)
{
  size_t ia = 0;
  size_t ib = 0;
  while (ia < a.size() && ib < b.size()) {
    const bool digitA = std::isdigit(static_cast<unsigned char>(a[ia]));
    const bool digitB = std::isdigit(static_cast<unsigned char>(b[ib]));
    if (digitA && digitB) {
      size_t enda = ia;
      size_t endb = ib;
      while (enda < a.size()
          && std::isdigit(static_cast<unsigned char>(a[enda])))
        ++enda;
      while (endb < b.size()
          && std::isdigit(static_cast<unsigned char>(b[endb])))
        ++endb;

      auto na = a.substr(ia, enda - ia);
      auto nb = b.substr(ib, endb - ib);
      na.erase(0, na.find_first_not_of('0'));
      nb.erase(0, nb.find_first_not_of('0'));
      if (na.size() != nb.size())
        return na.size() < nb.size() ? -1 : 1;
      if (na != nb)
        return na < nb ? -1 : 1;
      ia = enda;
      ib = endb;
      continue;
    }

    if (a[ia] != b[ib])
      return a[ia] < b[ib] ? -1 : 1;
    ++ia;
    ++ib;
  }

  if (ia == a.size() && ib == b.size())
    return 0;
  return ia == a.size() ? -1 : 1;
}

bool naturalPathLess(const std::string &a, const std::string &b)
{
  const auto fa = std::filesystem::path(a).filename().string();
  const auto fb = std::filesystem::path(b).filename().string();
  const int filenameCompare = naturalCompareString(fa, fb);
  if (filenameCompare != 0)
    return filenameCompare < 0;
  return naturalCompareString(a, b) < 0;
}

std::string trimmedGeneratedName(std::string name)
{
  while (!name.empty()) {
    const char c = name.back();
    if (c == ' ' || c == '_' || c == '-' || c == '.')
      name.pop_back();
    else
      break;
  }
  return name;
}

std::string commonStemPrefix(const std::vector<std::string> &paths)
{
  if (paths.empty())
    return {};

  std::string prefix = std::filesystem::path(paths.front()).stem().string();
  for (size_t i = 1; i < paths.size() && !prefix.empty(); ++i) {
    const auto stem = std::filesystem::path(paths[i]).stem().string();
    size_t n = 0;
    while (n < prefix.size() && n < stem.size() && prefix[n] == stem[n])
      ++n;
    prefix.resize(n);
  }

  prefix = trimmedGeneratedName(prefix);
  if (!prefix.empty())
    return prefix;
  return std::filesystem::path(paths.front()).stem().string();
}

std::string extensionLabel(const std::string &extension)
{
  return extension.empty() ? std::string("<none>") : extension;
}

bool anySelected(const std::vector<char> &selectedRows)
{
  return std::any_of(
      selectedRows.begin(), selectedRows.end(), [](char selected) {
        return selected != 0;
      });
}

void resizeSelection(std::vector<char> &selectedRows, size_t size)
{
  selectedRows.resize(size, 0);
}

} // namespace

AddFileAnimationDatasetDialog::AddFileAnimationDatasetDialog(
    tsd::ui::imgui::Application *app, ProjectContext *projectContext)
    : Modal(app, "Add File Animation Dataset"),
      m_projectContext(projectContext)
{}

AddFileAnimationDatasetDialog::~AddFileAnimationDatasetDialog() = default;

void AddFileAnimationDatasetDialog::reset()
{
  m_name.fill('\0');
  m_sourcePaths.clear();
  m_browsedSourcePaths.clear();
  m_invalidRows.clear();
  m_selectedRows.clear();
  m_validationMessage.clear();
  m_extensionWarning.clear();
  m_nameEditedByUser = false;
  m_showInvalidRows = false;
}

void AddFileAnimationDatasetDialog::clearValidation()
{
  m_invalidRows.clear();
  m_validationMessage.clear();
  m_extensionWarning.clear();
  m_showInvalidRows = false;
}

void AddFileAnimationDatasetDialog::appendBrowsedFiles()
{
  if (m_browsedSourcePaths.empty())
    return;

  std::sort(
      m_browsedSourcePaths.begin(), m_browsedSourcePaths.end(), naturalPathLess);
  m_sourcePaths.insert(m_sourcePaths.end(),
      m_browsedSourcePaths.begin(),
      m_browsedSourcePaths.end());
  m_browsedSourcePaths.clear();
  resizeSelection(m_selectedRows, m_sourcePaths.size());
  clearValidation();
  updateGeneratedName();
}

void AddFileAnimationDatasetDialog::updateGeneratedName()
{
  if (m_nameEditedByUser)
    return;

  copyToInputBuffer(m_name, commonStemPrefix(m_sourcePaths));
}

bool AddFileAnimationDatasetDialog::validateForImport()
{
  m_invalidRows.assign(m_sourcePaths.size(), false);
  m_validationMessage.clear();
  m_extensionWarning.clear();

  if (m_sourcePaths.empty()) {
    m_validationMessage = "Select at least one frame.";
    return false;
  }

  std::set<std::string> extensions;
  bool ok = true;
  for (size_t i = 0; i < m_sourcePaths.size(); ++i) {
    const std::filesystem::path path = m_sourcePaths[i];
    extensions.insert(path.extension().string());

    std::error_code ec;
    if (!std::filesystem::exists(path, ec) || ec) {
      m_invalidRows[i] = true;
      ok = false;
      continue;
    }
    if (!std::filesystem::is_regular_file(path, ec) || ec) {
      m_invalidRows[i] = true;
      ok = false;
    }
  }

  if (extensions.size() > 1) {
    m_extensionWarning = "Mixed extensions: ";
    bool first = true;
    for (const auto &ext : extensions) {
      if (!first)
        m_extensionWarning += ", ";
      m_extensionWarning += extensionLabel(ext);
      first = false;
    }
  }

  if (!ok)
    m_validationMessage = "One or more selected frames are missing or invalid.";
  return ok;
}

void AddFileAnimationDatasetDialog::buildUI()
{
  appendBrowsedFiles();

  if (ImGui::InputText("Name", m_name.data(), m_name.size()))
    m_nameEditedByUser = true;

  if (ImGui::Button("Add Files..."))
    m_app->getFilenamesFromDialog(m_browsedSourcePaths);
  ImGui::SameLine();
  if (ImGui::Button("Remove") && anySelected(m_selectedRows)) {
    for (int i = static_cast<int>(m_sourcePaths.size()) - 1; i >= 0; --i) {
      if (i < static_cast<int>(m_selectedRows.size()) && m_selectedRows[i])
        m_sourcePaths.erase(m_sourcePaths.begin() + i);
    }
    m_selectedRows.assign(m_sourcePaths.size(), 0);
    clearValidation();
    updateGeneratedName();
  }
  ImGui::SameLine();
  if (ImGui::Button("Clear")) {
    m_sourcePaths.clear();
    m_selectedRows.clear();
    clearValidation();
    updateGeneratedName();
  }

  if (ImGui::Button("Move Up") && anySelected(m_selectedRows)
      && !m_selectedRows.front()) {
    for (size_t i = 1; i < m_sourcePaths.size(); ++i) {
      if (m_selectedRows[i] && !m_selectedRows[i - 1]) {
        std::swap(m_sourcePaths[i], m_sourcePaths[i - 1]);
        std::swap(m_selectedRows[i], m_selectedRows[i - 1]);
      }
    }
    clearValidation();
    updateGeneratedName();
  }
  ImGui::SameLine();
  if (ImGui::Button("Move Down") && anySelected(m_selectedRows)
      && !m_selectedRows.empty() && !m_selectedRows.back()) {
    for (int i = static_cast<int>(m_sourcePaths.size()) - 2; i >= 0; --i) {
      if (m_selectedRows[i] && !m_selectedRows[i + 1]) {
        std::swap(m_sourcePaths[i], m_sourcePaths[i + 1]);
        std::swap(m_selectedRows[i], m_selectedRows[i + 1]);
      }
    }
    clearValidation();
    updateGeneratedName();
  }
  ImGui::SameLine();
  if (ImGui::Button("Sort by Name")) {
    std::sort(m_sourcePaths.begin(), m_sourcePaths.end(), naturalPathLess);
    m_selectedRows.assign(m_sourcePaths.size(), 0);
    clearValidation();
    updateGeneratedName();
  }

  ImGui::Text("Frames: %zu", m_sourcePaths.size());
  if (!m_extensionWarning.empty())
    ImGui::TextWrapped("%s", m_extensionWarning.c_str());
  if (!m_validationMessage.empty())
    ImGui::TextWrapped("%s", m_validationMessage.c_str());

  if (ImGui::BeginChild(
          "FileAnimationFrames", ImVec2(0.f, 240.f), true)) {
    resizeSelection(m_selectedRows, m_sourcePaths.size());
    for (int i = 0; i < static_cast<int>(m_sourcePaths.size()); ++i) {
      const bool invalid =
          m_showInvalidRows && i < static_cast<int>(m_invalidRows.size())
          && m_invalidRows[i];
      if (invalid)
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.f, 0.35f, 0.25f, 1.f));

      const auto filename =
          std::filesystem::path(m_sourcePaths[i]).filename().string();
      const auto label = std::to_string(i) + "  " + filename;
      if (ImGui::Selectable(label.c_str(), m_selectedRows[i])) {
        const bool append = ImGui::GetIO().KeyCtrl || ImGui::GetIO().KeyShift;
        if (!append)
          m_selectedRows.assign(m_sourcePaths.size(), 0);
        m_selectedRows[i] = m_selectedRows[i] ? 0 : 1;
      }
      if (ImGui::IsItemHovered())
        ImGui::SetTooltip("%s", m_sourcePaths[i].c_str());

      if (invalid)
        ImGui::PopStyleColor();
    }
  }
  ImGui::EndChild();

  ImGui::Spacing();
  if (ImGui::Button("Cancel") || ImGui::IsKeyPressed(ImGuiKey_Escape)) {
    reset();
    hide();
    return;
  }

  ImGui::SameLine();
  if (ImGui::Button("Import")) {
    if (!validateForImport()) {
      m_showInvalidRows = true;
      tsd::core::logWarning(
          "[SciVisStudio] File animation dataset validation failed: %s",
          m_validationMessage.c_str());
      return;
    }

    if (!m_extensionWarning.empty())
      tsd::core::logWarning(
          "[SciVisStudio] %s", m_extensionWarning.c_str());

    const std::string name = m_name.data();
    std::vector<std::filesystem::path> sourcePaths;
    sourcePaths.reserve(m_sourcePaths.size());
    for (const auto &path : m_sourcePaths)
      sourcePaths.emplace_back(path);

    reset();
    hide();
    m_app->showTaskModal(
        [ctx = m_projectContext, name, sourcePaths]() {
          if (ctx) {
            ctx->addFileAnimationDataset(name,
                sourcePaths,
                tsd::io::ImporterType::VOLUME_ANIMATION);
          }
        },
        "Importing File Animation Dataset...");
  }
}

} // namespace tsd::scivis_studio
