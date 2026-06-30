// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ProjectContext.h"
#include "tsd/ui/imgui/windows/Window.h"

#include <string>
#include <vector>

namespace tsd::scivis_studio {

struct DatasetEditor : public tsd::ui::imgui::Window
{
  DatasetEditor(
      tsd::ui::imgui::Application *app, ProjectContext *projectContext);
  ~DatasetEditor() override;

  void buildUI() override;

 private:
  enum class PendingFileIO
  {
    None,
    Import,
    Export
  };

  void pollPendingFileIO();
  void buildDiscoveryReview();
  void buildErrorPopup();

  ProjectContext *m_projectContext{nullptr};
  int m_selectedDataset{0};
  PendingFileIO m_pendingFileIO{PendingFileIO::None};
  std::string m_pendingFilename;
  DatasetID m_pendingExportDataset;
  DatasetID m_pendingRemoveDataset;
  bool m_keepRemovedAsset{false};
  DatasetID m_nameBufferDataset;
  std::string m_nameBuffer;
  std::string m_nameError;
  std::string m_ioError;
  std::vector<DatasetCandidate> m_candidates;
  std::vector<char> m_candidateSelected;
  std::vector<std::string> m_candidateNames;
};

} // namespace tsd::scivis_studio
