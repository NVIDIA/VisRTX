// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ProjectContext.h"
#include "tsd/ui/imgui/windows/Window.h"

#include <atomic>
#include <memory>
#include <string>
#include <vector>

namespace tsd::ui::imgui {
struct Viewport;
}

namespace tsd::scivis_studio {

struct DatasetEditor : public tsd::ui::imgui::Window
{
  DatasetEditor(tsd::ui::imgui::Application *app,
      ProjectContext *projectContext,
      tsd::ui::imgui::Viewport *viewport);
  ~DatasetEditor() override;

  void buildUI() override;

 private:
  enum class PendingFileIO
  {
    None,
    Load,
    Save
  };

  struct ReimportResult
  {
    std::atomic_bool complete{false};
    std::string error;
  };

  void pollPendingFileIO();
  void pollPendingReimport();
  void buildDiscoveryReview();
  void buildErrorPopup();

  ProjectContext *m_projectContext{nullptr};
  tsd::ui::imgui::Viewport *m_viewport{nullptr};
  int m_selectedDataset{0};
  PendingFileIO m_pendingFileIO{PendingFileIO::None};
  std::string m_pendingFilename;
  DatasetID m_pendingSaveDataset;
  DatasetID m_pendingRemoveDataset;
  std::shared_ptr<ReimportResult> m_pendingReimport;
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
