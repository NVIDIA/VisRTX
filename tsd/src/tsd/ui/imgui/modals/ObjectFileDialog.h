// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Modal.h"
// std
#include <cstddef>
#include <string>

namespace tsd::ui::imgui {

enum class TSDObjectFileType
{
  Surface,
  Volume
};

struct ObjectFileDialog : public Modal
{
  ObjectFileDialog(Application *app);
  ~ObjectFileDialog() override;

  void showImport(
      TSDObjectFileType fileType, tsd::scene::LayerNodeRef importRoot);
  void showExport(TSDObjectFileType fileType,
      anari::DataType objectType,
      size_t objectIndex);

  void buildUI() override;

 private:
  enum class Mode
  {
    Import,
    Export
  };

  const char *fileTypeLabel() const;
  const char *actionLabel() const;
  const char *taskLabel() const;
  anari::DataType anariObjectType() const;

  void importFile();
  void exportFile();

  std::string m_filename;
  std::string m_dialogFilename;
  Mode m_mode{Mode::Import};
  TSDObjectFileType m_fileType{TSDObjectFileType::Surface};
  tsd::scene::LayerNodeRef m_importRoot;
  anari::DataType m_exportObjectType{ANARI_UNKNOWN};
  size_t m_exportObjectIndex{tsd::core::INVALID_INDEX};
};

} // namespace tsd::ui::imgui
