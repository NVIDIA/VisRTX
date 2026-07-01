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

  void showLoadObject(tsd::scene::LayerNodeRef destination);
  void showSaveObject(TSDObjectFileType fileType,
      anari::DataType objectType,
      size_t objectIndex);

  // Layer subtree files (a node and its descendants + referenced objects) //
  void showLoadLayerSubtree(tsd::scene::LayerNodeRef destinationParent);
  void showSaveLayerSubtree(tsd::scene::LayerNodeRef sourceRoot);

  void buildUI() override;

 private:
  enum class Mode
  {
    Load,
    Save
  };

  enum class Kind
  {
    Object,
    LayerSubtree
  };

  const char *fileTypeLabel() const;
  const char *actionLabel() const;
  const char *taskLabel() const;
  anari::DataType anariObjectType() const;

  void loadObjectArchive();
  void saveObjectArchive();
  void loadLayerSubtreeArchive();
  void saveLayerSubtreeArchive();

  std::string m_filename;
  std::string m_dialogFilename;
  Mode m_mode{Mode::Load};
  Kind m_kind{Kind::Object};
  TSDObjectFileType m_fileType{TSDObjectFileType::Surface};
  tsd::scene::LayerNodeRef m_destination;
  tsd::scene::LayerNodeRef m_subtreeNode;
  anari::DataType m_objectType{ANARI_UNKNOWN};
  size_t m_objectIndex{tsd::core::INVALID_INDEX};
};

} // namespace tsd::ui::imgui
