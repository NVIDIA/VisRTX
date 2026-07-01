// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// tsd_io
#include "tsd/io/archives/ArchiveValidation.hpp"
// tsd_scene
#include "tsd/scene/Layer.hpp"

namespace tsd::core {
struct DataNode;
} // namespace tsd::core

namespace tsd::io {

bool serialize_LayerSubtreeArchive(
    scene::LayerNodeRef subtree, core::DataNode &archive);
ArchiveValidationResult validate_LayerSubtreeArchive(core::DataNode &archive);
scene::LayerNodeRef deserialize_LayerSubtreeArchive(
    scene::LayerNodeRef destination, core::DataNode &archive);

bool save_LayerSubtreeArchive(
    scene::LayerNodeRef subtree, const char *filename);
scene::LayerNodeRef load_LayerSubtreeArchive(
    scene::LayerNodeRef destination, const char *filename);

} // namespace tsd::io
