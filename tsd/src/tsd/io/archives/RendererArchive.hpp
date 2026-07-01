// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// tsd_io
#include "tsd/io/archives/ArchiveValidation.hpp"

namespace tsd::core {
struct DataNode;
} // namespace tsd::core

namespace tsd::scene {
struct Scene;
} // namespace tsd::scene

namespace tsd::io {

bool serialize_RendererArchive(
    const scene::Scene &scene, core::DataNode &archive);
ArchiveValidationResult validate_RendererArchive(core::DataNode &archive);
bool deserialize_RendererArchive(scene::Scene &scene,
    core::DataNode &archive,
    ArchiveValidationResult *validation = nullptr);
bool save_RendererArchive(const scene::Scene &scene, const char *filename);
bool load_RendererArchive(scene::Scene &scene,
    const char *filename,
    ArchiveValidationResult *validation = nullptr);

} // namespace tsd::io
