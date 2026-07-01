// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// tsd_io
#include "tsd/io/archives/ArchiveValidation.hpp"

namespace tsd::core {
struct DataNode;
} // namespace tsd::core

namespace tsd::scene {
struct Object;
struct Scene;
} // namespace tsd::scene

namespace tsd::io {

bool serialize_ObjectArchive(
    const scene::Object &object, core::DataNode &archive);
ArchiveValidationResult validate_ObjectArchive(core::DataNode &archive);
scene::Object *deserialize_ObjectArchive(scene::Scene &scene,
    core::DataNode &archive,
    ArchiveValidationResult *validation = nullptr);

bool save_ObjectArchive(const scene::Object &object, const char *filename);
scene::Object *load_ObjectArchive(scene::Scene &scene,
    const char *filename,
    ArchiveValidationResult *validation = nullptr);

} // namespace tsd::io
