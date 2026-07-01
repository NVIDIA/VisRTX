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

enum class ArrayDataPolicy
{
  IncludeData,
  ProxyOnly
};

bool serialize_SceneArchive(const scene::Scene &scene,
    core::DataNode &archive,
    ArrayDataPolicy arrayData = ArrayDataPolicy::IncludeData);
ArchiveValidationResult validate_SceneArchive(core::DataNode &archive);
bool deserialize_SceneArchive(scene::Scene &scene,
    core::DataNode &archive,
    ArchiveValidationResult *validation = nullptr);

bool save_SceneArchive(const scene::Scene &scene,
    const char *filename,
    ArrayDataPolicy arrayData = ArrayDataPolicy::IncludeData);
bool load_SceneArchive(scene::Scene &scene,
    const char *filename,
    ArchiveValidationResult *validation = nullptr);

} // namespace tsd::io
