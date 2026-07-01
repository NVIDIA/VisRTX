// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/io/archives/ArchiveValidation.hpp"
#include "tsd/scene/Layer.hpp"

#include <filesystem>
#include <string>
#include <string_view>

namespace tsd::core {
struct DataNode;
}

namespace tsd::scene {
struct Scene;
}

namespace tsd::scivis_studio {

tsd::io::ArchiveValidationResult validateLightRigArchive(
    tsd::core::DataNode &archive);
bool saveLightRigArchiveFile(tsd::scene::LayerNodeRef root,
    const std::filesystem::path &file,
    std::string_view displayName);
tsd::scene::LayerNodeRef deserializeLightRigArchive(tsd::scene::Scene &scene,
    tsd::core::DataNode &archive,
    tsd::scene::LayerNodeRef destination,
    std::string *displayName = nullptr);
tsd::scene::LayerNodeRef loadLightRigArchiveFile(tsd::scene::Scene &scene,
    const std::filesystem::path &file,
    tsd::scene::LayerNodeRef destination,
    std::string *displayName = nullptr);

} // namespace tsd::scivis_studio
