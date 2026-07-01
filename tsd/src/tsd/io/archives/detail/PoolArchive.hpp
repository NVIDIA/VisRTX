// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// tsd_io
#include "tsd/io/archives/ArchiveValidation.hpp"
// anari
#include <anari/anari_cpp.hpp>
// std
#include <string_view>

namespace tsd::core {
struct DataNode;
} // namespace tsd::core

namespace tsd::scene {
struct Scene;
} // namespace tsd::scene

namespace tsd::io::detail {

bool serializePoolArchive(const scene::Scene &scene,
    core::DataNode &archive,
    anari::DataType objectType,
    std::string_view poolName,
    std::string_view schema);
ArchiveValidationResult validatePoolArchive(core::DataNode &archive,
    anari::DataType objectType,
    std::string_view poolName,
    std::string_view schema);
bool deserializePoolArchive(scene::Scene &scene,
    core::DataNode &archive,
    anari::DataType objectType,
    std::string_view poolName,
    std::string_view schema,
    ArchiveValidationResult *validation);

} // namespace tsd::io::detail
