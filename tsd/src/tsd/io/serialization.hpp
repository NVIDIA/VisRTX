// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <limits>
#include "tsd/core/scene/Scene.hpp"
#include "tsd/rendering/view/Manipulator.hpp"

namespace tsd::io {

using namespace tsd::core;

// clang-format off

void objectToNode(const Object &obj, core::DataNode &node);
void nodeToObject(core::DataNode &node, Object &obj);

void cameraPoseToNode(const rendering::CameraPose &pose, core::DataNode &node);
void nodeToCameraPose(core::DataNode &node, rendering::CameraPose &pose);

void save_Scene(Scene &scene, const char *filename);
void save_Scene(Scene &scene, core::DataNode &root);
void load_Scene(Scene &scene, const char *filename);
void load_Scene(Scene &scene, core::DataNode &root);

void export_SceneToUSD(Scene &scene, const char *filename);
void export_StructuredRegularVolumeToVDB(const SpatialField *spatialField, std::string_view outputFilename, bool useUndefinedValue, float undefinedValue);

// clang-format on

} // namespace tsd::io
