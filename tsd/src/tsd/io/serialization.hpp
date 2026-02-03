// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <limits>
#include "tsd/core/scene/Scene.hpp"
#include "tsd/rendering/view/Manipulator.hpp"

namespace tsd::io {

using namespace tsd::core;

// NanoVDB quantization precision options
enum class VDBPrecision
{
  Float32, // No quantization (32-bit float)
  Fp4, // 4-bit fixed-point (~8:1 compression)
  Fp8, // 8-bit fixed-point (~4:1 compression)
  Fp16, // 16-bit fixed-point (~2:1 compression)
  FpN, // Variable bit fixed-point
  Half // IEEE 16-bit half float
};

// clang-format off

// Parameters //

void parameterToNode(const Parameter &p, core::DataNode &node);
void nodeToParameter(core::DataNode &node, Parameter &p);
void nodeToObjectParameters(core::DataNode &node, Object &obj);

// Objects //

void objectToNode(const Object &obj, core::DataNode &node);
void nodeToObject(core::DataNode &node, Object &obj);
void nodeToObjectMetadata(core::DataNode &node, Object &obj);
void nodeToNewObject(Scene &scene, core::DataNode &node);

// Camera poses //

void cameraPoseToNode(const rendering::CameraPose &pose, core::DataNode &node);
void nodeToCameraPose(core::DataNode &node, rendering::CameraPose &pose);

// Arrays //

void arrayToNode(const Array &arr, core::DataNode &node, bool forceProxyArrays = false);

// Layers //

void layerToNode(Layer &layer, core::DataNode &node);
void nodeToLayer(core::DataNode &rootNode, Layer &layer, Scene &scene);

// Scenes //

void save_Scene(Scene &scene, const char *filename);
void save_Scene(Scene &scene, core::DataNode &root, bool forceProxyArrays);
void load_Scene(Scene &scene, const char *filename);
void load_Scene(Scene &scene, core::DataNode &root);

void export_SceneToUSD(
    Scene &scene, const char *filename, int framesPerSecond = 30);
void export_StructuredVolumeToNanoVDB(
  const SpatialField* spatialField,
  std::string_view outputFilename,
  bool useUndefinedValue = false,
  float undefinedValue = std::numeric_limits<float>::quiet_NaN(),
  VDBPrecision precision = VDBPrecision::Fp16,
  bool enableDithering = false
);

// clang-format on

} // namespace tsd::io
