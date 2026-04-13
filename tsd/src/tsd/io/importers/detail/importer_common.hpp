// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/animation/Animation.hpp"
#include "tsd/core/ColorMapUtil.hpp"
#include "tsd/scene/Scene.hpp"
// std
#include <cstdio>
#include <string>
#include <unordered_map>
#include <vector>
#if TSD_USE_VTK
// vtk
#include <vtkDataArray.h>
#endif

namespace tsd::io {

std::string pathOf(const std::string &filepath);
std::string fileOf(const std::string &filepath);
std::string extensionOf(const std::string &filepath);
bool isAbsolute(const std::string &filepath);

std::vector<std::string> splitString(const std::string &s, char delim);

tsd::scene::ArrayRef readArray(
    tsd::scene::Scene &scene, anari::DataType elementType, std::FILE *fp);

using TextureCache = std::unordered_map<std::string, tsd::scene::ArrayRef>;
tsd::scene::SamplerRef importTexture(tsd::scene::Scene &scene,
    std::string filepath,
    TextureCache &cache,
    bool isLinear = false);

tsd::scene::SamplerRef makeDefaultColorMapSampler(
    tsd::scene::Scene &scene, const tsd::math::float2 &range);

// Transfer function import functions
tsd::core::TransferFunction importTransferFunction(const std::string &filepath);

// Sample a TransferFunction into a 256-entry RGBA colormap and apply it to a
// volume (sets color array, valueRange, and opacityControlPoints metadata).
void applyTransferFunction(tsd::scene::Scene &scene,
    tsd::scene::VolumeRef volume,
    const tsd::core::TransferFunction &transferFunction);

bool calcTangentsForTriangleMesh(const tsd::math::uint3 *indices,
    const tsd::math::float3 *vertexPositions,
    const tsd::math::float3 *vertexNormals,
    const tsd::math::float2 *texCoords,
    tsd::math::float4 *tangents,
    size_t numIndices,
    size_t numVertices);

#if TSD_USE_VTK
anari::DataType vtkTypeToANARIType(
    int vtkType, int numComps, const char *errorIdentifier = "");
tsd::scene::ArrayRef makeArray1DFromVTK(tsd::scene::Scene &scene,
    vtkDataArray *array,
    const char *errorIdentifier = "");
tsd::scene::ArrayRef makeArray3DFromVTK(tsd::scene::Scene &scene,
    vtkDataArray *array,
    size_t w,
    size_t h,
    size_t d,
    const char *errorIdentifier = "");
#endif

// Animation helpers ///////////////////////////////////////////////////////////

// Create a linear time base [0..1] with `count` evenly spaced samples.
std::vector<float> makeLinearTimeBase(size_t count);

// Build bindings for the "one value-array per parameter" pattern (e.g. camera
// animation where each Array has N elements of the parameter's scalar type).
void addValueTimeStepBindings(tsd::animation::Animation &anim,
    tsd::scene::Object *target,
    const std::vector<tsd::core::Token> &paramNames,
    const std::vector<tsd::scene::ObjectUsePtr<tsd::scene::Array>> &dataArrays,
    const std::vector<float> &timeBase,
    tsd::animation::InterpolationRule interp =
        tsd::animation::InterpolationRule::STEP);

// Build bindings for the "array-of-arrays per parameter" pattern (e.g. geometry
// animation where each timestep swaps a different Array object).
void addArrayTimeStepBindings(tsd::animation::Animation &anim,
    tsd::scene::Object *target,
    const std::vector<tsd::core::Token> &paramNames,
    const std::vector<std::vector<tsd::scene::ObjectUsePtr<tsd::scene::Array>>>
        &arraysPerParam,
    const std::vector<float> &timeBase);

// Build a TransformBinding from a sequence of mat4 frames (decomposes each
// frame into rotation quaternion, translation, and scale).
void addTransformStepBinding(tsd::animation::Animation &anim,
    tsd::scene::LayerNodeRef target,
    const std::vector<tsd::math::mat4> &frames,
    const std::vector<float> &timeBase);

} // namespace tsd::io
