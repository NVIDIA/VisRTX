// Copyright 2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// tsd
#include "tsd/core/Logging.hpp"
#include "tsd/core/scene/objects/Array.hpp"
#include "tsd/core/scene/objects/SpatialField.hpp"
#include "tsd/io/serialization.hpp"

// nanovdb
#include <nanovdb/io/IO.h>
#include <nanovdb/tools/CreateNanoGrid.h>
#include <nanovdb/tools/GridBuilder.h>
#include <nanovdb/tools/GridStats.h>

// std
#include <array>
#include <string_view>
#include <type_traits>

namespace tsd::io {

template <typename T>
void doExportStructuredRegularVolumeToNanoVDB(const T *data,
    math::float3 origin,
    math::float3 spacing,
    math::int3 dims,
    std::string_view outputFilename,
    bool useUndefinedValue,
    float undefinedValue,
    VDBPrecision precision,
    bool enableDithering)
{
  tsd::core::logStatus(
      "[export_StructuredRegularVolumeToVDB] Volume dimensions: %u x %u x %u",
      dims.x,
      dims.y,
      dims.z);
  tsd::core::logStatus(
      "[export_StructuredRegularVolumeToVDB] Origin: (%.3f, %.3f, %.3f)",
      origin.x,
      origin.y,
      origin.z);
  tsd::core::logStatus(
      "[export_StructuredRegularVolumeToVDB] Spacing: (%.3f, %.3f, %.3f)",
      spacing.x,
      spacing.y,
      spacing.z);

  // Adjust spacing to account node vs cell storage
  spacing = spacing / dims * (dims - math::int3(1));

  // Create a build grid, for now, always float. We can always use nanovdb
  // toolings to convert later if needed.
  auto buildGrid = std::make_shared<nanovdb::tools::build::Grid<float>>(
      useUndefinedValue ? undefinedValue : 0.0f);

  // Grid location. Seems that NanoVDB expects uniform voxel size, so we use the
  // average of spacing components This is ugly, but I am not sure how to deal
  // with that otherwise. Basically, the final built grid is supposed to handle
  // the non-uniform spacing, but the builder API only allows for uniform
  // scaling. Let's assume this works and force getting the transform as
  // mutable.
  constexpr size_t MatrixSize = 3;
  const std::array<std::array<double, MatrixSize>, MatrixSize> mat = {{
      {{spacing.x, 0.0, 0.0}}, // row 0
      {{0.0, spacing.y, 0.0}}, // row 1
      {{0.0, 0.0, spacing.z}} // row 2
  }};
  const std::array<std::array<double, MatrixSize>, MatrixSize> invMat = {{
      {{1.0 / spacing.x, 0.0, 0.0}}, // row 0
      {{0.0, 1.0 / spacing.y, 0.0}}, // row 1
      {{0.0, 0.0, 1.0 / spacing.z}} // row 2
  }};
  const std::array<double, MatrixSize> trans = {origin.x, origin.y, origin.z};
  const_cast<nanovdb::Map &>(buildGrid->map())
      .set(mat.data(), invMat.data(), trans.data());

  auto acc = buildGrid->getAccessor();
  for (size_t k = 0; k < static_cast<size_t>(dims.z); ++k) {
    for (size_t j = 0; j < static_cast<size_t>(dims.y); ++j) {
      for (size_t i = 0; i < static_cast<size_t>(dims.x); ++i) {
        const size_t idx = i + j * dims.x + k * dims.x * dims.y;
        double value = data[idx];

        if constexpr (std::is_integral_v<T>) {
          // For integral types, normalize and consider signedness
          if constexpr (std::is_signed_v<T>) {
            if (value >= 0) {
              value = value / std::numeric_limits<T>::max();
            } else {
              value = value / -std::numeric_limits<T>::min();
            }
          } else {
            value = value / std::numeric_limits<T>::max();
          }
        }

        if (useUndefinedValue && std::abs(value - undefinedValue) < 1e-6) {
          continue;
        }

        const nanovdb::Coord ijk(
            static_cast<int>(i), static_cast<int>(j), static_cast<int>(k));
        acc.setValue(ijk, value);
      }
    }
  }

  // Calculate inactive cell compression (before quantization) and value range
  auto totalVoxelsCount = static_cast<size_t>(dims.x) * dims.y * dims.z;
  size_t activeVoxelsCount = 0;
  float minVal = std::numeric_limits<float>::max();
  float maxVal = std::numeric_limits<float>::lowest();

  // Count active voxels and compute value range by checking what was set
  auto acc_check = buildGrid->getAccessor();
  for (size_t k = 0; k < dims.z; ++k) {
    for (size_t j = 0; j < dims.y; ++j) {
      for (size_t i = 0; i < dims.x; ++i) {
        nanovdb::Coord ijk(
            static_cast<int>(i), static_cast<int>(j), static_cast<int>(k));
        if (acc_check.isValueOn(ijk)) {
          activeVoxelsCount++;
          float val = acc_check.getValue(ijk);
          minVal = std::min(minVal, val);
          maxVal = std::max(maxVal, val);
        }
      }
    }
  }

  size_t uncompressedSize = sizeof(T) * totalVoxelsCount;
  size_t uncompressedActiveSize =
      sizeof(float) * activeVoxelsCount; // Build grid uses float
  float inactiveCellCompression = activeVoxelsCount > 0
      ? (1.0f - static_cast<float>(activeVoxelsCount) / totalVoxelsCount)
      : 0.0f;

  tsd::core::logStatus(
      "[export_StructuredRegularVolumeToVDB] Active voxels: %zu/%zu (%.1f%% inactive cell compression)",
      activeVoxelsCount,
      totalVoxelsCount,
      inactiveCellCompression * 100.0f);

  float valueRange = (activeVoxelsCount > 0) ? (maxVal - minVal) : 0.0f;

  // Validate quantization choice
  if (precision == VDBPrecision::Fp4 && valueRange > 30.0f) {
    tsd::core::logWarning(
        "[export_StructuredRegularVolumeToVDB] Fp4 quantization with value range %.2f may introduce significant error (recommended < 30.0)",
        valueRange);
  } else if (precision == VDBPrecision::Fp8 && valueRange > 500.0f) {
    tsd::core::logWarning(
        "[export_StructuredRegularVolumeToVDB] Fp8 quantization with value range %.2f may introduce significant error (recommended < 500.0)",
        valueRange);
  }

  // Convert build grid to NanoVDB grid with quantization
  nanovdb::GridHandle<> handle;

  switch (precision) {
  case VDBPrecision::Float32:
    handle = nanovdb::tools::createNanoGrid(
        *buildGrid, nanovdb::tools::StatsMode::All, nanovdb::CheckMode::Full);
    break;
  case VDBPrecision::Fp4:
    handle = nanovdb::tools::createNanoGrid<nanovdb::tools::build::Grid<float>,
        nanovdb::Fp4>(*buildGrid,
        nanovdb::tools::StatsMode::All,
        nanovdb::CheckMode::Full,
        0,
        enableDithering);
    break;
  case VDBPrecision::Fp8:
    handle = nanovdb::tools::createNanoGrid<nanovdb::tools::build::Grid<float>,
        nanovdb::Fp8>(*buildGrid,
        nanovdb::tools::StatsMode::All,
        nanovdb::CheckMode::Full,
        0,
        enableDithering);
    break;
  case VDBPrecision::Fp16:
    handle = nanovdb::tools::createNanoGrid<nanovdb::tools::build::Grid<float>,
        nanovdb::Fp16>(*buildGrid,
        nanovdb::tools::StatsMode::All,
        nanovdb::CheckMode::Full,
        0,
        enableDithering);
    break;
  case VDBPrecision::FpN:
    handle = nanovdb::tools::createNanoGrid<nanovdb::tools::build::Grid<float>,
        nanovdb::FpN>(*buildGrid,
        nanovdb::tools::StatsMode::All,
        nanovdb::CheckMode::Full,
        0,
        enableDithering,
        nanovdb::tools::AbsDiff());
    break;
  case VDBPrecision::Half:
    tsd::core::logWarning(
        "[export_StructuredRegularVolumeToVDB] Half precision not supported in this NanoVDB build, using Fp16 instead");
    handle = nanovdb::tools::createNanoGrid<nanovdb::tools::build::Grid<float>,
        nanovdb::Fp16>(*buildGrid,
        nanovdb::tools::StatsMode::All,
        nanovdb::CheckMode::Full,
        0,
        enableDithering);
    break;
  }

  if (!handle) {
    tsd::core::logError(
        "[export_StructuredRegularVolumeToVDB] Failed to create NanoVDB grid.");
    return;
  }

  // Calculate quantization compression (after quantization)
  size_t finalSize = handle.size();
  float quantizationCompression = uncompressedActiveSize > 0
      ? (1.0f - static_cast<float>(finalSize) / uncompressedActiveSize)
      : 0.0f;
  float totalCompression = uncompressedSize > 0
      ? (1.0f - static_cast<float>(finalSize) / uncompressedSize)
      : 0.0f;

  tsd::core::logStatus(
      "[export_StructuredRegularVolumeToVDB] Quantization compression: %.1f%% (%.2f MB -> %.2f MB)",
      quantizationCompression * 100.0f,
      uncompressedActiveSize / (1024.0f * 1024.0f),
      finalSize / (1024.0f * 1024.0f));
  tsd::core::logStatus(
      "[export_StructuredRegularVolumeToVDB] Total compression: %.1f%% (%.2f MB -> %.2f MB)",
      totalCompression * 100.0f,
      uncompressedSize / (1024.0f * 1024.0f),
      finalSize / (1024.0f * 1024.0f));

  // Write to file
  try {
    nanovdb::io::writeGrid(
        std::string(outputFilename).c_str(), handle, nanovdb::io::Codec::NONE);
    tsd::core::logStatus(
        "[export_StructuredRegularVolumeToVDB] Successfully wrote VDB file: %s",
        std::string(outputFilename).c_str());
  } catch (const std::exception &e) {
    tsd::core::logError(
        "[export_StructuredRegularVolumeToVDB] Failed to write VDB file: %s",
        e.what());
  }
}

void export_StructuredRegularVolumeToVDB(const SpatialField *spatialField,
    std::string_view outputFilename,
    bool useUndefinedValue,
    float undefinedValue,
    VDBPrecision precision,
    bool enableDithering)
{
  if (spatialField->subtype() != tokens::volume::structuredRegular) {
    tsd::core::logError(
        "[export_StructuredRegularVolumeToVDB] Not a structuredRegularVolume.");
    return;
  }

  tsd::core::logStatus("Exporting StructuredRegularVolume to VDB file: %s",
      std::string(outputFilename).c_str());

  // Get volume data array object
  const auto *volumeData = spatialField->parameterValueAsObject<Array>("data");
  if (!volumeData) {
    tsd::core::logError(
        "[export_StructuredRegularVolumeToVDB] No volume data found.");
    return;
  }

  const auto dims =
      math::int3(volumeData->dim(0), volumeData->dim(1), volumeData->dim(2));

  const auto origin =
      spatialField->parameterValueAs<math::float3>("origin").value_or(
          math::float3(0.0f));
  const auto spacing =
      spatialField->parameterValueAs<math::float3>("spacing").value_or(
          math::float3(1.0f));

  switch (volumeData->elementType()) {
  case ANARI_UFIXED8:
    doExportStructuredRegularVolumeToNanoVDB(
        reinterpret_cast<const uint8_t *>(volumeData->data()),
        origin,
        spacing,
        dims,
        outputFilename,
        useUndefinedValue,
        undefinedValue,
        precision,
        enableDithering);
    break;
  case ANARI_FIXED8:
    doExportStructuredRegularVolumeToNanoVDB(
        reinterpret_cast<const int8_t *>(volumeData->data()),
        origin,
        spacing,
        dims,
        outputFilename,
        useUndefinedValue,
        undefinedValue,
        precision,
        enableDithering);
    break;
  case ANARI_UFIXED16:
    doExportStructuredRegularVolumeToNanoVDB(
        reinterpret_cast<const uint16_t *>(volumeData->data()),
        origin,
        spacing,
        dims,
        outputFilename,
        useUndefinedValue,
        undefinedValue,
        precision,
        enableDithering);
    break;
  case ANARI_FIXED16:
    doExportStructuredRegularVolumeToNanoVDB(
        reinterpret_cast<const int16_t *>(volumeData->data()),
        origin,
        spacing,
        dims,
        outputFilename,
        useUndefinedValue,
        undefinedValue,
        precision,
        enableDithering);
    break;
  case ANARI_FLOAT32:
    doExportStructuredRegularVolumeToNanoVDB(
        reinterpret_cast<const float *>(volumeData->data()),
        origin,
        spacing,
        dims,
        outputFilename,
        useUndefinedValue,
        undefinedValue,
        precision,
        enableDithering);
    break;
  case ANARI_FLOAT64:
    doExportStructuredRegularVolumeToNanoVDB(
        reinterpret_cast<const double *>(volumeData->data()),
        origin,
        spacing,
        dims,
        outputFilename,
        useUndefinedValue,
        undefinedValue,
        precision,
        enableDithering);
    break;
  default:
    tsd::core::logError(
        "[export_StructuredRegularVolumeToVDB] Volume data is not of a supported float type.");
    return;
  }
}

} // namespace tsd::io
