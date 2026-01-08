// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "NanoVdbSidecar.hpp"
#include "tsd/core/DataTree.hpp"

// std
#include <cmath>

namespace tsd::io {

namespace {

bool isFinite(const math::float3 &v)
{
  return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

bool isFinite(const math::box3 &b)
{
  return isFinite(b.lower) && isFinite(b.upper);
}

std::optional<math::float3> parseFloat3(const core::DataNode *node)
{
  if (!node)
    return std::nullopt;

  math::float3 result;
  if (!node->getValue(ANARI_FLOAT32_VEC3, &result))
    return std::nullopt;

  return result;
}

std::optional<math::box3> parseBox3(const core::DataNode *node)
{
  if (!node)
    return std::nullopt;

  const auto *lower = node->child("lower");
  const auto *upper = node->child("upper");

  const auto lowerVec = parseFloat3(lower);
  const auto upperVec = parseFloat3(upper);

  if (!lowerVec || !upperVec)
    return std::nullopt;

  return math::box3(*lowerVec, *upperVec);
}

bool parseCoords(const core::DataNode *node, std::vector<double> &out)
{
  out.clear();
  if (!node)
    return true;

  const double *dataPtr = nullptr;
  size_t numElements = 0;
  node->getValueAsArray(&dataPtr, &numElements);

  if (dataPtr && numElements > 0) {
    out.assign(dataPtr, dataPtr + numElements);
    return true;
  }

  return true;
}

} // namespace

std::filesystem::path makeSidecarPath(const std::filesystem::path &nvdbPath)
{
  auto path = nvdbPath;
  path.replace_extension("tsd");
  return path;
}

bool writeSidecar(const NanoVdbSidecar &sidecar,
    const std::filesystem::path &path,
    std::string &errorMessage)
{
  core::DataTree tree;
  auto &root = tree.root();

  root["schemaVersion"] = static_cast<int>(sidecar.schemaVersion);
  root["volumeType"] = sidecar.volumeType;

  if (!sidecar.dataCentering.empty())
    root["dataCentering"] = sidecar.dataCentering;

  if (sidecar.origin)
    root["origin"] = *sidecar.origin;

  if (sidecar.spacing)
    root["spacing"] = *sidecar.spacing;

  if (sidecar.roi && isFinite(*sidecar.roi)) {
    auto &roiNode = root["roi"];
    roiNode["lower"] = sidecar.roi->lower;
    roiNode["upper"] = sidecar.roi->upper;
  }

  if (!sidecar.coordsX.empty()) {
    root["coordsX"].setValueAsArray(
        sidecar.coordsX.data(), sidecar.coordsX.size());
  }

  if (!sidecar.coordsY.empty()) {
    root["coordsY"].setValueAsArray(
        sidecar.coordsY.data(), sidecar.coordsY.size());
  }

  if (!sidecar.coordsZ.empty()) {
    root["coordsZ"].setValueAsArray(
        sidecar.coordsZ.data(), sidecar.coordsZ.size());
  }

  if (!tree.save(path.string().c_str())) {
    errorMessage = "failed to save sidecar file";
    return false;
  }

  return true;
}

std::optional<NanoVdbSidecar> readSidecar(
    const std::filesystem::path &path, std::string &errorMessage)
{
  core::DataTree tree;
  if (!tree.load(path.string().c_str())) {
    errorMessage = "failed to load sidecar file";
    return std::nullopt;
  }

  auto &root = tree.root();
  NanoVdbSidecar sidecar;

  // Read schema version
  const auto *schemaVersion = root.child("schemaVersion");
  if (!schemaVersion) {
    errorMessage = "sidecar missing schemaVersion";
    return std::nullopt;
  }
  int version = 0;
  if (!schemaVersion->getValue(ANARI_INT32, &version)) {
    errorMessage = "sidecar schemaVersion must be numeric";
    return std::nullopt;
  }
  sidecar.schemaVersion = static_cast<uint32_t>(version);

  // Read volume type
  const auto *volumeType = root.child("volumeType");
  if (!volumeType) {
    errorMessage = "sidecar missing volumeType";
    return std::nullopt;
  }
  sidecar.volumeType = volumeType->getValue().is(ANARI_STRING)
      ? volumeType->getValueAs<std::string>()
      : "structuredRegular";

  // Read data centering (optional)
  if (const auto *dc = root.child("dataCentering")) {
    sidecar.dataCentering = dc->getValueAs<std::string>();
  }

  // Read origin (optional)
  sidecar.origin = parseFloat3(root.child("origin"));

  // Read spacing (optional)
  sidecar.spacing = parseFloat3(root.child("spacing"));

  // Read ROI (optional)
  if (const auto *roi = root.child("roi")) {
    sidecar.roi = parseBox3(roi);
    if (!sidecar.roi) {
      errorMessage = "sidecar roi must contain lower/upper float3 vectors";
      return std::nullopt;
    }
  }

  // Read coordinate arrays (optional)
  if (!parseCoords(root.child("coordsX"), sidecar.coordsX)
      || !parseCoords(root.child("coordsY"), sidecar.coordsY)
      || !parseCoords(root.child("coordsZ"), sidecar.coordsZ)) {
    errorMessage = "sidecar coords arrays must be numeric";
    return std::nullopt;
  }

  return sidecar;
}

} // namespace tsd::io
