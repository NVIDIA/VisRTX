// Copyright 2025-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/animation/AnimationManager.hpp"
#include "tsd/core/DataTree.hpp"
#include "tsd/core/Logging.hpp"
#include "tsd/io/animation/EnSightFileBinding.hpp"
#include "tsd/io/importers.hpp"
#include "tsd/io/importers/detail/ensight_io.hpp"
#include "tsd/io/importers/detail/importer_common.hpp"
#include "tsd/scene/algorithms/computeScalarRange.hpp"
// std
#include <algorithm>
#include <map>
#include <string>
#include <vector>

namespace tsd::io {

using namespace tsd::core;
using namespace tsd::io::ensight;

using PerPartMaterialMap = core::FlatMap<std::string, MaterialRef>;

// Filter out triangles that lie on the wrong side of the plane.
// Plane: nx*x + ny*y + nz*z + d = 0.
// Keeps triangles whose centroid satisfies nx*cx + ny*cy + nz*cz + d >= 0.
static size_t applyCutPlaneToPart(
    Part &part, float nx, float ny, float nz, float d)
{
  const std::vector<float> &x = part.x;
  const std::vector<float> &y = part.y;
  const std::vector<float> &z = part.z;
  std::vector<uint32_t> &indices = part.triIndices;

  if (x.size() != y.size() || x.size() != z.size() || indices.size() % 3 != 0)
    return indices.size() / 3;

  std::vector<uint32_t> kept;
  kept.reserve(indices.size());

  for (size_t t = 0; t < indices.size(); t += 3) {
    uint32_t i0 = indices[t], i1 = indices[t + 1], i2 = indices[t + 2];
    if (i0 >= x.size() || i1 >= x.size() || i2 >= x.size())
      continue;

    constexpr float ONE_THIRD = 1.f / 3.f;
    float cx = (x[i0] + x[i1] + x[i2]) * ONE_THIRD;
    float cy = (y[i0] + y[i1] + y[i2]) * ONE_THIRD;
    float cz = (z[i0] + z[i1] + z[i2]) * ONE_THIRD;

    float signedDist = nx * cx + ny * cy + nz * cz + d;
    if (signedDist >= 0.f) {
      kept.push_back(i0);
      kept.push_back(i1);
      kept.push_back(i2);
    }
  }

  indices = std::move(kept);
  return indices.size() / 3;
}

static void import_ENSIGHT_impl(Scene &scene,
    tsd::animation::AnimationManager &animMgr,
    const char *filepath,
    LayerNodeRef location,
    const std::vector<std::string> *fields,
    const core::DataNode *settings,
    int timestep,
    MaterialRef overrideMaterial = {},
    const PerPartMaterialMap *perPartMaterials = nullptr)
{
  if (!location)
    location = scene.defaultLayer()->root();

  const std::string caseFile(filepath);
  const std::string caseDir = pathOf(caseFile);

  CaseData caseData;
  if (!parseCase(caseFile, caseData))
    return;

  if (timestep < 0 || timestep >= caseData.numSteps) {
    logError("[import_ENSIGHT] timestep %d out of range [0, %d)",
        timestep,
        caseData.numSteps);
    return;
  }

  logStatus("[import_ENSIGHT] loading '%s' (timestep %d/%d)",
      caseFile.c_str(),
      timestep,
      caseData.numSteps);

  const auto geoPatterns = expandPattern(caseData.geoPattern,
      caseData.startNumber,
      caseData.increment,
      caseData.numSteps);
  const std::string geoFile = caseDir
      + (timestep < (int)geoPatterns.size() ? geoPatterns[timestep]
                                            : geoPatterns.front());

  std::vector<Part> parts;
  if (!readGeoFile(geoFile, parts)) {
    logError(
        "[import_ENSIGHT] failed to read geometry file '%s'", geoFile.c_str());
    return;
  }

  logStatus("[import_ENSIGHT] read %zu part(s) from %s",
      parts.size(),
      geoFile.c_str());

  // Apply cut plane to filter triangles if settings provide one
  if (settings) {
    const auto *cpNode = settings->child("cutPlane");
    const auto *cpTargets = settings->child("cutPlaneTarget");
    if (cpNode) {
      auto cp = cpNode->getValueAs<tsd::math::float4>();
      logStatus("[import_ENSIGHT] applying cut plane (%f, %f, %f, %f)",
          cp.x,
          cp.y,
          cp.z,
          cp.w);
      for (auto &part : parts) {
        auto sanitized = sanitizePrimName(part.description, part.id);
        if (cpTargets && !cpTargets->child(sanitized))
          continue;

        size_t before = part.triIndices.size() / 3;
        size_t after = applyCutPlaneToPart(part, cp.x, cp.y, cp.z, cp.w);
        if (after < before) {
          logStatus(
              "[import_ENSIGHT] cut plane: part '%s' %zu -> %zu triangles",
              part.description.c_str(),
              before,
              after);
        }
      }
    }
  }

  // Read per-node variable data for all parts (first timestep only)
  struct VarData
  {
    core::FlatMap<int, std::vector<float>> perPart;
    int numComponents{1};
  };
  core::FlatMap<std::string, VarData> varData;

  // Build the ordered list of variable names to load.
  // If the caller specified a field list, use that order exactly; otherwise
  // use all node-centered scalar/vector variables (up to the 4-slot ANARI
  // limit).
  std::vector<std::string> varOrder;
  if (fields != nullptr) {
    for (const auto &f : *fields)
      if (caseData.variables.contains(f))
        varOrder.push_back(f);
      else
        logWarning(
            "[import_ENSIGHT] requested field '%s' not found in case "
            "file, skipping",
            f.c_str());
  } else {
    for (const auto &[name, info] : caseData.variables) {
      if (info.association != "vertex")
        continue;
      if (info.type != "scalar" && info.type != "vector")
        continue;
      varOrder.push_back(name);
      if (varOrder.size() == 4)
        break;
    }
  }

  for (const auto &name : varOrder) {
    const auto &info = caseData.variables.at(name);
    if (info->association != "vertex")
      continue;
    if (info->type != "scalar" && info->type != "vector")
      continue;

    int nc = (info->type == "vector") ? 3 : 1;
    const auto expandedPatterns = expandPattern(info->filenamePattern,
        caseData.startNumber,
        caseData.increment,
        caseData.numSteps);
    const std::string varFile = caseDir
        + (timestep < (int)expandedPatterns.size() ? expandedPatterns[timestep]
                                                   : expandedPatterns.front());

    VarData vd;
    vd.numComponents = nc;
    readVarFile(varFile, parts, nc, vd.perPart);
    varData[name] = std::move(vd);
  }

  auto root = scene.insertChildTransformNode(
      location, tsd::math::IDENTITY_MAT4, fileOf(geoFile).c_str());

  // Track geometry refs for animation binding
  std::vector<EnSightFileBinding::PartBinding> partBindings;

  for (const auto &part : parts) {
    const int numNodes = (int)part.x.size();
    const int numTris = (int)part.triIndices.size() / 3;
    if (numTris == 0)
      continue;

    const std::string partName = part.description.empty()
        ? ("part_" + std::to_string(part.id))
        : part.description;

    auto geom = scene.createObject<Geometry>(tokens::geometry::triangle);
    geom->setName(partName.c_str());

    // Vertex positions
    auto posArr = scene.createArray(ANARI_FLOAT32_VEC3, numNodes);
    auto *pos = posArr->mapAs<float3>();
    for (int i = 0; i < numNodes; ++i)
      pos[i] = float3(part.x[i], part.y[i], part.z[i]);
    posArr->unmap();
    geom->setParameterObject("vertex.position", *posArr);

    // Triangle indices
    auto idxArr = scene.createArray(ANARI_UINT32_VEC3, numTris);
    auto *idx = idxArr->mapAs<uint3>();
    const uint32_t *src = part.triIndices.data();
    for (int t = 0; t < numTris; ++t)
      idx[t] = uint3(src[t * 3], src[t * 3 + 1], src[t * 3 + 2]);
    idxArr->unmap();
    geom->setParameterObject("primitive.index", *idxArr);

    // Attach per-node variables as vertex attributes in the requested order.
    // ANARI supports attribute0..attribute3; warn if more than 4 are requested.
    int attrSlot = 0;
    ArrayRef firstScalarArr;
    for (const auto &varName : varOrder) {
      if (attrSlot > 3) {
        logWarning(
            "[import_ENSIGHT] more than 4 fields requested; '%s' and "
            "subsequent fields exceed the ANARI attribute limit and will "
            "not be loaded",
            varName.c_str());
        break;
      }
      if (!varData.contains(varName))
        continue;
      auto &vd = varData[varName];
      if (!vd.perPart.contains(part.id))
        continue;
      const auto &data = vd.perPart[part.id];
      const std::string param = "vertex.attribute" + std::to_string(attrSlot);

      if (vd.numComponents == 1) {
        if ((int)data.size() != numNodes)
          continue;
        auto arr = scene.createArray(ANARI_FLOAT32, numNodes);
        std::copy(data.begin(), data.end(), arr->mapAs<float>());
        arr->unmap();
        geom->setParameterObject(param.c_str(), *arr);
        if (!firstScalarArr)
          firstScalarArr = arr;
      } else { // vector (3 components) -> vec3 attribute
        if ((int)data.size() != numNodes * vd.numComponents)
          continue;
        auto arr = scene.createArray(ANARI_FLOAT32_VEC3, numNodes);
        auto *vec = arr->mapAs<float3>();
        for (int i = 0; i < numNodes; ++i) {
          vec[i] = float3(data[i * 3], data[i * 3 + 1], data[i * 3 + 2]);
        }
        arr->unmap();
        geom->setParameterObject(param.c_str(), *arr);
      }
      ++attrSlot;
    }

    // Per-part override > scope-level override > scalar colormap > default
    MaterialRef mat;
    if (perPartMaterials) {
      auto key = sanitizePrimName(part.description, part.id);
      if (auto *p = perPartMaterials->at(key); p && *p) {
        mat = *p;
        logStatus("[import_ENSIGHT] part '%s': per-part material override '%s'",
            partName.c_str(),
            mat->name().c_str());
      }
    }
    if (!mat && overrideMaterial)
      mat = overrideMaterial;
    if (!mat && firstScalarArr) {
      mat = scene.createObject<Material>(tokens::material::physicallyBased);
      auto range = computeScalarRange(*firstScalarArr);
      mat->setParameterObject(
          "baseColor", *makeDefaultColorMapSampler(scene, range));
    }
    if (!mat)
      mat = scene.defaultMaterial();

    auto surface = scene.createSurface(partName.c_str(), geom, mat);
    auto nodeRef = scene.insertChildObjectNode(root, surface, partName.c_str());

    partBindings.push_back({part.id, geom.data()});
  }

  // Wire animation binding if this dataset has multiple timesteps
  if (caseData.numSteps > 1 && !partBindings.empty()) {
    // Expand geo file paths for all timesteps. Always store at least one
    // geo file — readVarFile needs part metadata even for static geometry.
    std::vector<std::string> allGeoFiles;
    allGeoFiles.reserve(geoPatterns.size());
    for (const auto &p : geoPatterns)
      allGeoFiles.push_back(caseDir + p);

    // Build field mappings with expanded file paths for all timesteps
    std::vector<EnSightFileBinding::FieldMapping> fmBindings;
    int attrSlot = 0;
    for (const auto &varName : varOrder) {
      if (attrSlot > 3)
        break;
      if (!caseData.variables.contains(varName))
        continue;
      const auto &info = caseData.variables[varName];
      if (info.association != "vertex")
        continue;
      if (info.type != "scalar" && info.type != "vector")
        continue;

      const auto varPatterns = expandPattern(info.filenamePattern,
          caseData.startNumber,
          caseData.increment,
          caseData.numSteps);

      EnSightFileBinding::FieldMapping fm;
      fm.attributeName =
          Token(("vertex.attribute" + std::to_string(attrSlot)).c_str());
      fm.ensightVarName = varName;
      fm.type = info.type;
      fm.files.reserve(varPatterns.size());
      for (const auto &p : varPatterns)
        fm.files.push_back(caseDir + p);
      fmBindings.push_back(std::move(fm));

      ++attrSlot;
    }

    const size_t numFields = fmBindings.size();
    auto &anim = animMgr.addAnimation(fileOf(geoFile).c_str());
    anim.emplaceFileBinding<EnSightFileBinding>(&scene,
        std::move(partBindings),
        std::move(allGeoFiles),
        std::move(fmBindings));

    logStatus("[import_ENSIGHT] created animation '%s' (%d frames, %zu fields)",
        fileOf(geoFile).c_str(),
        caseData.numSteps,
        numFields);
  }

  logStatus("[import_ENSIGHT] done, %zu part(s) loaded", parts.size());
}

void import_ENSIGHT(Scene &scene,
    tsd::animation::AnimationManager &animMgr,
    const char *filepath,
    LayerNodeRef location,
    int timestep)
{
  import_ENSIGHT_impl(
      scene, animMgr, filepath, location, nullptr, nullptr, timestep);
}

void import_ENSIGHT(Scene &scene,
    tsd::animation::AnimationManager &animMgr,
    const char *filepath,
    LayerNodeRef location,
    const std::vector<std::string> &fields,
    int timestep)
{
  import_ENSIGHT_impl(
      scene, animMgr, filepath, location, &fields, nullptr, timestep);
}

void import_ENSIGHT(Scene &scene,
    tsd::animation::AnimationManager &animMgr,
    const char *filepath,
    LayerNodeRef location,
    const std::vector<std::string> &fields,
    const core::DataNode &settings,
    int timestep)
{
  import_ENSIGHT_impl(
      scene, animMgr, filepath, location, &fields, &settings, timestep);
}

void import_ENSIGHT(Scene &scene,
    tsd::animation::AnimationManager &animMgr,
    const char *filepath,
    LayerNodeRef location,
    const std::vector<std::string> &fields,
    const core::DataNode &settings,
    MaterialRef overrideMaterial,
    int timestep)
{
  import_ENSIGHT_impl(scene,
      animMgr,
      filepath,
      location,
      &fields,
      &settings,
      timestep,
      overrideMaterial);
}

void import_ENSIGHT(Scene &scene,
    tsd::animation::AnimationManager &animMgr,
    const char *filepath,
    LayerNodeRef location,
    const std::vector<std::string> &fields,
    const core::DataNode &settings,
    MaterialRef overrideMaterial,
    const core::FlatMap<std::string, MaterialRef> &perPartMaterials,
    int timestep)
{
  import_ENSIGHT_impl(scene,
      animMgr,
      filepath,
      location,
      &fields,
      &settings,
      timestep,
      overrideMaterial,
      &perPartMaterials);
}

} // namespace tsd::io
