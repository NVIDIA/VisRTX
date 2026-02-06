// Copyright 2025-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/io/importers.hpp"

#include "tsd/core/ColorMapUtil.hpp"
#include "tsd/core/Logging.hpp"
#include "tsd/core/algorithms/computeScalarRange.hpp"
#include "tsd/io/importers/detail/importer_common.hpp"
// std
#include <algorithm>
#include <vector>
// agx
#define AGX_READ_IMPL
#include <agx/agx_read.h>

namespace tsd::io {

// Importer Animated Geometry eXchange (AGX) format
//
// See https://github.com/jeffamstutz/agx
//
void import_AGX(Scene &scene, const char *filepath, LayerNodeRef location)
{
  std::string file = fileOf(filepath);
  if (file.empty()) {
    logError("[import_AGX] no file specified from path '%s'", filepath);
    return;
  }

  AGXReader r = agxNewReader(filepath);
  if (!r) {
    logError("[import_AGX] failed to open file '%s'", filepath);
    return;
  }

  //// Parse header ////

  AGXHeader hdr{};
  if (agxReaderGetHeader(r, &hdr) != 0) {
    logError("[import_AGX] failed to read header");
    agxReleaseReader(r);
    return;
  }

  logInfo(
      "[import_AGX] file version=%u, objectType=%s, timeSteps=%u, constants=%u",
      hdr.version,
      anari::toString(hdr.objectType),
      hdr.timeSteps,
      hdr.constantParamCount);

  //// Validate usable file contents ////

  if (hdr.objectType != ANARI_GEOMETRY) {
    logError("[import_AGX] unsupported object type '%s' in file '%s'",
        anari::toString(hdr.objectType),
        filepath);
    agxReleaseReader(r);
    return;
  }

  const char *subtype = agxReaderGetSubtype(r);
  if (!subtype || subtype[0] == '\0') {
    logError("[import_AGX] no subtype specified in file '%s'", filepath);
    agxReleaseReader(r);
    return;
  }

  //// Create TSD objects ////

  auto agx_root = scene.insertChildTransformNode(
      location ? location : scene.defaultLayer()->root());
  (*agx_root)->name() = "agx_transform_" + file;

  // geometry

  auto geom = scene.createObject<tsd::core::Geometry>(subtype);
  if (!geom) {
    logError("[import_AGX] failed to create geometry of type '%s'", subtype);
    agxReleaseReader(r);
    return;
  }

  geom->setName(("agx_geometry_" + file).c_str());

  // geometry constants

  agxReaderResetConstants(r);
  AGXParamView pv{};
  int constantsRead = 0;
  while (true) {
    int rc = agxReaderNextConstant(r, &pv);
    if (rc < 0) {
      logWarning(
          "[import_AGX] error reading constant from file '%s'", filepath);
      break;
    }
    if (rc == 0)
      break;

    std::string paramName(pv.name, pv.nameLength);
    logInfo("[import_AGX] constant param: %s, isArray=%d",
        paramName.c_str(),
        pv.isArray);

    if (pv.isArray) {
      auto arr = scene.createArray(pv.elementType, pv.elementCount);
      if (!arr) {
        logError("[import_AGX] failed to create array for parameter '%s'",
            paramName.c_str());
        continue;
      }
      arr->setData(pv.data);
      geom->setParameterObject(paramName.c_str(), *arr);
    } else {
      geom->setParameter(paramName.c_str(), anari::DataType(pv.type), pv.data);
    }
    constantsRead++;
  }
  logInfo("[import_AGX] read %d constant parameters", constantsRead);

  // geometry time steps

  std::vector<Token> timeStepNames;
  std::vector<TimeStepArrays> timeSteps;

  agxReaderResetTimeSteps(r);
  uint32_t stepIndex = 0, paramCount = 0;
  bool firstStep = true;
  while (agxReaderBeginNextTimeStep(r, &stepIndex, &paramCount) == 1) {
    logInfo("[import_AGX] time step %u: %u params", stepIndex, paramCount);
    int paramIdx = 0;
    while (true) {
      int rc = agxReaderNextTimeStepParam(r, &pv);
      if (rc < 0) {
        logError("[import_AGX] error reading step params");
        break;
      }
      if (rc == 0)
        break;

      std::string tsParamName(pv.name, pv.nameLength);
      logInfo("[import_AGX] timestep param: %s, isArray=%d, elementCount=%lu",
          tsParamName.c_str(),
          pv.isArray,
          (unsigned long)pv.elementCount);

      // TODO: support non-array animations
      if (firstStep && pv.isArray) {
        timeStepNames.push_back(Token(tsParamName.c_str()));
        timeSteps.emplace_back();
      }

      if (pv.isArray) {
        if (paramIdx >= (int)timeSteps.size()) {
          logError("[import_AGX] paramIdx %d out of bounds (size %zu)",
              paramIdx,
              timeSteps.size());
          break;
        }
        auto arr = scene.createArray(pv.elementType, pv.elementCount);
        if (!arr) {
          logError(
              "[import_AGX] failed to create array for timestep parameter '%s'",
              tsParamName.c_str());
          paramIdx++;
          continue;
        }
        arr->setData(pv.data);

        // Debug: print attribute values for first timestep
        if (firstStep && tsParamName.find("attribute") != std::string::npos) {
          logInfo("[import_AGX] DEBUG: %s data pointer: %p, bytes: %lu",
              tsParamName.c_str(),
              pv.data,
              (unsigned long)pv.dataBytes);
          if (pv.elementType == ANARI_FLOAT32 && pv.elementCount > 0) {
            const float *values = (const float *)pv.data;
            float minVal = values[0], maxVal = values[0];
            for (size_t i = 1; i < std::min((size_t)pv.elementCount, (size_t)100);
                 i++) {
              if (values[i] < minVal)
                minVal = values[i];
              if (values[i] > maxVal)
                maxVal = values[i];
            }
            logInfo(
                "[import_AGX] DEBUG: %s first 10 values: [%.6f, %.6f, %.6f, "
                "%.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f]",
                tsParamName.c_str(),
                values[0],
                values[1],
                values[2],
                values[3],
                values[4],
                values[5],
                values[6],
                values[7],
                values[8],
                values[9]);
            logInfo("[import_AGX] DEBUG: %s range (first 100): [%.6f, %.6f]",
                tsParamName.c_str(),
                minVal,
                maxVal);
          }
        }

        timeSteps[paramIdx++].push_back(arr);
        if (firstStep)
          geom->setParameterObject(tsParamName.c_str(), *arr);
      } else {
        logWarning(
            "[import_AGX] ignoring non-array parameter '%s' in time step",
            tsParamName.c_str());
      }
    }
    firstStep = false;
  }

  // material

  auto mat = scene.createObject<tsd::core::Material>(
      tsd::core::tokens::material::matte);
  if (!mat) {
    logError("[import_AGX] failed to create material");
    agxReleaseReader(r);
    return;
  }
  mat->setName("agx_material");

  // Check if we have vertex attributes to map to color
  bool hasAttribute0 = false;
  for (const auto& name : timeStepNames) {
    if (name.str().find("attribute0") != std::string::npos) {
      hasAttribute0 = true;
      break;
    }
  }

  if (hasAttribute0 && !timeSteps.empty() && timeSteps[0].size() > 0) {
    // Find vertex.attribute0 in first timestep
    size_t attr0Idx = 0;
    for (size_t i = 0; i < timeStepNames.size(); i++) {
      if (timeStepNames[i].str().find("attribute0") != std::string::npos) {
        attr0Idx = i;
        break;
      }
    }

    if (attr0Idx < timeSteps.size() && !timeSteps[attr0Idx].empty()) {
      auto attr0Array = timeSteps[attr0Idx][0];
      auto scalarRange = computeScalarRange(*attr0Array);
      logInfo("[import_AGX] vertex.attribute0 range: [%f, %f]", scalarRange.x, scalarRange.y);

      // Create colormap sampler
      auto colorMapSampler = makeDefaultColorMapSampler(scene, scalarRange);
      mat->setParameterObject("color", *colorMapSampler);
      logInfo("[import_AGX] applied colormap to material for vertex.attribute0");
    } else {
      mat->setParameter("color", tsd::math::float3(0.8f, 0.8f, 0.8f));
    }
  } else {
    // No attributes, use fixed color
    mat->setParameter("color", tsd::math::float3(0.8f, 0.8f, 0.8f));
  }

  // surface

  auto surface = scene.createSurface("agx_surface", geom, mat);
  if (!surface) {
    logError("[import_AGX] failed to create surface");
    agxReleaseReader(r);
    return;
  }
  scene.insertChildObjectNode(agx_root, surface);

  // animation

  if (!timeSteps.empty()) {
    auto *anim = scene.addAnimation(file.c_str());
    if (anim) {
      anim->setAsTimeSteps(*geom, timeStepNames, timeSteps);
      logInfo("[import_AGX] animation created successfully");
    } else {
      logError("[import_AGX] failed to create animation");
    }
  }

  //// Cleanup ////

  agxReleaseReader(r);
}

} // namespace tsd::io
