// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/io/importers.hpp"
#include "tsd/io/importers/detail/importer_common.hpp"
#include "tsd/core/Logging.hpp"
#include "tsd/core/ColorMapUtil.hpp"

#if TSD_USE_SILO
#include <silo.h>
#include <filesystem>
#endif

namespace tsd::io {

using namespace tsd::core;

#if TSD_USE_SILO

// Helper: map Silo data types to ANARI types
static anari::DataType siloTypeToANARIType(int siloType)
{
  switch (siloType) {
  case DB_FLOAT:
    return ANARI_FLOAT32;
  case DB_DOUBLE:
    return ANARI_FLOAT64;
  case DB_INT:
    return ANARI_INT32;
  case DB_SHORT:
    return ANARI_INT16;
  case DB_LONG:
    return ANARI_INT64;
  case DB_CHAR:
    return ANARI_INT8;
  default:
    logWarning("[import_SILO] unknown Silo data type %d, using float", siloType);
    return ANARI_FLOAT32;
  }
}

// Helper: create a SpatialField from a Silo quadmesh + quadvar
static SpatialFieldRef createFieldFromQuadMesh(Scene &scene,
    DBfile *dbfile,
    const char *meshName,
    const char *varName,
    const std::string &fieldName)
{
  // Read the mesh
  DBquadmesh *mesh = DBGetQuadmesh(dbfile, meshName);
  if (!mesh) {
    logError("[import_SILO] failed to read quad mesh '%s'", meshName);
    return {};
  }

  // Read the variable
  DBquadvar *var = DBGetQuadvar(dbfile, varName);
  if (!var) {
    logError("[import_SILO] failed to read quad var '%s'", varName);
    DBFreeQuadmesh(mesh);
    return {};
  }

  // Create structured regular spatial field
  auto field =
      scene.createObject<SpatialField>(tokens::spatial_field::structuredRegular);
  field->setName(fieldName.c_str());

  // Get dimensions (convert node counts to cell counts)
  int3 dims(mesh->dims[0] - 1, mesh->dims[1] - 1, mesh->dims[2] - 1);
  if (dims.x < 1)
    dims.x = 1;
  if (dims.y < 1)
    dims.y = 1;
  if (dims.z < 1)
    dims.z = 1;

  // Get origin and spacing from coordinates
  float3 origin(0.f);
  float3 spacing(1.f);

  if (mesh->coords[0]) {
    if (mesh->datatype == DB_FLOAT) {
      float *x = (float *)mesh->coords[0];
      origin.x = x[0];
      if (mesh->dims[0] > 1)
        spacing.x = x[1] - x[0];
    } else if (mesh->datatype == DB_DOUBLE) {
      double *x = (double *)mesh->coords[0];
      origin.x = x[0];
      if (mesh->dims[0] > 1)
        spacing.x = x[1] - x[0];
    }
  }

  if (mesh->coords[1]) {
    if (mesh->datatype == DB_FLOAT) {
      float *y = (float *)mesh->coords[1];
      origin.y = y[0];
      if (mesh->dims[1] > 1)
        spacing.y = y[1] - y[0];
    } else if (mesh->datatype == DB_DOUBLE) {
      double *y = (double *)mesh->coords[1];
      origin.y = y[0];
      if (mesh->dims[1] > 1)
        spacing.y = y[1] - y[0];
    }
  }

  if (mesh->ndims > 2 && mesh->coords[2]) {
    if (mesh->datatype == DB_FLOAT) {
      float *z = (float *)mesh->coords[2];
      origin.z = z[0];
      if (mesh->dims[2] > 1)
        spacing.z = z[1] - z[0];
    } else if (mesh->datatype == DB_DOUBLE) {
      double *z = (double *)mesh->coords[2];
      origin.z = z[0];
      if (mesh->dims[2] > 1)
        spacing.z = z[1] - z[0];
    }
  }

  field->setParameter("origin", origin);
  field->setParameter("spacing", spacing);

  // Copy variable data
  size_t numElements = dims.x * dims.y * dims.z;
  auto dataType = siloTypeToANARIType(var->datatype);
  auto dataArray = scene.createArray(dataType, dims.x, dims.y, dims.z);

  void *dst = dataArray->map();
  void *src = var->vals[0];
  std::memcpy(dst, src, numElements * anari::sizeOf(dataType));
  dataArray->unmap();

  field->setParameterObject("data", *dataArray);

  DBFreeQuadmesh(mesh);
  DBFreeQuadvar(var);

  return field;
}

// Helper: create a SpatialField from a Silo ucdmesh + ucdvar
static SpatialFieldRef createFieldFromUcdMesh(Scene &scene,
    DBfile *dbfile,
    const char *meshName,
    const char *varName,
    const std::string &fieldName)
{
  // Read the mesh
  DBucdmesh *mesh = DBGetUcdmesh(dbfile, meshName);
  if (!mesh) {
    logError("[import_SILO] failed to read UCD mesh '%s'", meshName);
    return {};
  }

  // Read the variable
  DBucdvar *var = DBGetUcdvar(dbfile, varName);
  if (!var) {
    logError("[import_SILO] failed to read UCD var '%s'", varName);
    DBFreeUcdmesh(mesh);
    return {};
  }

  // Create unstructured spatial field
  auto field =
      scene.createObject<SpatialField>(tokens::spatial_field::unstructured);
  field->setName(fieldName.c_str());

  // Copy vertex positions
  size_t numVerts = mesh->nnodes;
  auto vertexPositions = scene.createArray(ANARI_FLOAT32_VEC3, numVerts);
  float3 *positions = (float3 *)vertexPositions->map();

  for (size_t i = 0; i < numVerts; i++) {
    float x = 0.f, y = 0.f, z = 0.f;
    if (mesh->datatype == DB_FLOAT) {
      x = ((float *)mesh->coords[0])[i];
      y = ((float *)mesh->coords[1])[i];
      if (mesh->ndims > 2)
        z = ((float *)mesh->coords[2])[i];
    } else if (mesh->datatype == DB_DOUBLE) {
      x = ((double *)mesh->coords[0])[i];
      y = ((double *)mesh->coords[1])[i];
      if (mesh->ndims > 2)
        z = ((double *)mesh->coords[2])[i];
    }
    positions[i] = float3(x, y, z);
  }
  vertexPositions->unmap();
  field->setParameterObject("vertex.position", *vertexPositions);

  // Copy cell data (indices and types)
  // Silo uses zone lists, we need to convert to proper cell arrays
  if (mesh->zones && mesh->zones->nzones > 0) {
    DBzonelist *zl = mesh->zones;
    size_t numCells = zl->nzones;

    // For now, just pass through the nodelist and shape arrays
    // This is simplified - proper handling would convert to ANARI cell types
    auto cellIndex = scene.createArray(ANARI_UINT32, numCells);
    uint32_t *indices = (uint32_t *)cellIndex->map();
    for (size_t i = 0; i < numCells; i++) {
      indices[i] = i * 8; // Assuming hex cells for now
    }
    cellIndex->unmap();
    field->setParameterObject("cell.index", *cellIndex);

    // Cell type (assume hexahedron = 12 in ANARI)
    auto cellType = scene.createArray(ANARI_UINT8, numCells);
    uint8_t *types = (uint8_t *)cellType->map();
    for (size_t i = 0; i < numCells; i++) {
      types[i] = 12; // ANARI_HEXAHEDRON
    }
    cellType->unmap();
    field->setParameterObject("cell.type", *cellType);
  }

  // Copy variable data
  auto dataType = siloTypeToANARIType(var->datatype);
  auto dataArray = scene.createArray(dataType, var->nvals);

  void *dst = dataArray->map();
  void *src = var->vals[0];
  std::memcpy(dst, src, var->nvals * anari::sizeOf(dataType));
  dataArray->unmap();

  field->setParameterObject("vertex.data", *dataArray);

  DBFreeUcdmesh(mesh);
  DBFreeUcdvar(var);

  return field;
}

// Main import function
void import_SILO(Scene &scene, const char *filepath, LayerNodeRef location)
{
  if (!location)
    location = scene.defaultLayer()->root();

  // Parse variable name if present (format: file.silo or file.silo:varname)
  std::string file(filepath);
  std::string varName;
  size_t colonPos = file.find(':');
  if (colonPos != std::string::npos) {
    varName = file.substr(colonPos + 1);
    file = file.substr(0, colonPos);
  }

  // Open Silo file
  DBfile *dbfile = DBOpen(file.c_str(), DB_UNKNOWN, DB_READ);
  if (!dbfile) {
    logError("[import_SILO] failed to open file '%s'", file.c_str());
    return;
  }

  // Get table of contents
  DBtoc *toc = DBGetToc(dbfile);
  if (!toc) {
    logError("[import_SILO] failed to get table of contents");
    DBClose(dbfile);
    return;
  }

  logStatus("[import_SILO] loading '%s'", file.c_str());

  // Check for multimesh (collection of meshes)
  if (toc->nmultimesh > 0) {
    logStatus("[import_SILO] found %d multimeshes", toc->nmultimesh);

    // Find the multivar to use
    const char *multivarName = nullptr;
    if (!varName.empty() && toc->nmultivar > 0) {
      // Search for the requested multivar
      for (int i = 0; i < toc->nmultivar; i++) {
        if (varName == toc->multivar_names[i]) {
          multivarName = toc->multivar_names[i];
          break;
        }
      }
    } else if (toc->nmultivar > 0) {
      // Use first available multivar
      multivarName = toc->multivar_names[0];
    }

    if (!multivarName) {
      logError("[import_SILO] no multivar found (requested: %s)",
          varName.c_str());
      DBClose(dbfile);
      return;
    }

    DBmultivar *mv = DBGetMultivar(dbfile, multivarName);
    if (!mv) {
      logError("[import_SILO] failed to read multivar '%s'", multivarName);
      DBClose(dbfile);
      return;
    }

    // Create one transform node for all volumes
    auto tx = scene.insertChildTransformNode(location);

    // Create shared color map
    auto colorArray = scene.createArray(ANARI_FLOAT32_VEC4, 256);
    colorArray->setData(makeDefaultColorMap(colorArray->size()).data());

    // Collect all spatial fields and compute global value range
    std::vector<SpatialFieldRef> fields;
    float2 globalValueRange(std::numeric_limits<float>::max(),
        std::numeric_limits<float>::lowest());

    // Process each block in the multivar
    std::filesystem::path basePath = std::filesystem::path(file).parent_path();

    for (int i = 0; i < mv->nvars; i++) {
      const char *varPath = mv->varnames[i];
      if (!varPath || strlen(varPath) == 0)
        continue;

      // Parse block name (format: "../dir/file.silo:varname" or "varname")
      std::string blockPath(varPath);
      std::string blockFile = file;
      std::string blockVarName;

      size_t blockColon = blockPath.find(':');
      if (blockColon != std::string::npos) {
        std::string relPath = blockPath.substr(0, blockColon);
        blockVarName = blockPath.substr(blockColon + 1);

        // Resolve relative path
        std::filesystem::path fullPath = basePath / relPath;
        blockFile = fullPath.string();
      } else {
        blockVarName = blockPath;
      }

      // Open block file
      DBfile *blockDbfile = nullptr;
      if (blockFile == file) {
        blockDbfile = dbfile; // Same file
      } else {
        blockDbfile = DBOpen(blockFile.c_str(), DB_UNKNOWN, DB_READ);
        if (!blockDbfile) {
          logWarning("[import_SILO] failed to open block file '%s'",
              blockFile.c_str());
          continue;
        }
      }

      // Determine variable type
      int varType = DBInqVarType(blockDbfile, blockVarName.c_str());

      SpatialFieldRef field;
      if (varType == DB_QUADVAR) {
        // Find associated quad mesh
        DBquadvar *qv = DBGetQuadvar(blockDbfile, blockVarName.c_str());
        if (qv && qv->meshname) {
          field = createFieldFromQuadMesh(scene,
              blockDbfile,
              qv->meshname,
              blockVarName.c_str(),
              blockVarName);
          DBFreeQuadvar(qv);
        }
      } else if (varType == DB_UCDVAR) {
        // Find associated UCD mesh
        DBucdvar *uv = DBGetUcdvar(blockDbfile, blockVarName.c_str());
        if (uv && uv->meshname) {
          field = createFieldFromUcdMesh(scene,
              blockDbfile,
              uv->meshname,
              blockVarName.c_str(),
              blockVarName);
          DBFreeUcdvar(uv);
        }
      }

      // Close block file if it's different
      if (blockDbfile != dbfile)
        DBClose(blockDbfile);

      if (field) {
        fields.push_back(field);

        // Update global value range
        float2 localRange = field->computeValueRange();
        globalValueRange.x = std::min(globalValueRange.x, localRange.x);
        globalValueRange.y = std::max(globalValueRange.y, localRange.y);

        logStatus("[import_SILO] loaded block %d: %s (range: %f to %f)",
            i,
            blockVarName.c_str(),
            localRange.x,
            localRange.y);
      }
    }

    DBFreeMultivar(mv);

    // Create all volumes under the same transform with shared transfer function
    logStatus(
        "[import_SILO] global value range: %f to %f", globalValueRange.x, globalValueRange.y);

    for (size_t i = 0; i < fields.size(); i++) {
      auto [inst, volume] = scene.insertNewChildObjectNode<Volume>(
          tx, tokens::volume::transferFunction1D);
      volume->setName((multivarName + std::string("_block_") + std::to_string(i)).c_str());
      volume->setParameterObject("value", *fields[i]);
      volume->setParameterObject("color", *colorArray);
      volume->setParameter("valueRange", ANARI_FLOAT32_BOX1, &globalValueRange);
    }
  }
  // Single mesh case
  else {
    // Find the variable to visualize
    const char *targetVarName = nullptr;
    int targetVarType = DB_INVALID_OBJECT;

    if (!varName.empty()) {
      // Try to find the requested variable
      targetVarType = DBInqVarType(dbfile, varName.c_str());
      if (targetVarType == DB_QUADVAR || targetVarType == DB_UCDVAR) {
        targetVarName = varName.c_str();
      }
    }

    // If no variable specified or not found, use first available
    if (!targetVarName) {
      if (toc->nqvar > 0) {
        targetVarName = toc->qvar_names[0];
        targetVarType = DB_QUADVAR;
      } else if (toc->nucdvar > 0) {
        targetVarName = toc->ucdvar_names[0];
        targetVarType = DB_UCDVAR;
      }
    }

    if (!targetVarName) {
      logError("[import_SILO] no variables found in file");
      DBClose(dbfile);
      return;
    }

    SpatialFieldRef field;
    if (targetVarType == DB_QUADVAR) {
      DBquadvar *qv = DBGetQuadvar(dbfile, targetVarName);
      if (qv && qv->meshname) {
        field = createFieldFromQuadMesh(
            scene, dbfile, qv->meshname, targetVarName, targetVarName);
        DBFreeQuadvar(qv);
      }
    } else if (targetVarType == DB_UCDVAR) {
      DBucdvar *uv = DBGetUcdvar(dbfile, targetVarName);
      if (uv && uv->meshname) {
        field = createFieldFromUcdMesh(
            scene, dbfile, uv->meshname, targetVarName, targetVarName);
        DBFreeUcdvar(uv);
      }
    }

    if (field) {
      // Create transform node
      auto tx = scene.insertChildTransformNode(location);

      // Create default color map
      auto colorArray = scene.createArray(ANARI_FLOAT32_VEC4, 256);
      colorArray->setData(makeDefaultColorMap(colorArray->size()).data());

      // Compute value range
      float2 valueRange = field->computeValueRange();

      // Create volume
      auto [inst, volume] = scene.insertNewChildObjectNode<Volume>(
          tx, tokens::volume::transferFunction1D);
      volume->setName(targetVarName);
      volume->setParameterObject("value", *field);
      volume->setParameterObject("color", *colorArray);
      volume->setParameter("valueRange", ANARI_FLOAT32_BOX1, &valueRange);

      logStatus("[import_SILO] loaded variable '%s' (range: %f to %f)",
          targetVarName,
          valueRange.x,
          valueRange.y);
    }
  }

  DBClose(dbfile);
  logStatus("[import_SILO] done!");
}

#else

void import_SILO(Scene &scene, const char *filepath, LayerNodeRef location)
{
  logError("[import_SILO] Silo support not enabled in this build");
}

#endif

} // namespace tsd::io
