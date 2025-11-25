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

// Silo ghost zone type constants (from silo.h):
// DB_GHOSTTYPE_NOGHOST (0) - real data, not a ghost/halo cell
// DB_GHOSTTYPE_INTDUP (1)  - duplicated internal ghost zone (halo cell)

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

  // Check for ghost zones using ghost_zone_labels
  // Note: min_index/max_index refer to NODE indices, not zone indices!
  int3 realNodeFirst(0, 0, 0);
  int3 realNodeLast(mesh->dims[0] - 1, mesh->dims[1] - 1, mesh->dims[2] - 1);
  bool hasGhostZones = false;

  if (mesh->ghost_zone_labels) {
    hasGhostZones = true;
    logStatus("[import_SILO] mesh '%s' has ghost zone labels", meshName);
    
    // Use min_index/max_index to determine real NODE bounds
    // These define the extent of real (non-ghost) nodes
    if (mesh->min_index && mesh->max_index) {
      for (int i = 0; i < mesh->ndims && i < 3; i++) {
        realNodeFirst[i] = mesh->min_index[i];
        realNodeLast[i] = mesh->max_index[i];
      }
      logStatus("[import_SILO] real nodes: [%d:%d, %d:%d, %d:%d]",
          realNodeFirst.x, realNodeLast.x,
          realNodeFirst.y, realNodeLast.y,
          realNodeFirst.z, realNodeLast.z);
    } else {
      // No min/max_index, need to scan ghost_zone_labels to find real extents
      // For now, log a warning
      logWarning("[import_SILO] ghost_zone_labels present but no min_index/max_index");
    }
  }

  // Get dimensions (only real zones)
  // Zones are defined between nodes, so zone count = node_count - 1
  int3 totalNodeDims(mesh->dims[0], mesh->dims[1], mesh->dims[2]);
  int3 totalDims(mesh->dims[0] - 1, mesh->dims[1] - 1, mesh->dims[2] - 1);
  int3 realZoneFirst(0, 0, 0);
  int3 realZoneLast = totalDims;
  int3 dims;
  
  if (hasGhostZones) {
    // Real zones are between real nodes
    // If real nodes go from index A to B, real zones go from A to B-1
    realZoneFirst = realNodeFirst;
    realZoneLast = int3(
        std::max(realNodeLast.x - 1, realNodeFirst.x),
        std::max(realNodeLast.y - 1, realNodeFirst.y),
        std::max(realNodeLast.z - 1, realNodeFirst.z));
    
    dims = int3(
        realZoneLast.x - realZoneFirst.x + 1,
        realZoneLast.y - realZoneFirst.y + 1,
        realZoneLast.z - realZoneFirst.z + 1);
    
    logStatus("[import_SILO] real zone range: [%d:%d, %d:%d, %d:%d] = %d x %d x %d zones",
        realZoneFirst.x, realZoneLast.x,
        realZoneFirst.y, realZoneLast.y,
        realZoneFirst.z, realZoneLast.z,
        dims.x, dims.y, dims.z);
  } else {
    dims = totalDims;
  }

  if (dims.x < 1) dims.x = 1;
  if (dims.y < 1) dims.y = 1;
  if (dims.z < 1) dims.z = 1;
  
  logStatus("[import_SILO] total node dims: %d x %d x %d, total zone dims: %d x %d x %d, final dims: %d x %d x %d",
      totalNodeDims.x, totalNodeDims.y, totalNodeDims.z,
      totalDims.x, totalDims.y, totalDims.z,
      dims.x, dims.y, dims.z);

  // Get origin and spacing from coordinates
  float3 origin(0.f);
  float3 spacing(1.f);

  if (mesh->coords[0]) {
    if (mesh->datatype == DB_FLOAT) {
      float *x = (float *)mesh->coords[0];
      // If we have ghost zones, adjust origin to start of real data (at first real node)
      int startIdx = hasGhostZones ? realNodeFirst.x : 0;
      origin.x = x[startIdx];
      if (mesh->dims[0] > 1)
        spacing.x = x[1] - x[0];
    } else if (mesh->datatype == DB_DOUBLE) {
      double *x = (double *)mesh->coords[0];
      int startIdx = hasGhostZones ? realNodeFirst.x : 0;
      origin.x = x[startIdx];
      if (mesh->dims[0] > 1)
        spacing.x = x[1] - x[0];
    }
  }

  if (mesh->coords[1]) {
    if (mesh->datatype == DB_FLOAT) {
      float *y = (float *)mesh->coords[1];
      int startIdx = hasGhostZones ? realNodeFirst.y : 0;
      origin.y = y[startIdx];
      if (mesh->dims[1] > 1)
        spacing.y = y[1] - y[0];
    } else if (mesh->datatype == DB_DOUBLE) {
      double *y = (double *)mesh->coords[1];
      int startIdx = hasGhostZones ? realNodeFirst.y : 0;
      origin.y = y[startIdx];
      if (mesh->dims[1] > 1)
        spacing.y = y[1] - y[0];
    }
  }

  if (mesh->ndims > 2 && mesh->coords[2]) {
    if (mesh->datatype == DB_FLOAT) {
      float *z = (float *)mesh->coords[2];
      int startIdx = hasGhostZones ? realNodeFirst.z : 0;
      origin.z = z[startIdx];
      if (mesh->dims[2] > 1)
        spacing.z = z[1] - z[0];
    } else if (mesh->datatype == DB_DOUBLE) {
      double *z = (double *)mesh->coords[2];
      int startIdx = hasGhostZones ? realNodeFirst.z : 0;
      origin.z = z[startIdx];
      if (mesh->dims[2] > 1)
        spacing.z = z[1] - z[0];
    }
  }

  field->setParameter("origin", origin);
  field->setParameter("spacing", spacing);

  // Copy variable data - only copy non-ghost zones
  size_t numElements = dims.x * dims.y * dims.z;
  auto dataType = siloTypeToANARIType(var->datatype);
  auto dataArray = scene.createArray(dataType, dims.x, dims.y, dims.z);

  void *dst = dataArray->map();

  if (hasGhostZones) {
    // Copy only the real zone sub-region defined by [realZoneFirst, realZoneLast]
    // For structured grids, min_index/max_index are NODE indices, zones are nodes-1
    size_t dstIdx = 0;
    size_t elementSize = anari::sizeOf(dataType);
    
    for (int k = realZoneFirst.z; k <= realZoneLast.z; k++) {
      for (int j = realZoneFirst.y; j <= realZoneLast.y; j++) {
        for (int i = realZoneFirst.x; i <= realZoneLast.x; i++) {
          int srcIdx = k * totalDims.y * totalDims.x + j * totalDims.x + i;
          std::memcpy(
              (char *)dst + dstIdx * elementSize,
              (char *)var->vals[0] + srcIdx * elementSize,
              elementSize);
          dstIdx++;
        }
      }
    }
    logStatus("[import_SILO] copied %zu real zones (excluded ghosts from %d total)",
        dstIdx, totalDims.x * totalDims.y * totalDims.z);
  } else {
    // No ghost zones, direct copy
    void *src = var->vals[0];
    std::memcpy(dst, src, numElements * anari::sizeOf(dataType));
  }

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

  // Check for ghost zones using ghost_zone_labels
  bool hasGhostZones = false;
  std::vector<int> realZoneIndices;
  
  if (mesh->zones && mesh->zones->ghost_zone_labels) {
    hasGhostZones = true;
    logStatus("[import_SILO] UCD mesh '%s' has ghost zone labels", meshName);
    
    // Build list of real (non-ghost) zones
    for (int i = 0; i < mesh->zones->nzones; i++) {
      if (mesh->zones->ghost_zone_labels[i] == DB_GHOSTTYPE_NOGHOST) {
        realZoneIndices.push_back(i);
      }
    }
    
    logStatus("[import_SILO] %zu real zones out of %d total",
        realZoneIndices.size(), mesh->zones->nzones);
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

  // Copy cell data (indices and types) - only real zones
  if (mesh->zones && mesh->zones->nzones > 0) {
    DBzonelist *zl = mesh->zones;
    size_t numCells = hasGhostZones ? realZoneIndices.size() : zl->nzones;

    // For now, just pass through the nodelist and shape arrays
    // This is simplified - proper handling would convert to ANARI cell types
    auto cellIndex = scene.createArray(ANARI_UINT32, numCells);
    uint32_t *indices = (uint32_t *)cellIndex->map();
    
    if (hasGhostZones) {
      // Only include real zones
      for (size_t i = 0; i < realZoneIndices.size(); i++) {
        indices[i] = realZoneIndices[i] * 8; // Assuming hex cells for now
      }
    } else {
      for (size_t i = 0; i < numCells; i++) {
        indices[i] = i * 8; // Assuming hex cells for now
      }
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

  // Copy variable data - handle ghost zones for zone-centered variables
  auto dataType = siloTypeToANARIType(var->datatype);
  
  if (var->centering == DB_ZONECENT && hasGhostZones) {
    // Zone-centered variable: filter out ghost zones
    size_t numRealZones = realZoneIndices.size();
    auto dataArray = scene.createArray(dataType, numRealZones);
    void *dst = dataArray->map();
    size_t elementSize = anari::sizeOf(dataType);
    
    for (size_t i = 0; i < realZoneIndices.size(); i++) {
      int srcIdx = realZoneIndices[i];
      std::memcpy(
          (char *)dst + i * elementSize,
          (char *)var->vals[0] + srcIdx * elementSize,
          elementSize);
    }
    dataArray->unmap();
    field->setParameterObject("cell.data", *dataArray);
    
    logStatus("[import_SILO] copied zone-centered data for %zu real zones",
        numRealZones);
  } else if (var->centering == DB_NODECENT) {
    // Node-centered variable: copy all node data
    // Note: could also filter ghost nodes if mesh->ghost_node_labels exists
    auto dataArray = scene.createArray(dataType, var->nvals);
    void *dst = dataArray->map();
    void *src = var->vals[0];
    std::memcpy(dst, src, var->nvals * anari::sizeOf(dataType));
    dataArray->unmap();
    field->setParameterObject("vertex.data", *dataArray);
  } else {
    // No ghost zones or other centering, copy as-is
    auto dataArray = scene.createArray(dataType, var->nvals);
    void *dst = dataArray->map();
    void *src = var->vals[0];
    std::memcpy(dst, src, var->nvals * anari::sizeOf(dataType));
    dataArray->unmap();
    field->setParameterObject("vertex.data", *dataArray);
  }

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

    // Create one transform node for the unified volume
    auto tx = scene.insertChildTransformNode(location);

    // Create shared color map
    auto colorArray = scene.createArray(ANARI_FLOAT32_VEC4, 256);
    colorArray->setData(makeDefaultColorMap(colorArray->size()).data());

    // First pass: collect block metadata to determine global grid
    struct BlockInfo {
      std::string file;
      std::string varName;
      float3 origin;
      float3 spacing;
      int3 dims;
      int varType;
    };
    
    std::vector<BlockInfo> blocks;
    float3 globalSpacing(1.f, 1.f, 1.f);
    float3 globalMin(std::numeric_limits<float>::max());
    float3 globalMax(std::numeric_limits<float>::lowest());
    float2 globalValueRange(std::numeric_limits<float>::max(),
        std::numeric_limits<float>::lowest());
    anari::DataType globalDataType = ANARI_FLOAT32;
    bool firstBlock = true;

    std::filesystem::path basePath = std::filesystem::path(file).parent_path();

    logStatus("[import_SILO] collecting %d blocks to create unified volume", mv->nvars);

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

      // Get block metadata
      BlockInfo info;
      info.file = blockFile;
      info.varName = blockVarName;
      info.varType = varType;

      if (varType == DB_QUADVAR) {
        DBquadvar *qv = DBGetQuadvar(blockDbfile, blockVarName.c_str());
        DBquadmesh *qm = nullptr;
        
        if (qv && qv->meshname) {
          qm = DBGetQuadmesh(blockDbfile, qv->meshname);
          
          if (qm) {
            // Get spacing (assume uniform)
            if (firstBlock) {
              if (qm->coords[0] && qm->dims[0] > 1) {
                if (qm->datatype == DB_FLOAT) {
                  float *x = (float *)qm->coords[0];
                  globalSpacing.x = x[1] - x[0];
                } else if (qm->datatype == DB_DOUBLE) {
                  double *x = (double *)qm->coords[0];
                  globalSpacing.x = x[1] - x[0];
                }
              }
              if (qm->coords[1] && qm->dims[1] > 1) {
                if (qm->datatype == DB_FLOAT) {
                  float *y = (float *)qm->coords[1];
                  globalSpacing.y = y[1] - y[0];
                } else if (qm->datatype == DB_DOUBLE) {
                  double *y = (double *)qm->coords[1];
                  globalSpacing.y = y[1] - y[0];
                }
              }
              if (qm->ndims > 2 && qm->coords[2] && qm->dims[2] > 1) {
                if (qm->datatype == DB_FLOAT) {
                  float *z = (float *)qm->coords[2];
                  globalSpacing.z = z[1] - z[0];
                } else if (qm->datatype == DB_DOUBLE) {
                  double *z = (double *)qm->coords[2];
                  globalSpacing.z = z[1] - z[0];
                }
              }
              globalDataType = siloTypeToANARIType(qv->datatype);
              firstBlock = false;
            }
            
            // Get real node bounds (accounting for ghost zones)
            int3 realNodeFirst(0, 0, 0);
            int3 realNodeLast(qm->dims[0] - 1, qm->dims[1] - 1, qm->dims[2] - 1);
            
            if (qm->ghost_zone_labels && qm->min_index && qm->max_index) {
              for (int j = 0; j < qm->ndims && j < 3; j++) {
                realNodeFirst[j] = qm->min_index[j];
                realNodeLast[j] = qm->max_index[j];
              }
            }
            
            // Calculate origin and extents
            float3 blockOrigin(0.f);
            if (qm->coords[0]) {
              if (qm->datatype == DB_FLOAT)
                blockOrigin.x = ((float *)qm->coords[0])[realNodeFirst.x];
              else if (qm->datatype == DB_DOUBLE)
                blockOrigin.x = ((double *)qm->coords[0])[realNodeFirst.x];
            }
            if (qm->coords[1]) {
              if (qm->datatype == DB_FLOAT)
                blockOrigin.y = ((float *)qm->coords[1])[realNodeFirst.y];
              else if (qm->datatype == DB_DOUBLE)
                blockOrigin.y = ((double *)qm->coords[1])[realNodeFirst.y];
            }
            if (qm->ndims > 2 && qm->coords[2]) {
              if (qm->datatype == DB_FLOAT)
                blockOrigin.z = ((float *)qm->coords[2])[realNodeFirst.z];
              else if (qm->datatype == DB_DOUBLE)
                blockOrigin.z = ((double *)qm->coords[2])[realNodeFirst.z];
            }
            
            info.origin = blockOrigin;
            info.spacing = globalSpacing;
            
            // Dims are in zones (nodes - 1)
            info.dims = int3(
                realNodeLast.x - realNodeFirst.x,
                realNodeLast.y - realNodeFirst.y,
                realNodeLast.z - realNodeFirst.z);
            
            // Update global bounds
            float3 blockMax = blockOrigin + float3(info.dims) * globalSpacing;
            globalMin = min(globalMin, blockOrigin);
            globalMax = max(globalMax, blockMax);
            
            DBFreeQuadmesh(qm);
          }
          DBFreeQuadvar(qv);
        }
      }

      // Close block file if it's different
      if (blockDbfile != dbfile)
        DBClose(blockDbfile);

      if (info.dims.x > 0) {
        blocks.push_back(info);
      }
    }

    // Don't free multivar yet - we need it for the second pass
    // It will be freed after the second pass completes

    if (blocks.empty()) {
      logError("[import_SILO] no valid blocks found");
      DBFreeMultivar(mv);
      DBClose(dbfile);
      return;
    }

    // Calculate global grid dimensions
    float3 globalExtent = globalMax - globalMin;
    int3 globalDims(
        (int)std::round(globalExtent.x / globalSpacing.x),
        (int)std::round(globalExtent.y / globalSpacing.y),
        (int)std::round(globalExtent.z / globalSpacing.z));
    
    if (globalDims.x < 1) globalDims.x = 1;
    if (globalDims.y < 1) globalDims.y = 1;
    if (globalDims.z < 1) globalDims.z = 1;

    logStatus("[import_SILO] unified volume: origin=(%.2f,%.2f,%.2f) spacing=(%.2f,%.2f,%.2f) dims=%dx%dx%d",
        globalMin.x, globalMin.y, globalMin.z,
        globalSpacing.x, globalSpacing.y, globalSpacing.z,
        globalDims.x, globalDims.y, globalDims.z);

    // Create unified data array
    size_t totalElements = (size_t)globalDims.x * globalDims.y * globalDims.z;
    auto unifiedDataArray = scene.createArray(globalDataType, globalDims.x, globalDims.y, globalDims.z);
    void *unifiedData = unifiedDataArray->map();
    
    // Initialize to zero/background value
    std::memset(unifiedData, 0, totalElements * anari::sizeOf(globalDataType));

    // Second pass: copy each block's data into the unified grid
    for (size_t blockIdx = 0; blockIdx < blocks.size(); blockIdx++) {
      const BlockInfo &info = blocks[blockIdx];
      
      // Open block file
      DBfile *blockDbfile = nullptr;
      if (info.file == file) {
        blockDbfile = dbfile;
      } else {
        blockDbfile = DBOpen(info.file.c_str(), DB_UNKNOWN, DB_READ);
        if (!blockDbfile) continue;
      }

      if (info.varType == DB_QUADVAR) {
        DBquadvar *qv = DBGetQuadvar(blockDbfile, info.varName.c_str());
        DBquadmesh *qm = nullptr;
        
        if (qv && qv->meshname) {
          qm = DBGetQuadmesh(blockDbfile, qv->meshname);
          
          if (qm) {
            // Calculate this block's position in the global grid
            float3 offset = info.origin - globalMin;
            int3 globalOffset(
                (int)std::round(offset.x / globalSpacing.x),
                (int)std::round(offset.y / globalSpacing.y),
                (int)std::round(offset.z / globalSpacing.z));
            
            // Get real node bounds for this block
            int3 realNodeFirst(0, 0, 0);
            int3 realNodeLast(qm->dims[0] - 1, qm->dims[1] - 1, qm->dims[2] - 1);
            
            if (qm->ghost_zone_labels && qm->min_index && qm->max_index) {
              for (int j = 0; j < qm->ndims && j < 3; j++) {
                realNodeFirst[j] = qm->min_index[j];
                realNodeLast[j] = qm->max_index[j];
              }
            }
            
            int3 realZoneFirst = realNodeFirst;
            int3 realZoneLast = int3(
                std::max(realNodeLast.x - 1, realNodeFirst.x),
                std::max(realNodeLast.y - 1, realNodeFirst.y),
                std::max(realNodeLast.z - 1, realNodeFirst.z));
            
            int3 blockTotalDims(qm->dims[0] - 1, qm->dims[1] - 1, qm->dims[2] - 1);
            size_t elementSize = anari::sizeOf(globalDataType);
            
            // Copy only real zones from this block into the global grid
            for (int k = realZoneFirst.z; k <= realZoneLast.z; k++) {
              for (int j = realZoneFirst.y; j <= realZoneLast.y; j++) {
                for (int i = realZoneFirst.x; i <= realZoneLast.x; i++) {
                  // Source index in block's local array
                  int srcIdx = k * blockTotalDims.y * blockTotalDims.x + 
                               j * blockTotalDims.x + i;
                  
                  // Destination index in global array
                  int gi = globalOffset.x + (i - realZoneFirst.x);
                  int gj = globalOffset.y + (j - realZoneFirst.y);
                  int gk = globalOffset.z + (k - realZoneFirst.z);
                  int dstIdx = gk * globalDims.y * globalDims.x + 
                               gj * globalDims.x + gi;
                  
                  std::memcpy(
                      (char *)unifiedData + dstIdx * elementSize,
                      (char *)qv->vals[0] + srcIdx * elementSize,
                      elementSize);
                }
              }
            }
            
            // Update global value range
            float2 localRange(std::numeric_limits<float>::max(),
                std::numeric_limits<float>::lowest());
            
            // Scan the block's data for min/max
            size_t blockElements = (realZoneLast.x - realZoneFirst.x + 1) *
                                   (realZoneLast.y - realZoneFirst.y + 1) *
                                   (realZoneLast.z - realZoneFirst.z + 1);
            
            for (int k = realZoneFirst.z; k <= realZoneLast.z; k++) {
              for (int j = realZoneFirst.y; j <= realZoneLast.y; j++) {
                for (int i = realZoneFirst.x; i <= realZoneLast.x; i++) {
                  int srcIdx = k * blockTotalDims.y * blockTotalDims.x + 
                               j * blockTotalDims.x + i;
                  float val = 0.f;
                  if (qv->datatype == DB_FLOAT)
                    val = ((float *)qv->vals[0])[srcIdx];
                  else if (qv->datatype == DB_DOUBLE)
                    val = ((double *)qv->vals[0])[srcIdx];
                  
                  localRange.x = std::min(localRange.x, val);
                  localRange.y = std::max(localRange.y, val);
                }
              }
            }
            
            globalValueRange.x = std::min(globalValueRange.x, localRange.x);
            globalValueRange.y = std::max(globalValueRange.y, localRange.y);
            
            logStatus("[import_SILO] block %d: %s offset=(%d,%d,%d) dims=%dx%dx%d range=(%.2f,%.2f)",
                (int)blockIdx, info.varName.c_str(),
                globalOffset.x, globalOffset.y, globalOffset.z,
                info.dims.x, info.dims.y, info.dims.z,
                localRange.x, localRange.y);
            
            DBFreeQuadmesh(qm);
          }
          DBFreeQuadvar(qv);
        }
      }

      // Close block file if it's different
      if (blockDbfile != dbfile)
        DBClose(blockDbfile);
    }

    unifiedDataArray->unmap();
    DBFreeMultivar(mv);

    // Create unified spatial field
    auto field = scene.createObject<SpatialField>(tokens::spatial_field::structuredRegular);
    field->setName(multivarName);
    field->setParameter("origin", globalMin);
    field->setParameter("spacing", globalSpacing);
    field->setParameterObject("data", *unifiedDataArray);

    // Create single volume with unified field
    logStatus("[import_SILO] creating unified volume with range: %.2f to %.2f",
        globalValueRange.x, globalValueRange.y);

    auto [inst, volume] = scene.insertNewChildObjectNode<Volume>(
        tx, tokens::volume::transferFunction1D);
    volume->setName(multivarName);
    volume->setParameterObject("value", *field);
    volume->setParameterObject("color", *colorArray);
    volume->setParameter("valueRange", ANARI_FLOAT32_BOX1, &globalValueRange);
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
