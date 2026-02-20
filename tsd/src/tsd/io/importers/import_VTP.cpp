// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#ifndef TSD_USE_VTK
#define TSD_USE_VTK 1
#endif

// tsd_io
#include "tsd/io/importers.hpp"
#include "tsd/io/importers/detail/importer_common.hpp"
// tsd_core
#include "tsd/core/Logging.hpp"
#include "tsd/core/algorithms/computeScalarRange.hpp"
#if TSD_USE_VTK
// vtk
#include <vtkAbstractArray.h>
#include <vtkCellArray.h>
#include <vtkCellData.h>
#include <vtkDataArray.h>
#include <vtkPointData.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkSmartPointer.h>
#include <vtkTriangleFilter.h>
#include <vtkUnsignedCharArray.h>
#include <vtkVariant.h>
#include <vtkXMLPolyDataReader.h>
#endif

namespace tsd::io {

#if TSD_USE_VTK
void import_VTP(Scene &scene, const char *filepath, LayerNodeRef location)
{
  auto filename = fileOf(filepath);

  auto reader = vtkSmartPointer<vtkXMLPolyDataReader>::New();
  reader->SetFileName(filepath);
  reader->Update();

  auto *polyData = reader->GetOutput();
  if (!polyData) {
    tsd::core::logError("[import_VTP] Failed to read VTP file: %s", filepath);
    return;
  }

  auto triangleFilter = vtkSmartPointer<vtkTriangleFilter>::New();
  triangleFilter->SetInputData(polyData);
  triangleFilter->Update();

  vtkPolyData *triangleMesh = triangleFilter->GetOutput();

  // --- Collect relevant VTK classes --- //

  vtkPoints *points = triangleMesh->GetPoints();
  if (!points || points->GetNumberOfPoints() == 0) {
    tsd::core::logError("[import_VTP] VTP file contains no points");
    return;
  }

  vtkCellArray *triangles = triangleMesh->GetPolys();
  if (!triangles || triangles->GetNumberOfCells() == 0) {
    tsd::core::logError("[import_VTP] VTP file contains no triangle cells");
    return;
  }

  // --- Extract points as float --- //

  const vtkIdType numPoints = points->GetNumberOfPoints();
  auto vertexArray = scene.createArray(ANARI_FLOAT32_VEC3, numPoints);
  auto *vertexPtr = vertexArray->mapAs<math::float3>();
  double p[3];
  for (vtkIdType i = 0; i < numPoints; ++i) {
    points->GetPoint(i, p);
    vertexPtr[i] = math::float3(static_cast<float>(p[0]),
        static_cast<float>(p[1]),
        static_cast<float>(p[2]));
  }
  vertexArray->unmap();

  // --- Extract triangles as uint32_t --- //

  vtkSmartPointer<vtkIdList> idList = vtkSmartPointer<vtkIdList>::New();
  const vtkIdType numCells = triangles->GetNumberOfCells();
  std::vector<uint32_t> triangleIndices;
  triangleIndices.reserve(numCells * 3);

  triangles->InitTraversal();
  while (triangles->GetNextCell(idList)) {
    const vtkIdType n = idList->GetNumberOfIds();
    if (n != 3) {
      tsd::core::logWarning(
          "[import_VTP] Non-triangle cell with %d points found (skipped).", n);
      continue;
    }
    for (int i = 0; i < 3; ++i) {
      vtkIdType idx = idList->GetId(i);
      if (idx < 0 || idx >= numPoints) {
        tsd::core::logWarning(
            "[import_VTP] Invalid point index %d (skipping triangle).", idx);
        continue;
      }
      triangleIndices.push_back(static_cast<uint32_t>(idx));
    }
  }

  auto indexArray =
      scene.createArray(ANARI_UINT32_VEC3, triangleIndices.size() / 3);
  indexArray->setData(triangleIndices.data());

  // --- Point data: serialize arrays as raw bytes --- //

  std::vector<tsd::core::ArrayRef> pointDataArrays;

  vtkPointData *pointData = triangleMesh->GetPointData();
  const int numPointArrays = pointData->GetNumberOfArrays();
  pointDataArrays.reserve(numPointArrays);

  for (int i = 0; i < numPointArrays; ++i) {
    vtkDataArray *arr = pointData->GetArray(i);
    if (arr) {
      std::string name =
          arr->GetName() ? arr->GetName() : "unnamed_point_array";
      auto a = makeArray1DFromVTK(scene, arr, "[import_VTP]");
      a->setName(name.c_str());
      pointDataArrays.push_back(a);

      tsd::core::logInfo(
          "[import_VTP]   ...imported point data array '%s' as %s",
          name.c_str(),
          anari::toString(a->elementType()));
    }
  }

  // --- Cell data: serialize arrays as raw bytes --- //

  std::vector<tsd::core::ArrayRef> cellDataArrays;

  vtkCellData *cellData = triangleMesh->GetCellData();
  const int numCellArrays = cellData->GetNumberOfArrays();
  cellDataArrays.reserve(numCellArrays);

  for (int i = 0; i < numCellArrays; ++i) {
    vtkDataArray *arr = cellData->GetArray(i);
    if (arr) {
      std::string name = arr->GetName() ? arr->GetName() : "unnamed_cell_array";
      auto a = makeArray1DFromVTK(scene, arr, "[import_VTP]");
      a->setName(name.c_str());
      cellDataArrays.push_back(a);

      tsd::core::logInfo(
          "[import_VTP]   ...imported cell data array '%s' as %s",
          name.c_str(),
          anari::toString(a->elementType()));
    }
  }

  // --- Create remaining TSD objects --- //

  // geometry

  auto mesh = scene.createObject<Geometry>(tokens::geometry::triangle);
  mesh->setName("vtp_mesh | " + std::string(filename));
  mesh->setParameterObject("vertex.position", *vertexArray);
  mesh->setParameterObject("primitive.index", *indexArray);

  for (size_t i = 0; i < pointDataArrays.size(); ++i) {
    const auto &arr = pointDataArrays[i];
    mesh->setParameterObject("vertex.attribute" + std::to_string(i), *arr);
  }

  for (size_t i = 0; i < cellDataArrays.size(); ++i) {
    const auto &arr = cellDataArrays[i];
    mesh->setParameterObject("primitive.attribute" + std::to_string(i), *arr);
  }

  // color map + material

  auto mat = scene.createObject<tsd::core::Material>(
      tokens::material::physicallyBased);

  mat->setName("vtp_material | " + std::string(filename));

  if (!pointDataArrays.empty()) {
    auto &colorArray = pointDataArrays[0];
    auto colorRange = computeScalarRange(*colorArray);
    mat->setParameterObject(
        "baseColor", *makeDefaultColorMapSampler(scene, colorRange));
  } else if (!cellDataArrays.empty()) {
    auto &colorArray = cellDataArrays[0];
    auto colorRange = computeScalarRange(*colorArray);
    mat->setParameterObject(
        "baseColor", *makeDefaultColorMapSampler(scene, colorRange));
  }

  // final surface

  auto vtp_root = scene.insertChildNode(
      location ? location : scene.defaultLayer()->root(), filename.c_str());
  scene.insertChildObjectNode(vtp_root,
      scene.createSurface(
          ("vtp_surface | " + std::string(filename)).c_str(), mesh, mat));
}
#else
void import_VTP(Scene &scene, const char *filepath, LayerNodeRef location)
{
  logError("[import_VTP] VTK not enabled in TSD build.");
}
#endif

} // namespace tsd::io
