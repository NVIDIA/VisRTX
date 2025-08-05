// Copyright 2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#ifndef TSD_USE_VTK
#define TSD_USE_VTK 1
#endif

#include "tsd/io/importers.hpp"
#include "tsd/io/importers/detail/importer_common.hpp"
#include "tsd/core/Logging.hpp"
#if TSD_USE_VTK
// vtk
#include <vtkCell.h>
#include <vtkCellData.h>
#include <vtkDataArray.h>
#include <vtkIdList.h>
#include <vtkPointData.h>
#include <vtkSmartPointer.h>
#include <vtkUnstructuredGrid.h>
#include <vtkXMLUnstructuredGridReader.h>
#endif
// std
#include <iomanip>
#include <iostream>

namespace tsd::io {

#if TSD_USE_VTK
static ArrayRef makeFloatArray1D(
    Context &ctx, vtkDataArray *array, vtkIdType count)
{
  int numComponents = array->GetNumberOfComponents();
  if (numComponents > 1) {
    logWarning(
        "[import_VTU] only single-component arrays are supported, "
        "array '%s' has %d components -- only using first component",
        array->GetName(),
        numComponents);
  }
  auto arr = ctx.createArray(ANARI_FLOAT32, count);
  auto *buffer = arr->mapAs<float>();
  for (vtkIdType i = 0; i < count; ++i)
    buffer[i] = static_cast<float>(array->GetComponent(i, 0));
  arr->unmap();
  return arr;
}

SpatialFieldRef import_VTU(Context &ctx, const char *filepath)
{
  // Read .vtu file
  vtkSmartPointer<vtkXMLUnstructuredGridReader> reader =
      vtkSmartPointer<vtkXMLUnstructuredGridReader>::New();
  reader->SetFileName(filepath);
  reader->Update();

  vtkUnstructuredGrid *grid = reader->GetOutput();
  if (!grid) {
    logError("[import_VTU] failed to load .vtu file '%s'", filepath);
    return {};
  }

  auto field =
      ctx.createObject<SpatialField>(tokens::spatial_field::unstructured);
  field->setName(fileOf(filepath).c_str());

  vtkIdType numPoints = grid->GetNumberOfPoints();
  vtkIdType numCells = grid->GetNumberOfCells();

  // --- Write vertex positions ---
  auto vertexArray = ctx.createArray(ANARI_FLOAT32_VEC3, numPoints);
  auto *vertexData = vertexArray->mapAs<float3>();
  for (vtkIdType i = 0; i < numPoints; ++i) {
    double *pt = grid->GetPoint(i);
    vertexData[i] = float3(pt[0], pt[1], pt[2]);
  }
  vertexArray->unmap();
  field->setParameterObject("vertex.position", *vertexArray);

  // --- Write cell indices (flattened connectivity) ---
  std::vector<uint32_t> connectivity;
  std::vector<uint32_t> cellIndex;
  for (vtkIdType i = 0; i < numCells; ++i) {
    vtkCell *cell = grid->GetCell(i);
    int n = cell->GetNumberOfPoints();
    cellIndex.push_back(static_cast<uint32_t>(connectivity.size()));
    for (int j = 0; j < n; ++j)
      connectivity.push_back(static_cast<uint32_t>(cell->GetPointId(j)));
  }
  auto indexArray = ctx.createArray(ANARI_UINT32, connectivity.size());
  indexArray->setData(connectivity.data());
  field->setParameterObject("index", *indexArray);
  auto cellIndexArray = ctx.createArray(ANARI_UINT32, cellIndex.size());
  cellIndexArray->setData(cellIndex.data());
  field->setParameterObject("cell.index", *cellIndexArray);

  // --- Write cell types ---
  auto cellTypesArray = ctx.createArray(ANARI_UINT8, numCells);
  auto *cellTypes = cellTypesArray->mapAs<uint8_t>();
  for (vtkIdType i = 0; i < numCells; ++i)
    cellTypes[i] = static_cast<uint8_t>(grid->GetCellType(i));
  cellTypesArray->unmap();
  field->setParameterObject("cell.type", *cellTypesArray);

  // --- Write point data arrays ---
  vtkPointData *pointData = grid->GetPointData();
  uint32_t numPointArrays = pointData->GetNumberOfArrays();
  for (uint32_t i = 0; i < std::min(1u, numPointArrays); ++i) {
    vtkDataArray *array = pointData->GetArray(i);
    auto a = makeFloatArray1D(ctx, array, numPoints);
    field->setParameterObject("vertex.data", *a);
  }

  // --- Write cell data arrays ---
  vtkCellData *cellData = grid->GetCellData();
  uint32_t numCellArrays = cellData->GetNumberOfArrays();
  for (uint32_t i = 0; i < std::min(1u, numCellArrays); ++i) {
    vtkDataArray *array = cellData->GetArray(i);
    auto a = makeFloatArray1D(ctx, array, numCells);
    field->setParameterObject("cell.data", *a);
  }

  return field;
}
#else
SpatialFieldRef import_VTU(Context &ctx, const char *filepath)
{
  logError("[import_VTU] VTK not enabled in TSD build.");
  return {};
}
#endif

} // namespace tsd
