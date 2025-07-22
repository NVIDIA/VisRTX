// Copyright 2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#ifndef TSD_USE_VTK
#define TSD_USE_VTK 1
#endif

#include "tsd/authoring/importers.hpp"
#include "tsd/authoring/importers/detail/importer_common.hpp"
#include "tsd/core/Logging.hpp"
#if TSD_USE_VTK
// vtk
#include <vtkCellData.h>
#include <vtkDataArray.h>
#include <vtkImageData.h>
#include <vtkNew.h>
#include <vtkPointData.h>
#include <vtkXMLImageDataReader.h>
#endif
// std
#include <iomanip>
#include <iostream>

namespace tsd {

#if TSD_USE_VTK
static ArrayRef makeArray3D(
    Context &ctx, vtkDataArray *array, vtkIdType w, vtkIdType h, vtkIdType d)
{
  int numComponents = array->GetNumberOfComponents();
  if (numComponents > 1) {
    logWarning(
        "[import_VTU] only single-component arrays are supported, "
        "array '%s' has %d components -- only using first component",
        array->GetName(),
        numComponents);
  }
  auto arr = ctx.createArray(ANARI_FLOAT32, w, h, d);
  auto *buffer = arr->mapAs<float>();
  for (vtkIdType i = 0; i < (w * h * d); ++i)
    buffer[i] = static_cast<float>(array->GetComponent(i, 0));
  arr->unmap();
  return arr;
}

SpatialFieldRef import_VTI(Context &ctx, const char *filepath)
{
  vtkNew<vtkXMLImageDataReader> reader;
  reader->SetFileName(filepath);
  reader->Update();

  vtkImageData *grid = reader->GetOutput();

  if (!grid) {
    logError("[import_VTI] failed to load .vti file '%s'", filepath);
    return {};
  }

  int dims[3] = {0, 0, 0};
  double spacing[3] = {1.0, 1.0, 1.0};
  double origin[3] = {0.0, 0.0, 0.0};

  grid->GetDimensions(dims);
  grid->GetSpacing(spacing);
  grid->GetOrigin(origin);

  auto field =
      ctx.createObject<SpatialField>(tokens::spatial_field::structuredRegular);
  field->setName(fileOf(filepath).c_str());

  // --- Write point data arrays ---
  vtkPointData *pointData = grid->GetPointData();
  uint32_t numPointArrays = pointData->GetNumberOfArrays();
  for (uint32_t i = 0; i < std::min(1u, numPointArrays); ++i) {
    vtkDataArray *array = pointData->GetArray(i);
    auto a = makeArray3D(ctx, array, dims[0], dims[1], dims[2]);
    field->setParameterObject("data", *a);
  }

  field->setParameter("origin", float3(origin[0], origin[1], origin[2]));
  field->setParameter("spacing", float3(spacing[0], spacing[1], spacing[2]));

  return field;
}
#else
SpatialFieldRef import_VTI(Context &ctx, const char *filepath)
{
  logError("[import_VTI] VTK not enabled in TSD build.");
  return {};
}
#endif

} // namespace tsd
