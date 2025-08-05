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

namespace tsd::io {

#if TSD_USE_VTK
static anari::DataType vtkTypeToANARIType(int vtkType)
{
  switch (vtkType) {
  case VTK_FLOAT:
    return ANARI_FLOAT32;
  case VTK_DOUBLE:
    return ANARI_FLOAT64;
  case VTK_CHAR:
    return ANARI_FIXED8;
  case VTK_SHORT:
    return ANARI_FIXED16;
  case VTK_INT:
    return ANARI_FIXED32;
  case VTK_UNSIGNED_CHAR:
    return ANARI_UFIXED8;
  case VTK_UNSIGNED_SHORT:
    return ANARI_UFIXED16;
  case VTK_UNSIGNED_INT:
    return ANARI_UFIXED32;
  default:
    logError("[import_VTI] unsupported vtk type %d", vtkType);
    return ANARI_UNKNOWN;
  }
}

static ArrayRef makeArray3D(
    Context &ctx, vtkDataArray *array, vtkIdType w, vtkIdType h, vtkIdType d)
{
  void *ptr = array->GetVoidPointer(0);
  int vtkType = array->GetDataType();

  auto arr = ctx.createArray(vtkTypeToANARIType(vtkType), w, h, d);
  arr->setData(ptr);
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
  field->setParameter("origin", float3(origin[0], origin[1], origin[2]));
  field->setParameter("spacing", float3(spacing[0], spacing[1], spacing[2]));

  // --- Write point data array ---
  vtkPointData *pointData = grid->GetPointData();
  for (uint32_t i = 0; i < pointData->GetNumberOfArrays(); ++i) {
    vtkDataArray *array = pointData->GetArray(i);

    int numComponents = array->GetNumberOfComponents();
    if (numComponents > 1) {
      logWarning(
          "[import_VTI] only single-component arrays are supported, "
          "array '%s' has %d components -- only using first component",
          array->GetName(),
          numComponents);
      continue;
    }

    auto a = makeArray3D(ctx, array, dims[0], dims[1], dims[2]);
    field->setParameterObject("data", *a);
    break;
  }

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
