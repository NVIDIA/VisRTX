// Copyright 2025-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#ifndef TSD_USE_VTK
#define TSD_USE_VTK 1
#endif

#include "tsd/core/Logging.hpp"
#include "tsd/io/importers.hpp"
#include "tsd/io/importers/detail/importer_common.hpp"
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
SpatialFieldRef import_VTI(Scene &scene, const char *filepath)
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

  auto field = scene.createObject<SpatialField>(
      tokens::spatial_field::structuredRegular);
  field->setName(fileOf(filepath).c_str());
  field->setParameter("origin", float3(origin[0], origin[1], origin[2]));
  field->setParameter("spacing", float3(spacing[0], spacing[1], spacing[2]));

  // --- Write point data arrays ---
  vtkPointData *pointData = grid->GetPointData();
  for (uint32_t i = 0; i < pointData->GetNumberOfArrays(); ++i) {
    vtkDataArray *array = pointData->GetArray(i);

    int numComponents = array->GetNumberOfComponents();

    if (numComponents == 1) {
      auto a = makeArray3DFromVTK(
          scene, array, dims[0], dims[1], dims[2], "[import_VTI]");
      field->setParameterObject("data", *a);
      break;
    } else if (numComponents == 3) {
      // Split into 3 scalar SpatialFields: {name}_x, {name}_y, {name}_z
      const char *arrName = array->GetName();
      std::string baseName = (arrName && arrName[0] != '\0')
          ? std::string(arrName)
          : fileOf(filepath);

      std::string nameX = baseName + "_x";
      std::string nameY = baseName + "_y";
      std::string nameZ = baseName + "_z";

      size_t n =
          (size_t)dims[0] * (size_t)dims[1] * (size_t)dims[2];

      auto arrX = scene.createArray(ANARI_FLOAT32, dims[0], dims[1], dims[2]);
      auto arrY = scene.createArray(ANARI_FLOAT32, dims[0], dims[1], dims[2]);
      auto arrZ = scene.createArray(ANARI_FLOAT32, dims[0], dims[1], dims[2]);

      float *pX = arrX->mapAs<float>();
      float *pY = arrY->mapAs<float>();
      float *pZ = arrZ->mapAs<float>();

      for (vtkIdType idx = 0; idx < (vtkIdType)n; ++idx) {
        double tuple[3] = {0.0, 0.0, 0.0};
        array->GetTuple(idx, tuple);
        pX[idx] = (float)tuple[0];
        pY[idx] = (float)tuple[1];
        pZ[idx] = (float)tuple[2];
      }

      arrX->unmap();
      arrY->unmap();
      arrZ->unmap();

      // Main field (_x component)
      field->setName(nameX.c_str());
      field->setParameterObject("data", *arrX);

      // _y component field in the scene object pool
      auto fieldY = scene.createObject<SpatialField>(
          tokens::spatial_field::structuredRegular);
      fieldY->setName(nameY.c_str());
      fieldY->setParameter("origin", float3(origin[0], origin[1], origin[2]));
      fieldY->setParameter(
          "spacing", float3(spacing[0], spacing[1], spacing[2]));
      fieldY->setParameterObject("data", *arrY);

      // _z component field in the scene object pool
      auto fieldZ = scene.createObject<SpatialField>(
          tokens::spatial_field::structuredRegular);
      fieldZ->setName(nameZ.c_str());
      fieldZ->setParameter("origin", float3(origin[0], origin[1], origin[2]));
      fieldZ->setParameter(
          "spacing", float3(spacing[0], spacing[1], spacing[2]));
      fieldZ->setParameterObject("data", *arrZ);

      logStatus(
          "[import_VTI] split 3-component array '%s' into '%s', '%s', '%s'",
          baseName.c_str(),
          nameX.c_str(),
          nameY.c_str(),
          nameZ.c_str());
      break;
    } else {
      logWarning(
          "[import_VTI] array '%s' has %d components (only 1 or 3 are "
          "supported) -- skipping",
          array->GetName(),
          numComponents);
    }
  }

  return field;
}
#else
SpatialFieldRef import_VTI(Scene &scene, const char *filepath)
{
  logError("[import_VTI] VTK not enabled in TSD build.");
  return {};
}
#endif

} // namespace tsd::io
