// Copyright 2025-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#ifndef TSD_USE_VTK
#define TSD_USE_VTK 1
#endif

#include "tsd/core/Logging.hpp"
#include "tsd/core/algorithms/computeScalarRange.hpp"
#include "tsd/io/importers.hpp"
#include "tsd/io/importers/detail/importer_common.hpp"
#if TSD_USE_VTK
// vtk
#include <vtkCell.h>
#include <vtkCellArray.h>
#include <vtkCellData.h>
#include <vtkDataArray.h>
#include <vtkIdList.h>
#include <vtkPointData.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkSmartPointer.h>
#include <vtkTriangleFilter.h>
#include <vtkUnstructuredGrid.h>
#include <vtkUnstructuredGridReader.h>
#include <vtkXMLUnstructuredGridReader.h>
#endif

namespace tsd::io {

#if TSD_USE_VTK

static bool isSurfaceCell(int type)
{
  return type == VTK_TRIANGLE || type == VTK_QUAD || type == VTK_POLYGON;
}

static bool isVolumeCell(int type)
{
  return type == VTK_TETRA || type == VTK_VOXEL || type == VTK_HEXAHEDRON
      || type == VTK_WEDGE || type == VTK_PYRAMID;
}

static vtkSmartPointer<vtkUnstructuredGrid> loadVTUGrid(const char *filepath)
{
  auto reader = vtkSmartPointer<vtkXMLUnstructuredGridReader>::New();
  auto legacyReader = vtkSmartPointer<vtkUnstructuredGridReader>::New();

  vtkUnstructuredGrid *grid = nullptr;
  if (reader->CanReadFile(filepath)) {
    reader->SetFileName(filepath);
    reader->Update();
    grid = reader->GetOutput();
  } else {
    legacyReader->SetFileName(filepath);
    legacyReader->Update();
    grid = legacyReader->GetOutput();
  }

  if (!grid) {
    logError("[import_VTU] failed to load .vtu file '%s'", filepath);
    return nullptr;
  }

  return grid;
}

static ArrayRef makeFloatArray1D(
    Scene &scene, vtkDataArray *array, vtkIdType count)
{
  int numComponents = array->GetNumberOfComponents();
  if (numComponents > 1) {
    logWarning(
        "[import_VTU] only single-component arrays are supported, "
        "array '%s' has %d components -- only using first component",
        array->GetName(),
        numComponents);
  }
  auto arr = scene.createArray(ANARI_FLOAT32, count);
  auto *buffer = arr->mapAs<float>();
  for (vtkIdType i = 0; i < count; ++i)
    buffer[i] = static_cast<float>(array->GetComponent(i, 0));
  arr->unmap();
  return arr;
}

static bool isScalarArray(const ArrayRef &a)
{
  if (!a)
    return false;

  const auto type = a->elementType();
  return !anari::isObject(type) && anari::componentsOf(type) == 1;
}

static ArrayRef firstScalarArray(const std::vector<ArrayRef> &arrays)
{
  for (const auto &a : arrays) {
    if (isScalarArray(a))
      return a;
  }

  return {};
}

// Build a triangle surface from surface cells in the grid
static void createSurfaceFromGrid(Scene &scene,
    vtkUnstructuredGrid *grid,
    const char *filepath,
    LayerNodeRef location)
{
  auto filename = fileOf(filepath);

  // Extract surface cells into a vtkPolyData for triangulation
  auto polyData = vtkSmartPointer<vtkPolyData>::New();
  polyData->SetPoints(grid->GetPoints());

  auto polys = vtkSmartPointer<vtkCellArray>::New();
  std::vector<vtkIdType> surfaceCellIndices;

  for (vtkIdType i = 0; i < grid->GetNumberOfCells(); ++i) {
    if (!isSurfaceCell(grid->GetCellType(i)))
      continue;
    surfaceCellIndices.push_back(i);
    vtkCell *cell = grid->GetCell(i);
    vtkIdType npts = cell->GetNumberOfPoints();
    std::vector<vtkIdType> pts(npts);
    for (vtkIdType j = 0; j < npts; ++j)
      pts[j] = cell->GetPointId(j);
    polys->InsertNextCell(npts, pts.data());
  }

  polyData->SetPolys(polys);
  polyData->GetPointData()->ShallowCopy(grid->GetPointData());

  auto *srcCellData = grid->GetCellData();
  auto *dstCellData = polyData->GetCellData();
  for (int a = 0; a < srcCellData->GetNumberOfArrays(); ++a) {
    vtkDataArray *srcArr = srcCellData->GetArray(a);
    if (!srcArr)
      continue;
    auto filtered = vtkSmartPointer<vtkDataArray>::Take(srcArr->NewInstance());
    filtered->SetName(srcArr->GetName());
    filtered->SetNumberOfComponents(srcArr->GetNumberOfComponents());
    filtered->SetNumberOfTuples(
        static_cast<vtkIdType>(surfaceCellIndices.size()));
    for (vtkIdType i = 0; i < (vtkIdType)surfaceCellIndices.size(); ++i)
      filtered->SetTuple(i, surfaceCellIndices[i], srcArr);
    dstCellData->AddArray(filtered);
  }

  // Triangulate
  auto triangleFilter = vtkSmartPointer<vtkTriangleFilter>::New();
  triangleFilter->SetInputData(polyData);
  triangleFilter->Update();

  vtkPolyData *triangleMesh = triangleFilter->GetOutput();

  vtkPoints *points = triangleMesh->GetPoints();
  if (!points || points->GetNumberOfPoints() == 0)
    return;

  vtkCellArray *triangles = triangleMesh->GetPolys();
  if (!triangles || triangles->GetNumberOfCells() == 0)
    return;

  // Build ANARI representation
  const vtkIdType numPoints = points->GetNumberOfPoints();
  auto vertexArray = scene.createArray(ANARI_FLOAT32_VEC3, numPoints);
  auto *vertexPtr = vertexArray->mapAs<tsd::math::float3>();
  double p[3];
  for (vtkIdType i = 0; i < numPoints; ++i) {
    points->GetPoint(i, p);
    vertexPtr[i] = tsd::math::float3(static_cast<float>(p[0]),
        static_cast<float>(p[1]),
        static_cast<float>(p[2]));
  }
  vertexArray->unmap();

  auto idList = vtkSmartPointer<vtkIdList>::New();
  const vtkIdType numCells = triangles->GetNumberOfCells();
  std::vector<uint32_t> triangleIndices;
  triangleIndices.reserve(numCells * 3);

  triangles->InitTraversal();
  while (triangles->GetNextCell(idList)) {
    if (idList->GetNumberOfIds() != 3)
      continue;
    for (int i = 0; i < 3; ++i)
      triangleIndices.push_back(static_cast<uint32_t>(idList->GetId(i)));
  }

  auto indexArray =
      scene.createArray(ANARI_UINT32_VEC3, triangleIndices.size() / 3);
  indexArray->setData(triangleIndices.data());

  // Vertex data
  std::vector<tsd::core::ArrayRef> vertexDataArrays;
  vtkPointData *ptData = triangleMesh->GetPointData();
  for (int i = 0; i < ptData->GetNumberOfArrays(); ++i) {
    vtkDataArray *arr = ptData->GetArray(i);
    if (!arr)
      continue;
    std::string name = arr->GetName() ? arr->GetName() : "unnamed_point_array";
    auto a = makeArray1DFromVTK(scene, arr, "[import_VTU]");
    a->setName(name.c_str());
    vertexDataArrays.push_back(a);
  }

  // Face data
  std::vector<tsd::core::ArrayRef> faceDataArrays;
  vtkCellData *clData = triangleMesh->GetCellData();
  for (int i = 0; i < clData->GetNumberOfArrays(); ++i) {
    vtkDataArray *arr = clData->GetArray(i);
    if (!arr)
      continue;
    std::string name = arr->GetName() ? arr->GetName() : "unnamed_cell_array";
    auto a = makeArray1DFromVTK(scene, arr, "[import_VTU]");
    a->setName(name.c_str());
    faceDataArrays.push_back(a);
  }

  // Assemble ANARI surfaces
  auto mesh =
      scene.createObject<tsd::core::Geometry>(tokens::geometry::triangle);
  mesh->setName(("vtu_mesh | " + std::string(filename)).c_str());
  mesh->setParameterObject("vertex.position", *vertexArray);
  mesh->setParameterObject("primitive.index", *indexArray);

  for (size_t i = 0; i < vertexDataArrays.size(); ++i)
    mesh->setParameterObject(
        "vertex.attribute" + std::to_string(i), *vertexDataArrays[i]);

  for (size_t i = 0; i < faceDataArrays.size(); ++i)
    mesh->setParameterObject(
        "primitive.attribute" + std::to_string(i), *faceDataArrays[i]);

  auto mat = scene.createObject<tsd::core::Material>(
      tokens::material::physicallyBased);
  mat->setName(("vtu_material | " + std::string(filename)).c_str());

  if (auto colorArray = firstScalarArray(vertexDataArrays); colorArray) {
    auto colorRange = tsd::core::computeScalarRange(*colorArray);
    mat->setParameterObject(
        "baseColor", *makeDefaultColorMapSampler(scene, colorRange));
  } else if (auto colorArray = firstScalarArray(faceDataArrays); colorArray) {
    auto colorRange = tsd::core::computeScalarRange(*colorArray);
    mat->setParameterObject(
        "baseColor", *makeDefaultColorMapSampler(scene, colorRange));
  } else if (!vertexDataArrays.empty() || !faceDataArrays.empty()) {
    logWarning(
        "[import_VTU] no scalar point/cell arrays found for colormapping");
  }

  scene.insertChildObjectNode(location,
      scene.createSurface(
          ("vtu_surface | " + std::string(filename)).c_str(), mesh, mat));
}

// Build an unstructured spatial field from volume cells only.
// All point-data arrays are exposed as named SpatialFields sharing topology:
//   1-component → one field named after the array
//   3-component → three fields named {arr}_x, {arr}_y, {arr}_z
// Returns the first SpatialField created (used for the scene volume wrapper).
static SpatialFieldRef createFieldFromVolumeCells(
    Scene &scene, vtkUnstructuredGrid *grid, const char *filepath)
{
  vtkIdType numCells = grid->GetNumberOfCells();

  // Collect volume cell indices
  std::vector<vtkIdType> volumeCellIndices;
  for (vtkIdType i = 0; i < numCells; ++i) {
    if (isVolumeCell(grid->GetCellType(i)))
      volumeCellIndices.push_back(i);
  }

  if (volumeCellIndices.empty())
    return {};

  vtkIdType numPoints = grid->GetNumberOfPoints();
  vtkIdType numVolumeCells = static_cast<vtkIdType>(volumeCellIndices.size());

  // Build shared topology arrays
  auto vertexArray = scene.createArray(ANARI_FLOAT32_VEC3, numPoints);
  auto *vertexData = vertexArray->mapAs<tsd::math::float3>();
  for (vtkIdType i = 0; i < numPoints; ++i) {
    double *pt = grid->GetPoint(i);
    vertexData[i] = tsd::math::float3(pt[0], pt[1], pt[2]);
  }
  vertexArray->unmap();

  std::vector<uint32_t> connectivity;
  std::vector<uint32_t> cellIndex;
  std::vector<uint8_t> cellTypes;

  for (vtkIdType idx : volumeCellIndices) {
    vtkCell *cell = grid->GetCell(idx);
    int n = cell->GetNumberOfPoints();
    int vtkType = grid->GetCellType(idx);

    cellIndex.push_back(static_cast<uint32_t>(connectivity.size()));

    if (vtkType == VTK_VOXEL) {
      // Voxels need vertex reordering to become hexahedra:
      // swap vertices 2<->3 and 6<->7
      std::vector<vtkIdType> pts(n);
      for (int j = 0; j < n; ++j)
        pts[j] = cell->GetPointId(j);
      if (n >= 8) {
        std::swap(pts[2], pts[3]);
        std::swap(pts[6], pts[7]);
      }
      for (int j = 0; j < n; ++j)
        connectivity.push_back(static_cast<uint32_t>(pts[j]));
      cellTypes.push_back(static_cast<uint8_t>(VTK_HEXAHEDRON));
    } else {
      for (int j = 0; j < n; ++j)
        connectivity.push_back(static_cast<uint32_t>(cell->GetPointId(j)));
      cellTypes.push_back(static_cast<uint8_t>(vtkType));
    }
  }

  auto indexArray = scene.createArray(ANARI_UINT32, connectivity.size());
  indexArray->setData(connectivity.data());

  auto cellIndexArray = scene.createArray(ANARI_UINT32, cellIndex.size());
  cellIndexArray->setData(cellIndex.data());

  auto cellTypesArray = scene.createArray(ANARI_UINT8, cellTypes.size());
  cellTypesArray->setData(cellTypes.data());

  // Helper: create a new unstructured SpatialField with shared topology
  auto makeTopoField = [&](const std::string &name) -> SpatialFieldRef {
    auto f = scene.createObject<tsd::core::SpatialField>(
        tokens::spatial_field::unstructured);
    f->setName(name.c_str());
    f->setParameterObject("vertex.position", *vertexArray);
    f->setParameterObject("index", *indexArray);
    f->setParameterObject("cell.index", *cellIndexArray);
    f->setParameterObject("cell.type", *cellTypesArray);
    return f;
  };

  // Vertex data: expose all point arrays
  vtkPointData *pointData = grid->GetPointData();
  SpatialFieldRef firstField;
  std::string baseName = fileOf(filepath);

  for (int i = 0; i < pointData->GetNumberOfArrays(); ++i) {
    vtkDataArray *array = pointData->GetArray(i);
    if (!array)
      continue;
    int nComp = array->GetNumberOfComponents();
    std::string arrName = (array->GetName() && array->GetName()[0] != '\0')
        ? array->GetName()
        : baseName;

    if (nComp == 1) {
      auto dataArr = scene.createArray(ANARI_FLOAT32, numPoints);
      auto *buf = dataArr->mapAs<float>();
      for (vtkIdType j = 0; j < numPoints; ++j)
        buf[j] = static_cast<float>(array->GetComponent(j, 0));
      dataArr->unmap();
      auto f = makeTopoField(arrName);
      f->setParameterObject("vertex.data", *dataArr);
      if (!firstField)
        firstField = f;
    } else if (nComp == 3) {
      for (int c = 0; c < 3; ++c) {
        const char *suffix = (c == 0) ? "_x" : (c == 1) ? "_y" : "_z";
        std::string compName = arrName + suffix;
        auto dataArr = scene.createArray(ANARI_FLOAT32, numPoints);
        auto *buf = dataArr->mapAs<float>();
        for (vtkIdType j = 0; j < numPoints; ++j)
          buf[j] = static_cast<float>(array->GetComponent(j, c));
        dataArr->unmap();
        auto f = makeTopoField(compName);
        f->setParameterObject("vertex.data", *dataArr);
        if (!firstField)
          firstField = f;
      }
      logStatus(
          "[import_VTU] split 3-component array '%s' into '%s_x', '%s_y', "
          "'%s_z'",
          arrName.c_str(),
          arrName.c_str(),
          arrName.c_str(),
          arrName.c_str());
    } else {
      logWarning(
          "[import_VTU] array '%s' has %d components (only 1 or 3 are "
          "supported) -- skipping",
          arrName.c_str(),
          nComp);
    }
  }

  // Fallback: no point arrays → create a topology-only field
  if (!firstField)
    firstField = makeTopoField(baseName);

  // Cell data: attach first scalar array to firstField for rendering
  vtkCellData *cellData = grid->GetCellData();
  for (int i = 0; i < std::min(1, cellData->GetNumberOfArrays()); ++i) {
    vtkDataArray *array = cellData->GetArray(i);
    if (!array)
      continue;
    auto a = makeFloatArray1D(scene, array, numCells);

    // Filter to volume cells only
    auto filtered = scene.createArray(ANARI_FLOAT32, numVolumeCells);
    auto *src = a->mapAs<float>();
    auto *dst = filtered->mapAs<float>();
    for (vtkIdType j = 0; j < numVolumeCells; ++j)
      dst[j] = src[volumeCellIndices[j]];
    a->unmap();
    filtered->unmap();

    firstField->setParameterObject("cell.data", *filtered);
  }

  return firstField;
}

// Full-scene importer: surfaces + volumes
void import_VTU(Scene &scene, const char *filepath, LayerNodeRef location)
{
  auto grid = loadVTUGrid(filepath);
  if (!grid)
    return;

  vtkIdType numCells = grid->GetNumberOfCells();
  bool hasSurface = false;
  bool hasVolume = false;
  int unsupportedCount = 0;

  for (vtkIdType i = 0; i < numCells; ++i) {
    int type = grid->GetCellType(i);
    if (isSurfaceCell(type))
      hasSurface = true;
    else if (isVolumeCell(type))
      hasVolume = true;
    else
      unsupportedCount++;
  }

  if (unsupportedCount > 0) {
    logWarning("[import_VTU] %d cells with unsupported types skipped",
        unsupportedCount);
  }

  auto root = scene.insertChildTransformNode(
      location ? location : scene.defaultLayer()->root(),
      tsd::math::mat4(tsd::math::identity),
      fileOf(filepath).c_str());

  if (hasSurface)
    createSurfaceFromGrid(scene, grid, filepath, root);

  if (hasVolume) {
    auto field = createFieldFromVolumeCells(scene, grid, filepath);
    if (field) {
      tsd::math::float2 valueRange = field->computeValueRange();
      auto [inst, volume] = scene.insertNewChildObjectNode<tsd::core::Volume>(
          root, tokens::volume::transferFunction1D);
      volume->setName(fileOf(filepath).c_str());
      volume->setParameterObject("value", *field);
      volume->setParameter("valueRange", ANARI_FLOAT32_BOX1, &valueRange);
      auto colorArr = scene.createArray(ANARI_FLOAT32_VEC4, 256);
      colorArr->setData(tsd::core::makeDefaultColorMap(256).data());
      volume->setParameterObject("color", *colorArr);
    }
  }
}

// Spatial field-only importer (backward compatible, used by -volume)
SpatialFieldRef import_VTU(Scene &scene, const char *filepath)
{
  auto grid = loadVTUGrid(filepath);
  if (!grid)
    return {};

  return createFieldFromVolumeCells(scene, grid, filepath);
}

#else

void import_VTU(Scene &scene, const char *filepath, LayerNodeRef location)
{
  logError("[import_VTU] VTK not enabled in TSD build.");
}

SpatialFieldRef import_VTU(Scene &scene, const char *filepath)
{
  logError("[import_VTU] VTK not enabled in TSD build.");
  return {};
}

#endif

} // namespace tsd::io
