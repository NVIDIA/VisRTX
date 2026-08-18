// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/io/usd/UsdDataSource.h"
// usd
#include <pxr/base/vt/array.h>
#include <pxr/base/vt/value.h>
#include <pxr/imaging/hd/meshSchema.h>
#include <pxr/imaging/hd/sceneIndex.h>
#include <pxr/usd/usd/stage.h>
// std
#include <string>
#include <utility>
#include <vector>

namespace tsd::io::usd {

// A named primvar value, flattened out of any indexing.
using NamedPrimvar = std::pair<std::string, pxr::VtValue>;

/*
 * The primvars of one mesh, split by the interpolation that decides how
 * OpenSubdiv must carry them through refinement.
 */
struct MeshPrimvars
{
  std::vector<NamedPrimvar> vertex; // vertex and varying
  std::vector<NamedPrimvar> faceVarying; // one value per face corner
  std::vector<NamedPrimvar> uniform; // one value per face
};

/*
 * A mesh after OpenSubdiv refinement: the limit-level topology together with
 * every primvar carried through the same refinement, so that smooth assets do
 * not arrive faceted and their attributes stay aligned with their points.
 */
struct RefinedMesh
{
  bool valid{false};
  pxr::VtIntArray faceVertexCounts;
  pxr::VtIntArray faceVertexIndices;
  pxr::VtIntArray holeIndices;
  pxr::VtVec3fArray points;
  MeshPrimvars primvars;
};

// True when this mesh should be refined: the Stage explicitly declares a
// subdivision scheme other than "none" and the caller asked for refinement.
// USD's schema default is catmullClark for every mesh, so authoring is what
// distinguishes a subdivision surface from an ordinary polygon mesh. Reads the
// raw Stage prim, which is where the scheme is authored.
bool meshWantsRefinement(const pxr::UsdStageRefPtr &stage,
    int refinementLevel,
    const pxr::SdfPath &primPath);

// Refine with OpenSubdiv, honouring subdivision tags -- creases, corners, and
// holes -- carried on the resolved prim.
RefinedMesh refineMesh(const pxr::HdMeshSchema &meshSchema,
    const pxr::VtIntArray &faceVertexCounts,
    const pxr::VtIntArray &faceVertexIndices,
    const pxr::VtIntArray &holeIndices,
    const pxr::TfToken &orientation,
    const pxr::VtVec3fArray &points,
    const MeshPrimvars &primvars,
    int refinementLevel);

} // namespace tsd::io::usd
