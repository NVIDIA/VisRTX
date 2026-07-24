// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/io/importers/detail/usd/UsdImportContext.h"
// usd
#include <pxr/imaging/hd/sceneIndex.h>
// std
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace tsd::io::usd {

// One Prototype converted once, as a flat set of Surfaces with each gprim's
// Prototype-root-relative transform baked into its vertex data (ADR 0016).
// Empty `surfaces` with `internalTransformsAnimated` set means the Prototype
// could not be baked and must be expanded per placement instead.
struct PrototypeContent
{
  std::vector<SurfaceRef> surfaces;
  bool internalTransformsAnimated{false};
};

/*
 * State shared between the mirrored-hierarchy traversal and the instancing
 * pass: where each USD prim landed in the Layer, and every Prototype converted
 * so far. Native-instance placements are attached after the traversal, because
 * their instancer lives outside the mirrored hierarchy.
 */
struct InstancerRegistry
{
  std::unordered_map<std::string, LayerNodeRef> nodeForPrimPath;
  std::unordered_map<std::string, std::shared_ptr<PrototypeContent>> prototypes;
};

// Turn one resolved instancer prim into instancing Layer content beneath
// `node`. Point instancers become a single transform-array node; native
// instancers are deferred to attachNativeInstances().
void convertInstancer(ImportContext &ctx,
    const pxr::HdSceneIndexBaseRefPtr &sceneIndex,
    const pxr::SdfPath &primPath,
    const pxr::HdSceneIndexPrim &prim,
    LayerNodeRef node,
    InstancerRegistry &registry);

// Attach each USD Instance's shared Prototype objects at the placement's own
// node in the mirrored hierarchy.
void attachNativeInstances(ImportContext &ctx,
    const pxr::HdSceneIndexBaseRefPtr &sceneIndex,
    InstancerRegistry &registry,
    LayerNodeRef importRoot);

} // namespace tsd::io::usd
