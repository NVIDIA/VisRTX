// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/core/TypeMacros.hpp"
#include "tsd/io/importers/detail/usd/UsdImportContext.h"
// usd
#include <pxr/imaging/hd/sceneIndex.h>
#include <pxr/usd/sdf/path.h>
#include <pxr/usd/usd/prim.h>
// std
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tsd::io::usd {

// The root of the subtree UsdImaging synthesises for native instancing. Its
// contents reach the Scene through their instancers, never directly.
constexpr const char *NATIVE_INSTANCING_ROOT = "/UsdNiPropagatedPrototypes";

// One Prototype converted once, as a flat set of Surfaces with each gprim's
// Prototype-root-relative transform baked into its vertex data (ADR 0016).
// Empty `surfaces` with `internalTransformsAnimated` set means the Prototype
// could not be baked and must be expanded per placement instead.
struct PrototypeContent
{
  std::vector<SurfaceRef> surfaces;
  // Populated only when the Prototype could not be baked: the resolved path of
  // the gprim each Surface came from, so placements can be expanded with the
  // gprim's own transform while still sharing the Surface.
  std::vector<pxr::SdfPath> gprimPaths;
  bool internalTransformsAnimated{false};
};

/*
 * What one resolved point-instancer prim says about its placements: which
 * Prototypes it scatters, the transform of every instance id, which ids each
 * Prototype claims, and which of them USD marks invisible.
 *
 * This is read once per instancer -- reading the per-instance transforms is
 * the expensive part at half a million instances -- and then queried once per
 * Prototype.
 */
struct InstancerPlacements
{
  pxr::VtArray<pxr::SdfPath> prototypes;
  // Non-empty only for native instancing, whose placements are attached to
  // each USD Instance's own node rather than to a transform array.
  pxr::VtArray<pxr::SdfPath> instanceLocations;
  std::vector<pxr::VtIntArray> instanceIndices;
  pxr::VtBoolArray mask;
  std::vector<tsd::math::mat4> transforms;

  bool isVisible(int instanceId) const;

  // The visible placements of one Prototype, in instance-index order.
  // Placements USD marks invisible are omitted rather than emitted hidden, so
  // this is also what an animation binding must reproduce to keep a scrub
  // selecting the same instances the Import did.
  std::vector<tsd::math::mat4> forPrototype(size_t prototypeIndex) const;
};

// Read one resolved instancer prim. Callers that only need the placements of a
// single Prototype still pay one read of the whole instancer, which is why
// this is separate from forPrototype(). Native instancing places its
// Prototypes at their own nodes rather than through a transform array, so its
// per-instance transforms -- the expensive part of this read at half a million
// instances -- are not read at all.
InstancerPlacements readInstancerPlacements(const pxr::HdSceneIndexPrim &prim);

// Every time code authored on any attribute that moves a point instancer's
// placements, in order and without duplicates. Empty when the instancer does
// not animate. Reads the raw Stage prim, not the resolved one.
std::vector<double> pointInstancerSampleTimes(const pxr::UsdPrim &prim);

/*
 * State shared between the mirrored-hierarchy traversal and the instancing
 * pass: where the prims that native instancing will place on landed in the
 * Layer, and every Prototype converted so far. Native-instance placements are
 * attached after the traversal, because their instancer lives outside the
 * mirrored hierarchy.
 *
 * Which prims those are is discovered from the instancers up front, so that
 * the traversal records a handful of nodes rather than one per prim on the
 * Stage -- a Stage-sized map to serve a lookup native instancing makes only
 * for its own placement paths, and not at all on the Stages that have none.
 */
struct InstancerRegistry
{
  explicit InstancerRegistry(const pxr::HdSceneIndexBaseRefPtr &sceneIndex);
  TSD_NOT_COPYABLE(InstancerRegistry)
  TSD_DEFAULT_MOVEABLE(InstancerRegistry)

  // Where a placement path landed in the Layer, or `fallback` if the traversal
  // never reached it.
  LayerNodeRef nodeFor(
      const pxr::SdfPath &primPath, LayerNodeRef fallback) const;

  // Remember where a prim landed, if native instancing will place on it.
  void recordNode(const pxr::SdfPath &primPath, LayerNodeRef node);

  std::unordered_map<std::string, std::shared_ptr<PrototypeContent>> prototypes;

 private:
  std::unordered_set<std::string> m_placementPaths;
  std::unordered_map<std::string, LayerNodeRef> m_nodeForPrimPath;
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
