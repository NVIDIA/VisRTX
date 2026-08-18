// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/animation/Binding.hpp"
// tsd_core
#include "tsd/core/DataTree.hpp"
// tsd_scene
#include "tsd/scene/LayerNodeData.hpp"
// std
#include <string>
#include <vector>

namespace tsd::animation {

struct Animation;

/*
 * Abstract base for animation bindings that load data from files.  Derived
 * classes live in tsd_io (which has access to format-specific importers) and
 * register runtime callbacks on the owning Animation.  The base class provides
 * the interface used by tsd_io serialization free-functions to save and
 * reconstruct these bindings without creating a cyclic dependency between
 * tsd_animation and tsd_io.
 *
 * Example:
 *   struct MyFileBinding : FileBinding {
 *     std::string kind() const override { return "myKind"; }
 *     void toDataNode(core::DataNode &n) const override { ... }
 *     void addCallbackToAnimation(Animation &a) override { ... }
 *   };
 */
struct FileBinding : public Binding
{
  FileBinding() = default;
  FileBinding(scene::Scene *scene);
  virtual ~FileBinding() = default;

  void update(float t) override = 0;

  // Type tag written to / read from the serialized DataNode.
  virtual std::string kind() const = 0;

  // Write binding-specific data to node (called by animationToNode in tsd_io).
  virtual void toDataNode(core::DataNode &node) const = 0;

  // Layer nodes this binding writes to, if any.  Archive planning classifies
  // these exactly as it classifies a transform binding's target, so a binding
  // that drives layer state rather than an object needs no per-kind case.
  virtual std::vector<scene::LayerNodeRef> layerTargets() const;

 protected:
  // Register the runtime callback on anim.  Called both on first import and
  // after reconstruction from a legacy application-state DataNode.
  virtual void addCallbackToAnimation(Animation &anim) = 0;

  friend struct Animation; // allow Animation to call addCallbackToAnimation()
                           // on new bindings
};

// Inlined definitions ////////////////////////////////////////////////////////

inline FileBinding::FileBinding(scene::Scene *scene) : Binding(scene) {}

inline std::vector<scene::LayerNodeRef> FileBinding::layerTargets() const
{
  return {};
}

} // namespace tsd::animation
