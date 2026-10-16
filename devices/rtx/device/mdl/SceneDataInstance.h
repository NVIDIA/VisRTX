// Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "gpu/materialAttributes.h"
#include "libmdl/SceneDataDescriptor.h"

#include <nonstd/span.hpp>

#include <string>
#include <vector>

namespace visrtx::mdl {

class SceneDataInstance
{
 public:
  struct NamedBinding
  {
    std::string name;
    MaterialAttribute attribute{MaterialAttribute::UNKNOWN};
  };

  explicit SceneDataInstance(const libmdl::SceneDataDescriptor &descriptor);

  SceneDataInstance(const SceneDataInstance &) = default;
  SceneDataInstance &operator=(const SceneDataInstance &) = default;
  SceneDataInstance(SceneDataInstance &&) = default;
  SceneDataInstance &operator=(SceneDataInstance &&) = default;

  nonstd::span<const MDLSceneDataBinding> data() const;

  // Replace the complete snapshot; omitted names become unbound. Unknown
  // names are ignored, unsupported attributes become unbound, and the last
  // binding wins for duplicate names. Returns whether the table changed.
  // A newly constructed instance still needs an initial upload even if this
  // returns false: it starts with a correctly sized, entirely unbound table.
  bool update(nonstd::span<const NamedBinding> bindings);

 private:
  libmdl::SceneDataDescriptor m_descriptor;
  std::vector<MDLSceneDataBinding> m_bindings;
};

} // namespace visrtx::mdl
