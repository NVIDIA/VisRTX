// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/animation/Animation.hpp"
#include "tsd/animation/FileBinding.hpp"
#include "tsd/scene/ObjectUsePtr.hpp"
#include "tsd/scene/Scene.hpp"
#include "tsd/scene/objects/Geometry.hpp"
// std
#include <optional>
#include <string>
#include <vector>

namespace tsd::io {

struct EnSightFileBinding : public tsd::animation::FileBinding
{
  struct FieldMapping
  {
    core::Token attributeName; // "vertex.attribute0", etc.
    std::string ensightVarName; // "SOMEVAR"
    std::string type; // "scalar" or "vector"
    std::vector<std::string> files; // expanded per-frame file paths
  };

  struct PartBinding
  {
    int partId;
    scene::ObjectUsePtr<scene::Geometry, scene::Object::UseKind::ANIM> geometry;
  };

  struct SerializedData
  {
    std::vector<PartBinding> parts;
    std::vector<std::string> geoFiles;
    std::vector<FieldMapping> fieldMappings;
  };

  EnSightFileBinding(scene::Scene *scene,
      std::vector<PartBinding> parts,
      std::vector<std::string> geoFiles,
      std::vector<FieldMapping> fieldMappings);

  static std::optional<SerializedData> fromDataNode(
      scene::Scene &scene, tsd::core::DataNode &node);

  // FileBinding interface

  std::string kind() const override;
  void toDataNode(tsd::core::DataNode &node) const override;
  void onDefragment(const scene::IndexRemapper &cb) override;
  void update(float t) override;

  size_t frameCount() const;
  int currentFrame() const;

 private:
  void addCallbackToAnimation(tsd::animation::Animation &anim) override;
  void loadFrame(int idx);

  std::vector<PartBinding> m_parts;
  std::vector<std::string> m_geoFiles;
  std::vector<FieldMapping> m_fieldMappings;
  int m_currentFrame{-1};
};

} // namespace tsd::io
