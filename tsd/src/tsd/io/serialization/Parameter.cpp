// Copyright 2024-2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/io/serialization/Parameter.hpp"
// std
#include <string>
#include <vector>

namespace tsd::io {

void serialize_Parameter(
    const scene::Parameter &parameter, core::DataNode &node)
{
  node["value"] = parameter.value();
  node["enabled"] = parameter.isEnabled();
  if (!parameter.description().empty())
    node["description"] = parameter.description();
  if (parameter.usage() != scene::ParameterUsageHint::NONE)
    node["usage"] = static_cast<int>(parameter.usage());
  if (parameter.hasMin())
    node["min"] = parameter.min();
  if (parameter.hasMax())
    node["max"] = parameter.max();

  if (!parameter.stringValues().empty()) {
    auto &stringValues = node["stringValues"];
    for (const auto &value : parameter.stringValues())
      stringValues.append() = value;
    node["stringSelection"] = parameter.stringSelection();
  }
}

void deserialize_Parameter(core::DataNode &node, scene::Parameter &parameter)
{
  if (auto *child = node.child("description"); child != nullptr)
    parameter.setDescription(child->getValueAs<std::string>().c_str());

  if (auto *child = node.child("usage"); child != nullptr) {
    parameter.setUsage(
        static_cast<scene::ParameterUsageHint>(child->getValueAs<int>()));
  }

  if (auto *child = node.child("min"); child != nullptr)
    parameter.setMin(child->getValue());

  if (auto *child = node.child("max"); child != nullptr)
    parameter.setMax(child->getValue());

  if (auto *child = node.child("stringValues"); child != nullptr) {
    std::vector<std::string> stringValues;
    child->foreach_child([&](core::DataNode &valueNode) {
      stringValues.push_back(valueNode.getValueAs<std::string>());
    });
    parameter.setStringValues(stringValues);
    parameter.setStringSelection(node["stringSelection"].getValueAs<int>());
  }

  if (auto *child = node.child("enabled"); child != nullptr)
    parameter.setEnabled(child->getValueAs<bool>());

  parameter.setValue(node["value"].getValue());
}

} // namespace tsd::io
