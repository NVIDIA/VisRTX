// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "ParameterChange.hpp"
// tsd_core
#include "tsd/core/Logging.hpp"
// tsd_io
#include "tsd/io/serialization.hpp"

namespace tsd::network::messages {

ParameterChange::ParameterChange(
    const tsd::core::Object *obj, const tsd::core::Parameter *param)
{
  if (!(obj && param)) {
    tsd::core::logError(
        "[message::ParameterChange] No object or parameter provided");
    return;
  }

  // NOTE(jda) - node names intentionally short to reduce message size
  auto root = m_tree.root();
  root["o"] = tsd::core::Any(obj->type(), obj->index()); // object
  root["n"] = param->name().str(); // parameter name
  tsd::io::parameterToNode(*param, root["v"]); // parameter value + info
}

ParameterChange::ParameterChange(const Message &msg, tsd::core::Scene *scene)
    : StructuredMessage(msg), m_scene(scene)
{
  tsd::core::logStatus(
      "[message::ParameterChange] Received object parameter from server"
      " (%zu bytes)",
      msg.header.payload_length);
}

void ParameterChange::execute()
{
  if (!m_scene) {
    tsd::core::logError(
        "[message::ParameterChange] No scene provided for exec");
    return;
  }

  auto o = m_tree.root()["o"].getValue();
  auto obj = m_scene->getObject(o);
  if (!obj) {
    tsd::core::logError(
        "[message::ParameterChange] Unable to find object (%s, %zu)",
        anari::toString(o.type()),
        o.getAsObjectIndex());
    return;
  }

  auto *paramName = m_tree.root()["n"].getValueAs<std::string>().c_str();
  auto *p = obj->parameter(paramName);
  if (!p) {
    tsd::core::logError(
        "[message::ParameterChange] Unable to find parameter '%s' on object "
        "(%s, %zu)",
        paramName,
        anari::toString(o.type()),
        o.getAsObjectIndex());
    return;
  }

  tsd::io::nodeToParameter(m_tree.root()["v"], *p);
}

} // namespace tsd::network::messages
