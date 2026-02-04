// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "ParameterRemove.hpp"
// tsd_core
#include "tsd/core/Logging.hpp"
// tsd_io
#include "tsd/io/serialization.hpp"

namespace tsd::network::messages {

ParameterRemove::ParameterRemove(
    const tsd::core::Object *obj, const tsd::core::Parameter *param)
{
  if (!(obj && param)) {
    tsd::core::logError(
        "[message::ParameterRemove] No object or parameter provided");
    return;
  }

  // NOTE(jda) - node names intentionally short to reduce message size
  auto root = m_tree.root();
  root["o"] = tsd::core::Any(obj->type(), obj->index()); // object
  root["n"] = param->name().str(); // parameter name
}

ParameterRemove::ParameterRemove(const Message &msg, tsd::core::Scene *scene)
    : StructuredMessage(msg), m_scene(scene)
{
  tsd::core::logDebug("[message::ParameterRemove] Received message (%zu bytes)",
      msg.header.payload_length);
}

void ParameterRemove::execute()
{
  if (!m_scene) {
    tsd::core::logError(
        "[message::ParameterRemove] No scene provided for exec");
    return;
  }

  auto o = m_tree.root()["o"].getValue();
  auto obj = m_scene->getObject(o);
  if (!obj) {
    tsd::core::logError(
        "[message::ParameterRemove] Unable to find object (%s, %zu)",
        anari::toString(o.type()),
        o.getAsObjectIndex());
    return;
  }

  auto paramName = m_tree.root()["n"].getValueAs<std::string>();
  obj->removeParameter(paramName.c_str());
}

} // namespace tsd::network::messages
