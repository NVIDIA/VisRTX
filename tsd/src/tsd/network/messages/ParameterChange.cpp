// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "ParameterChange.hpp"
// tsd_core
#include "tsd/core/Logging.hpp"
// tsd_io
#include "tsd/io/serialization.hpp"

namespace tsd::network::messages {

ParameterChange::ParameterChange(
    const tsd::scene::Object *obj, const tsd::scene::Parameter *param)
    : ParameterChange(obj, &param, 1)
{}

ParameterChange::ParameterChange(const tsd::scene::Object *obj,
    const tsd::scene::Parameter *const *params,
    size_t np)
{
  if (!obj) {
    tsd::core::logError(
        "[message::ParameterChange] No object provided for multi-param ctor");
    return;
  }

  auto root = m_tree.root();
  root["o"] = tsd::core::Any(obj->type(), obj->index()); // object

  auto &ps = root["p"];

  for (size_t i = 0; i < np; ++i) {
    auto &paramNode = ps.append(); // parameter node
    auto *param = params[i];
    paramNode["n"] = param->name().str(); // parameter name
    tsd::io::parameterToNode(*param, paramNode["v"]); // parameter value + info
  }
}

ParameterChange::ParameterChange(const Message &msg, tsd::scene::Scene *scene)
    : StructuredMessage(msg), m_scene(scene)
{
#if 0
  tsd::core::logDebug(
      "[message::ParameterChange] Received message (%zu bytes)",
      msg.header.payload_length);
#endif
}

void ParameterChange::execute()
{
  if (!m_scene) {
    tsd::core::logError(
        "[message::ParameterChange] No scene provided for exec");
    return;
  }

  auto &root = m_tree.root();
  auto o = root["o"].getValue();
  auto obj = m_scene->getObject(o);
  if (!obj) {
    tsd::core::logError(
        "[message::ParameterChange] Unable to find object (%s, %zu)",
        anari::toString(o.type()),
        o.getAsObjectIndex());
    return;
  }

  auto &pn = root["p"];

  pn.foreach_child([&](core::DataNode &child) {
    auto paramName = child["n"].getValueAs<std::string>();
    auto &p = obj->addParameter(paramName.c_str()); // parameter may be new
    tsd::io::nodeToParameter(child["v"], p);
  });
}

} // namespace tsd::network::messages
