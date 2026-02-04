// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/network/Message.hpp"
// tsd_core
#include "tsd/core/scene/Scene.hpp"

namespace tsd::network::messages {

struct ParameterRemove : public StructuredMessage
{
  // Sender -- will serialize the data on construction
  ParameterRemove(
      const tsd::core::Object *obj, const tsd::core::Parameter *param);

  // Receiver -- will setup deserialization on execute()
  ParameterRemove(const Message &msg, tsd::core::Scene *scene);

  // Receiver behavior
  void execute() override;

 private:
  tsd::core::Scene *m_scene{nullptr};
};

} // namespace tsd::network::messages
