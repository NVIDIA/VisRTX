// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/network/Message.hpp"
// tsd_core
#include "tsd/scene/Scene.hpp"

namespace tsd::network::messages {

struct ParameterChange : public StructuredMessage
{
  // Sender -- will serialize the data on construction
  ParameterChange(
      const tsd::scene::Object *obj, const tsd::scene::Parameter *param);

  // Sender -- will serialize the data on construction
  ParameterChange(const tsd::scene::Object *obj,
      const tsd::scene::Parameter *const *params,
      size_t np);

  // Receiver -- will setup deserialization on execute()
  ParameterChange(const Message &msg, tsd::scene::Scene *scene);

  // Receiver behavior
  void execute() override;

 private:
  tsd::scene::Scene *m_scene{nullptr};
};

} // namespace tsd::network::messages
