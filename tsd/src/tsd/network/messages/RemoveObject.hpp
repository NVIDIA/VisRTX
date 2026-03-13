// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/network/Message.hpp"
// tsd_core
#include "tsd/scene/Scene.hpp"

namespace tsd::network::messages {

struct RemoveObject : public StructuredMessage
{
  // Sender -- will serialize the data on construction
  RemoveObject(const tsd::scene::Object *obj);

  // Receiver -- will setup deserialization on execute()
  RemoveObject(const Message &msg, tsd::scene::Scene *scene);

  // Receiver behavior
  void execute() override;

 private:
  tsd::scene::Scene *m_scene{nullptr};
};

} // namespace tsd::network::messages
