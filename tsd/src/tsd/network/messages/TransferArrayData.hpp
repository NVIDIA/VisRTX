// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/network/Message.hpp"
// tsd_core
#include "tsd/scene/Scene.hpp"

namespace tsd::network::messages {

struct TransferArrayData : public StructuredMessage
{
  // Sender -- will serialize the data on construction
  TransferArrayData(const tsd::scene::Array *array);

  // Receiver -- will setup deserialization on execute()
  TransferArrayData(const Message &msg, tsd::scene::Scene *scene);

  // Receiver behavior
  void execute() override;

 private:
  tsd::scene::Scene *m_scene{nullptr};
};

} // namespace tsd::network::messages