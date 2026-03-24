// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/animation/CallbackBinding.hpp"

namespace tsd::animation {

CallbackBinding::CallbackBinding(Callback callback)
    : m_callback(std::move(callback))
{}

void CallbackBinding::update(float t)
{
  if (m_callback)
    m_callback(t);
}

} // namespace tsd::animation
