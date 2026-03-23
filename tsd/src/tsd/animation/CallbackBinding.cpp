// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/animation/CallbackBinding.hpp"

namespace tsd::animation {

CallbackBinding::CallbackBinding(Callback callback)
    : m_callback(std::move(callback))
{}

void CallbackBinding::invoke(float time) const
{
  if (m_callback)
    m_callback(time);
}

} // namespace tsd::animation
