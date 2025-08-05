// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// tsd_core
#include <tsd/core/TSDMath.hpp>
// wormhole
#include "wormhole/RMAWindow.hpp"
// std
#include <cstdio>

#define VERBOSE_STATUS_MESSAGES 0

extern int g_rank;

template <typename... Args>
inline void rank_printf(const char *fmt, Args &&...args)
{
#if VERBOSE_STATUS_MESSAGES
  printf("====[RANK %i]", g_rank);
  printf(fmt, std::forward<Args>(args)...);
  fflush(stdout);
#endif
}

struct CameraState
{
  tsd::math::float3 position{0.f, 0.f, 0.f};
  tsd::math::float3 direction{1.f, 0.f, 0.f};
  tsd::math::float3 up{0.f, 1.f, 0.f};
  float fovy{tsd::math::radians(60.f)};
  float aspect{1.f};
  float apertureRadius{0.f};
  float focusDistance{1.f};
  size_t version{0};
};

struct RendererState
{
  tsd::math::float3 ambientColor{1.f, 1.f, 1.f};
  float ambientRadiance{1.f};
  tsd::math::float4 background{0.f, 0.f, 0.f, 1.f};
  size_t version{0};
};

struct FrameState
{
  tsd::math::int2 size{1584, 600};
  size_t version{0};
};

struct RemoteAppState
{
  CameraState camera;
  RendererState renderer;
  FrameState frame;
  bool running{true};
};

using RemoteAppStateWindow = wormhole::RMAWindow<RemoteAppState>;
