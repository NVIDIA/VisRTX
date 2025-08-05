// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tsd/core/TSDMath.hpp>

namespace tsd::demo {

struct ParticleSystemParameters
{
  float gravity{1000.f};
  float particleMass{0.1f};
  float maxDistance{45.f};
  float deltaT{5e-4f};
};

// Compute new positions/velocities using existing GPU buffers
void particlesComputeTimestep(int numParticles,
    tsd::math::float3 *positions /* GPU */,
    tsd::math::float3 *velocities /* GPU */,
    float *distances /* GPU */,
    const tsd::math::float3 &bhPosition1,
    const tsd::math::float3 &bhPosition2,
    const ParticleSystemParameters &params);

} // namespace tsd::demo
