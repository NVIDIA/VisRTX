// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "particle_system.h"
// thrust
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
// std
#include <cmath>
#include <cstring>
// glm
#include <anari/anari_cpp/ext/glm.h>

#define GPU_FCN __host__ __device__

namespace tsd::demo {

void particlesComputeTimestep(int numParticles,
    tsd::math::float3 *positions_ /* GPU */,
    tsd::math::float3 *velocities_ /* GPU */,
    float *distances /* GPU */,
    const tsd::math::float3 &bhPos1_,
    const tsd::math::float3 &bhPos2_,
    const ParticleSystemParameters &params)
{
  auto *positions = (glm::vec3 *)positions_;
  auto *velocities = (glm::vec3 *)velocities_;
  glm::vec3 bhPos1, bhPos2;
  std::memcpy(&bhPos1, &bhPos1_, sizeof(bhPos1));
  std::memcpy(&bhPos2, &bhPos2_, sizeof(bhPos2));

  float particleInvMass = 1.f / params.particleMass;

  thrust::for_each(thrust::device,
      thrust::make_counting_iterator(0),
      thrust::make_counting_iterator(numParticles),
      [=] GPU_FCN(int idx) {
        auto p = positions[idx];
        auto v = velocities[idx];

        const glm::vec3 d1 = bhPos1 - p;
        const glm::vec3 d2 = bhPos2 - p;
        const float dist1 = glm::length(d1);
        const float dist2 = glm::length(d2);
        const glm::vec3 f1 = (params.gravity / dist1) * glm::normalize(d1);
        const glm::vec3 f2 = (params.gravity / dist2) * glm::normalize(d2);
        const glm::vec3 force = f1 + f2;

        const auto a = force * particleInvMass;
        p += v * params.deltaT + 0.5f * a * params.deltaT * params.deltaT;
        v += a * params.deltaT;

        if (dist1 > params.maxDistance || dist2 > params.maxDistance)
          p = glm::vec3(0.f);

        positions[idx] = p;
        velocities[idx] = v;
        distances[idx] = glm::length(p);
      });
}

} // namespace tsd::demo
