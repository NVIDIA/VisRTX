#pragma once

#include "gpu/gpu_objects.h"
#include "gpu/sampleLight.h"
#include "gpu/shading_api.h"

#include <optix.h>
#include <cstdint>

namespace visrtx {

VISRTX_DEVICE vec4 shadeMatteSurface(const FrameGPUData &fd,
    const MaterialGPUData::Matte &md,
    const Ray &ray,
    const SurfaceHit &hit,
    const LightSample &ls)
{
  auto viewDir = -ray.dir;

  // Call signature must match the actual implementation in MatteShader_ptx.cu
  return optixDirectCall<vec4>(static_cast<unsigned int>(MaterialType::MATTE),
      &fd,
      &md,
      &hit,
      &viewDir,
      &ls.dir,
      &ls.radiance);
}

} // namespace visrtx
