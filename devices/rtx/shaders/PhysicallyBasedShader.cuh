#pragma once

#include "gpu/gpu_objects.h"
#include "gpu/sampleLight.h"
#include "gpu/shadingState.h"

#include <optix.h>

namespace visrtx {

VISRTX_DEVICE vec3 physicallyBasedShadeSurface(
    const PhysicallyBasedShadingState &shadingState,
    const SurfaceHit& hit,
    const LightSample& lightSample,
    const vec3& outgoingDir)
{
  // Call signature must match the actual implementation in
  // PhysicallyBasedShader_ptx.cu
  return optixDirectCall<vec3>(
      static_cast<unsigned int>(MaterialType::PHYSICALLYBASED),
      &shadingState,
      &hit,
      &lightSample,
      &outgoingDir);
}

} // namespace visrtx
