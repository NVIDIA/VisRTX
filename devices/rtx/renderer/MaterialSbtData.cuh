#pragma once

#include "shaders/MDLShaderData.cuh"

namespace visrtx {

struct MaterialSbtData
{
  union
  {
    struct
    {
      const MDLMaterialData *materialData;
    } mdl;
  };
};

} // namespace visrtx