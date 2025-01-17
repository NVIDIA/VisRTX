#include "MDLTexture.cuh"

#include "gpu/evalMaterial.h"
#include "gpu/gpu_decl.h"
#include "gpu/gpu_objects.h"
#include "renderer/MaterialSbtData.cuh"

#include <curand.h>

#include "gpu/vector.cuh"

#include <anari/anari_cpp/ext/linalg.h>
#include <mi/neuraylib/target_code_types.h>
#include <optix_device.h>

// No derivatives yet
using MaterialExprFunc = mi::neuraylib::Material_expr_function;
using BsdfInitFunc = mi::neuraylib::Bsdf_init_function;
using BsdfSampleFunc = mi::neuraylib::Bsdf_sample_function;
using BsdfEvaluateFunc = mi::neuraylib::Bsdf_evaluate_function;
using BsdfPdfFunc = mi::neuraylib::Bsdf_pdf_function;

using ShadingStateMaterial = mi::neuraylib::Shading_state_material;
using ResourceData = mi::neuraylib::Resource_data;

using BsdfSampleData = mi::neuraylib::Bsdf_sample_data;
using BsdfEvaluateData =
    mi::neuraylib::Bsdf_evaluate_data<mi::neuraylib::DF_HSM_NONE>;
using BsdfPdfData = mi::neuraylib::Bsdf_pdf_data;

using BsdfIsThinWalled = bool(
    const ShadingStateMaterial *, const ResourceData *, const char *);

VISRTX_CALLABLE BsdfInitFunc mdlBsdf_init;
VISRTX_CALLABLE BsdfSampleFunc mdlBsdf_sample;
VISRTX_CALLABLE BsdfEvaluateFunc mdlBsdf_evaluate;
VISRTX_CALLABLE BsdfPdfFunc mdlBsdf_pdf;
VISRTX_CALLABLE BsdfIsThinWalled mdl_isThinWalled;

namespace visrtx {

VISRTX_DEVICE float4 make_float4(float f)
{
  return ::make_float4(f, f, f, f);
}

VISRTX_DEVICE float4 make_float4(const vec4 &v)
{
  return ::make_float4(v.x, v.y, v.z, v.w);
}

VISRTX_DEVICE float3 make_float3(float f)
{
  return ::make_float3(f, f, f);
}

VISRTX_DEVICE float3 make_float3(const vec4 &v)
{
  return ::make_float3(v.x, v.y, v.z);
}

VISRTX_DEVICE float3 make_float3(const vec3 &v)
{
  return ::make_float3(v.x, v.y, v.z);
}

// Signature must match the call inside shaderMDLSurface in MDLShader.cuh.
VISRTX_CALLABLE
vec4 __direct_callable__evalSurfaceMaterial(const FrameGPUData *fd,
    const ScreenSample *ss,
    const MaterialGPUData::MDL *md,
    const Ray *ray,
    const SurfaceHit *hit,
    const LightSample *ls)
{
  // Create MDL state
  ShadingStateMaterial state;
  auto position = make_float3(hit->hitpoint);
  auto Ns = make_float3(hit->Ns);
  auto Ng = make_float3(hit->Ng);

  auto objectToWorld =
      bit_cast<const std::array<float4, 3>>(hit->objectToWorld);
  auto worldToObject =
      bit_cast<const std::array<float4, 3>>(hit->worldToObject);
  // The number of texture spaces we support. Matching the number of attributes
  // ANARI exposes (4)
  auto textureResults = std::array<float4,
      32>{}; // The maximum number of samplers we support. See MDLCompiler.cpp
             // numTextureSpaces and numTextureResults.
  auto textureCoords = std::array{
      make_float3(readAttributeValue(0, *hit)),
      make_float3(readAttributeValue(1, *hit)),
      make_float3(readAttributeValue(2, *hit)),
      make_float3(readAttributeValue(3, *hit)),
  };
  auto textureTangentsU = std::array{
      float3{1.0f, 0.0f, 0.0f},
      float3{1.0f, 0.0f, 0.0f},
      float3{1.0f, 0.0f, 0.0f},
      float3{1.0f, 0.0f, 0.0f},
  };
  auto textureTangentsV = std::array{
      float3{0.0f, 1.0f, 0.0f},
      float3{0.0f, 1.0f, 0.0f},
      float3{0.0f, 1.0f, 0.0f},
      float3{0.0f, 1.0f, 0.0f},
  };

  state.animation_time = 0.0f;
  state.geom_normal = Ng;
  state.normal = Ns;
  state.position = position;
  state.meters_per_scene_unit = 1.0f;
  state.object_id = hit->objID;
  state.object_to_world = data(objectToWorld);
  state.world_to_object = data(worldToObject);
  state.ro_data_segment = nullptr;
  state.text_coords = data(textureCoords);
  state.text_results = data(textureResults);
  state.tangent_u = data(textureTangentsU);
  state.tangent_v = data(textureTangentsV);

  // Resources shared by all mdl calls.
  TextureHandler texHandler{};
  texHandler.fd = fd;
  texHandler.ss = ss;
  memcpy(texHandler.samplers, md->samplers, sizeof(md->samplers));
  texHandler.numSamplers = md->numSamplers;

  ResourceData resData = {nullptr, &texHandler};

  // Argument block
  auto argblock = md->argBlock;

  // Init
  mdlBsdf_init(&state, &resData, argblock); // Should be factored out

  // Eval
  BsdfEvaluateData eval_data = {};
  const float cos_theta = dot(make_float3(-ray->dir), normalize(state.normal));
  if (cos_theta > 0.0f) {
    float3 radiance_over_pdf = bit_cast<float3>(ls->radiance / ls->pdf);
    eval_data.ior1 = make_float3(1.0f);
    eval_data.ior2.x = MI_NEURAYLIB_BSDF_USE_MATERIAL_IOR;

    eval_data.k1 = make_float3(-ray->dir);
    eval_data.k2 = make_float3(normalize(ls->dir));
    eval_data.bsdf_diffuse = make_float3(0.0f);
    eval_data.bsdf_glossy = make_float3(0.0f);

    mdlBsdf_evaluate(&eval_data, &state, &resData, argblock);

    float3 contrib =
        radiance_over_pdf * (eval_data.bsdf_diffuse + eval_data.bsdf_glossy);
    return vec4(contrib.x, contrib.y, contrib.z, 1.0f);
  }

  return vec4(0.0f, 0.0f, 0.0f, 1.0f);
}

} // namespace visrtx
