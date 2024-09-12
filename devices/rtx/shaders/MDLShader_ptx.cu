#include "gpu/gpu_decl.h"
#include "gpu/gpu_objects.h"
#include "gpu/intersectRay.h"
#include "gpu/shading_api.h"
#include "renderer/MaterialSbtData.cuh"
#include "scene/World.h"

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

// Signature must match the call inside shaderMDLSurface in MDLShader.cuh.
VISRTX_CALLABLE
vec4 __direct_callable__evalSurfaceMaterial(const FrameGPUData *fd,
    const ScreenSample *ss,
    const MaterialGPUData::MDL *md,
    const Ray *ray,
    const SurfaceHit *hit,
    const LightSample *ls)
{
  auto materialSbtData =
      bit_cast<const MaterialSbtData *>(optixGetSbtDataPointer());
  auto materialData =
      materialSbtData ? materialSbtData->mdl.materialData : nullptr;

  float4 texture_results[2]; // must match num_texture_results parameter given
                             // to Mdl_helper

  // Create MDL state
  ShadingStateMaterial state;
  auto position = bit_cast<float3>(hit->hitpoint);
  auto Ns = bit_cast<float3>(hit->Ns);
  auto Ng = bit_cast<float3>(hit->Ng);
  auto T = bit_cast<float3>(hit->T);
  auto B = bit_cast<float3>(hit->B);
  auto objectToWorld =
      bit_cast<const std::array<float4, 3>>(hit->objectToWorld);
  auto worldToObject =
      bit_cast<const std::array<float4, 3>>(hit->worldToObject);
  auto textCoords = bit_cast<float3>(hit->uvw);

  state.animation_time = 0.0f;
  state.geom_normal = Ng;
  state.normal = Ns;
  state.position = position;
  state.meters_per_scene_unit = 1.0f;
  state.object_id = hit->objID;
  state.object_to_world = objectToWorld.data();
  state.world_to_object = worldToObject.data();
  state.ro_data_segment = nullptr;
  state.text_coords = &textCoords;
  state.text_results = texture_results;
  state.tangent_u = &T;
  state.tangent_v = &B;

  // Resources shared by all mdl calls.
  ResourceData resData = {nullptr, nullptr};
  auto argblock = materialData ? *materialData : nullptr;

  // Init
  mdlBsdf_init(&state, &resData, argblock); // Should be factored out

  // Eval
  BsdfEvaluateData eval_data = {};
  const float cos_theta =
      dot(bit_cast<float3>(-ray->dir), normalize(state.normal));
  if (cos_theta > 0.0f) {
    float3 radiance_over_pdf = bit_cast<float3>(ls->radiance / ls->pdf);
    eval_data.ior1 = make_float3(1.0f, 1.0f, 1.0f);
    eval_data.ior2.x = MI_NEURAYLIB_BSDF_USE_MATERIAL_IOR;

    eval_data.k1 = bit_cast<float3>(-ray->dir);
    eval_data.k2 = bit_cast<float3>(normalize(ls->dir));
    eval_data.bsdf_diffuse = make_float3(0.0f);
    eval_data.bsdf_glossy = make_float3(0.0f);

    mdlBsdf_evaluate(&eval_data, &state, &resData, argblock);

    float3 contrib =
        radiance_over_pdf * (eval_data.bsdf_diffuse + eval_data.bsdf_glossy);
    return vec4(bit_cast<vec3>(contrib), 1.0f);
  }

  return vec4(1.0f, 0.0f, 1.0f, 1.0f);
}

} // namespace visrtx
