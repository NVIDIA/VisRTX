// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/core/scene/objects/Sampler.hpp"

namespace tsd::core {

Sampler::Sampler(Token subtype) : Object(ANARI_SAMPLER, subtype)
{
  if (subtype == tokens::sampler::compressedImage2D
      || subtype == tokens::sampler::image1D
      || subtype == tokens::sampler::image2D
      || subtype == tokens::sampler::image3D) {
    addParameter("inAttribute")
        .setValue("attribute0")
        .setStringValues(
            {"attribute0", "attribute1", "attribute2", "attribute3", "color"});
    addParameter("filter").setValue("linear").setStringValues(
        {"linear", "nearest"});
    if (subtype == tokens::sampler::image1D) {
      addParameter("wrapMode")
          .setValue("clampToEdge")
          .setStringValues({"clampToEdge", "repeat", "mirrorRepeat"});
    } else if (subtype == tokens::sampler::compressedImage2D
        || subtype == tokens::sampler::image2D) {
      addParameter("wrapMode1")
          .setValue("clampToEdge")
          .setStringValues({"clampToEdge", "repeat", "mirrorRepeat"});
      addParameter("wrapMode2")
          .setValue("clampToEdge")
          .setStringValues({"clampToEdge", "repeat", "mirrorRepeat"});
    } else if (subtype == tokens::sampler::image3D) {
      addParameter("wrapMode1")
          .setValue("clampToEdge")
          .setStringValues({"clampToEdge", "repeat", "mirrorRepeat"});
      addParameter("wrapMode2")
          .setValue("clampToEdge")
          .setStringValues({"clampToEdge", "repeat", "mirrorRepeat"});
      addParameter("wrapMode3")
          .setValue("clampToEdge")
          .setStringValues({"clampToEdge", "repeat", "mirrorRepeat"});
    }
    addParameter("inTransform").setValue(math::scaling_matrix(float3(1.f)));
    addParameter("inOffset").setValue(float4(0.f));
    addParameter("outTransform").setValue(math::scaling_matrix(float3(1.f)));
    addParameter("outOffset").setValue(float4(0.f));
  } else if (subtype == tokens::sampler::transform) {
    addParameter("inAttribute")
        .setValue("attribute0")
        .setStringValues(
            {"attribute0", "attribute1", "attribute2", "attribute3", "color"});
    addParameter("outTransform").setValue(math::scaling_matrix(float3(1.f)));
    addParameter("outOffset").setValue(float4(0.f));
  }
}

anari::Object Sampler::makeANARIObject(anari::Device d) const
{
  return anari::newObject<anari::Sampler>(d, subtype().c_str());
}

namespace tokens::sampler {

const Token compressedImage2D = "compressedImage2D";
const Token image1D = "image1D";
const Token image2D = "image2D";
const Token image3D = "image3D";
const Token primitive = "primitive";
const Token transform = "transform";

} // namespace tokens::sampler

} // namespace tsd::core
