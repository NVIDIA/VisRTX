// Copyright 2024 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/objects/Sampler.hpp"

namespace tsd {

Sampler::Sampler(Token subtype) : Object(ANARI_SAMPLER, subtype)
{
  if (subtype == tokens::sampler::image1D || subtype == tokens::sampler::image2D
      || subtype == tokens::sampler::image3D) {
    addParameter("inAttribute"_t, "attribute0")
        .setStringValues(
            {"attribute0", "attribute1", "attribute2", "attribute3", "color"});
    addParameter("filter"_t, "linear").setStringValues({"linear", "nearest"});
    if (subtype == tokens::sampler::image1D) {
      addParameter("wrapMode"_t, "clampToEdge")
          .setStringValues({"clampToEdge", "repeat", "mirrorRepeat"});
    } else if (subtype == tokens::sampler::image2D) {
      addParameter("wrapMode1"_t, "clampToEdge")
          .setStringValues({"clampToEdge", "repeat", "mirrorRepeat"});
      addParameter("wrapMode2"_t, "clampToEdge")
          .setStringValues({"clampToEdge", "repeat", "mirrorRepeat"});
    } else if (subtype == tokens::sampler::image3D) {
      addParameter("wrapMode1"_t, "clampToEdge")
          .setStringValues({"clampToEdge", "repeat", "mirrorRepeat"});
      addParameter("wrapMode2"_t, "clampToEdge")
          .setStringValues({"clampToEdge", "repeat", "mirrorRepeat"});
      addParameter("wrapMode3"_t, "clampToEdge")
          .setStringValues({"clampToEdge", "repeat", "mirrorRepeat"});
    }
    addParameter("inTransform"_t, math::scaling_matrix(float3(1.f)));
    addParameter("inOffset"_t, float4(0.f));
    addParameter("outTransform"_t, math::scaling_matrix(float3(1.f)));
    addParameter("outOffset"_t, float4(0.f));
  } else if (subtype == tokens::sampler::transform) {
    addParameter("inAttribute"_t, "attribute0")
        .setStringValues(
            {"attribute0", "attribute1", "attribute2", "attribute3", "color"});
    addParameter("outTransform"_t, math::scaling_matrix(float3(1.f)));
    addParameter("outOffset"_t, float4(0.f));
  }
}

anari::Object Sampler::makeANARIObject(anari::Device d) const
{
  return anari::newObject<anari::Sampler>(d, subtype().c_str());
}

namespace tokens::sampler {

const Token image1D = "image1D";
const Token image2D = "image2D";
const Token image3D = "image3D";
const Token primitive = "primitive";
const Token transform = "transform";

} // namespace tokens::sampler

} // namespace tsd
