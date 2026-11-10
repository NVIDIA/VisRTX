// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#include "gpu/sampleLight.h"
#include "sobol/cudaResources.h"

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

using namespace visrtx;

VISRTX_DEVICE bool different(const vec3 &actual, const vec3 &expected)
{
  return !(glm::length(actual - expected) < 1e-6f);
}

VISRTX_GLOBAL void checkLightDraws(
    int *failures, cudaTextureObject_t texture, const float *cdf)
{
  int f = 0;
  RandState rs;
  pcg_init(&rs, 42, 7);
  auto expected = rs;
  const float u = pcg_uniform(&expected);
  const float v = pcg_uniform(&expected);
  LightGPUData ld{};
  ld.color = vec3(1.f);
  ld.rect.position = vec3(0, 0, 1);
  ld.rect.edge1 = vec3(2, 0, 0);
  ld.rect.edge2 = vec3(0, 3, 0);
  const auto rect = detail::sampleRectLight(ld, mat4(1), vec3(0), rs);
  const vec3 point = rect.dir * rect.dist;
  f += different(point, vec3(2.f * u, 3.f * v, 1.f));
  f += rs.state != expected.state;
  const auto qmcBefore = rs;
  const auto qmcRect =
      detail::sampleRectLight(ld, mat4(1), vec3(0), rs, true, 17, 123);
  const vec3 qmcPoint = qmcRect.dir * qmcRect.dist;
  f += different(qmcPoint,
      vec3(2.f * owenSobol(17, 2, 123), 3.f * owenSobol(17, 3, 123), 1.f));
  f += rs.state != qmcBefore.state;

  pcg_init(&rs, 42, 7);
  expected = rs;
  const float row = pcg_uniform(&expected);
  const float col = pcg_uniform(&expected);
  const float jitterU = pcg_uniform(&expected);
  const float jitterV = pcg_uniform(&expected);
  ld = {};
  ld.color = vec3(1.f);
  ld.hdri.size = uvec2(2, 2);
  ld.hdri.marginalCDF = cdf;
  ld.hdri.conditionalCDF = cdf + 2;
  ld.hdri.radiance = texture;
  ld.hdri.xfm = mat3(1);
  ld.hdri.scale = ld.hdri.pdfWeight = 1.f;
  const auto hdri = detail::sampleHDRILight(ld, mat4(1), rs);
  const float x = (col > .5f ? 1.f : 0.f) + jitterU;
  const float y = (row > .5f ? 1.f : 0.f) + jitterV;
  const auto expectedDir =
      sphericalCoordsToDirection(vec2(kPi * y / 2.f, kPi * x));
  f += different(hdri.dir, expectedDir);
  f += rs.state != expected.state;
  const auto beforeHdriQmc = rs;
  const auto qmcHdri = detail::sampleHDRILight(ld, mat4(1), rs, true, 17, 123);
  const float qmcX =
      (owenSobol(17, 3, 123) > .5f ? 1.f : 0.f) + owenSobol(17, 4, 123);
  const float qmcY =
      (owenSobol(17, 2, 123) > .5f ? 1.f : 0.f) + owenSobol(17, 5, 123);
  f += different(qmcHdri.dir,
      sphericalCoordsToDirection(vec2(kPi * qmcY / 2.f, kPi * qmcX)));
  f += rs.state != beforeHdriQmc.state;

  pcg_init(&rs, 42, 7);
  expected = rs;
  const float hemiU = pcg_uniform(&expected);
  const float hemiV = pcg_uniform(&expected);
  const auto hemisphere = sampleHemisphere(rs, vec3(0, 0, 1));
  const float radius = sqrtf(hemiU);
  const vec3 expectedHemi(radius * cosf(kTwoPi * hemiV),
      radius * sinf(kTwoPi * hemiV),
      sqrtf(1.f - hemiU));
  f += different(hemisphere, expectedHemi);
  f += rs.state != expected.state;
  *failures = f;
}

void cudaCheck(cudaError_t error, const char *operation)
{
  if (error != cudaSuccess) {
    throw std::runtime_error(
        std::string(operation) + ": " + cudaGetErrorString(error));
  }
}
int main()
try {
  CudaTexture image;
  const auto channel = cudaCreateChannelDesc<float4>();
  cudaCheck(
      cudaMallocArray(&image.array, &channel, 2, 2), "texture allocation");
  const float4 pixels[4] = {
      {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}};
  cudaCheck(cudaMemcpy2DToArray(image.array,
                0,
                0,
                pixels,
                2 * sizeof(float4),
                2 * sizeof(float4),
                2,
                cudaMemcpyHostToDevice),
      "texture upload");
  cudaResourceDesc resource{};
  resource.resType = cudaResourceTypeArray;
  resource.res.array.array = image.array;
  cudaTextureDesc description{};
  description.normalizedCoords = 1;
  description.readMode = cudaReadModeElementType;
  cudaCheck(
      cudaCreateTextureObject(&image.texture, &resource, &description, nullptr),
      "texture creation");
  CudaBuffer<float> cdf;
  const float cdfHost[] = {.5f, 1.f, .5f, 1.f, .5f, 1.f};
  cudaCheck(cdf.allocate(6), "CDF allocation");
  cudaCheck(
      cudaMemcpy(cdf.get(), cdfHost, sizeof(cdfHost), cudaMemcpyHostToDevice),
      "CDF upload");
  CudaBuffer<int> deviceFailures;
  cudaCheck(deviceFailures.allocate(1), "result allocation");
  checkLightDraws<<<1, 1>>>(deviceFailures.get(), image.texture, cdf.get());
  cudaCheck(cudaGetLastError(), "launch");
  cudaCheck(cudaDeviceSynchronize(), "synchronize");
  int failures;
  cudaCheck(
      cudaMemcpy(
          &failures, deviceFailures.get(), sizeof(int), cudaMemcpyDeviceToHost),
      "result download");
  if (failures)
    std::fprintf(stderr, "%d light draw-order failures\n", failures);
  return failures ? EXIT_FAILURE : EXIT_SUCCESS;
} catch (const std::exception &e) {
  std::fprintf(stderr, "%s\n", e.what());
  return EXIT_FAILURE;
}
