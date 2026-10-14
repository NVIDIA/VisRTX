/*
 * Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include <anari/anari_cpp/ext/std.h>
#include <anari/ext/visrtx/makeVisRTXDevice.h>
#include <anari/anari_cpp.hpp>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using uvec2 = std::array<unsigned int, 2>;
using vec3 = std::array<float, 3>;
using vec4 = std::array<float, 4>;

static constexpr uvec2 IMAGE_SIZE = {32, 32};
static constexpr vec3 GREEN = {0.f, 1.f, 0.f};
static constexpr vec3 RED = {1.f, 0.f, 0.f};
static constexpr vec3 BLUE = {0.f, 0.f, 1.f};
static constexpr vec3 MID = {0.2f, 0.4f, 0.7f};
static constexpr float COLOR_TOLERANCE = 0.05f;

enum class Ownership
{
  MANAGED,
  SHARED
};
enum class TexFormat
{
  SRGB_RGBA,
  SRGB_RGB,
  FLOAT_RGBA,
  FLOAT_RGB
};

static bool expectUploadError = false;
static unsigned uploadErrors = 0;

static void statusFunc(const void *,
    ANARIDevice,
    ANARIObject source,
    ANARIDataType,
    ANARIStatusSeverity severity,
    ANARIStatusCode,
    const char *message)
{
  if (severity == ANARI_SEVERITY_ERROR && expectUploadError
      && std::strstr(message, "CUDA array upload failed")) {
    ++uploadErrors;
    return;
  }
  if (severity == ANARI_SEVERITY_FATAL_ERROR
      || severity == ANARI_SEVERITY_ERROR) {
    fprintf(stderr, "FAIL[ANARI][%p] %s\n", source, message);
    std::exit(1);
  }
}

static ANARIDataType anariFormat(TexFormat format)
{
  switch (format) {
  case TexFormat::SRGB_RGBA:
    return ANARI_UFIXED8_RGBA_SRGB;
  case TexFormat::SRGB_RGB:
    return ANARI_UFIXED8_RGB_SRGB;
  case TexFormat::FLOAT_RGBA:
    return ANARI_FLOAT32_VEC4;
  case TexFormat::FLOAT_RGB:
    return ANARI_FLOAT32_VEC3;
  }
  std::abort();
}

static bool isFloat(TexFormat format)
{
  return format == TexFormat::FLOAT_RGBA || format == TexFormat::FLOAT_RGB;
}

static size_t channels(TexFormat format)
{
  return format == TexFormat::SRGB_RGB || format == TexFormat::FLOAT_RGB ? 3
                                                                         : 4;
}

static std::string formatName(TexFormat format)
{
  return std::string(isFloat(format) ? "float/" : "srgb/")
      + (channels(format) == 3 ? "rgb" : "rgba");
}

static uint8_t encodeSrgb(float linear)
{
  return uint8_t(std::lround(255.f
      * (linear <= 0.0031308f
              ? linear * 12.92f
              : 1.055f * std::pow(linear, 1.f / 2.4f) - 0.055f)));
}

static void writeTexels(
    void *dst, TexFormat format, const std::vector<vec3> &colors)
{
  if (!dst) {
    fprintf(stderr, "FAIL: array mapping returned null\n");
    std::exit(1);
  }
  auto *bytes = static_cast<uint8_t *>(dst);
  for (size_t i = 0; i < colors.size(); ++i) {
    for (size_t c = 0; c < channels(format); ++c) {
      const float value = c == 3 ? 1.f : colors[i][c];
      const size_t offset = i * channels(format) + c;
      if (isFloat(format))
        std::memcpy(bytes + offset * sizeof(float), &value, sizeof(value));
      else
        bytes[offset] = encodeSrgb(value);
    }
  }
}

static ANARIArray newImage(anari::Device d,
    int dim,
    ANARIDataType type,
    size_t width,
    const void *memory = nullptr)
{
  if (dim == 1)
    return anariNewArray1D(d, memory, nullptr, nullptr, type, width);
  if (dim == 2)
    return anariNewArray2D(d, memory, nullptr, nullptr, type, width, 2);
  return anariNewArray3D(d, memory, nullptr, nullptr, type, width, 2, 2);
}

static void setImage(anari::Device d,
    ANARIObject object,
    const char *name,
    ANARIArray image,
    int dim)
{
  const auto type = dim == 1 ? ANARI_ARRAY1D
      : dim == 2             ? ANARI_ARRAY2D
                             : ANARI_ARRAY3D;
  anariSetParameter(d, object, name, type, &image);
  anariCommitParameters(d, object);
}

static void rewriteImage(anari::Device d,
    ANARIArray image,
    TexFormat format,
    const std::vector<vec3> &colors)
{
  writeTexels(anariMapArray(d, image), format, colors);
  anariUnmapArray(d, image);
}

static anari::Surface makeQuad(anari::Device d, anari::Sampler sampler, float x)
{
  const std::array<vec3, 4> pos = {vec3{x - 2.f, -2.f, 0.f},
      vec3{x + 2.f, -2.f, 0.f},
      vec3{x + 2.f, 2.f, 0.f},
      vec3{x - 2.f, 2.f, 0.f}};
  const std::array<std::array<unsigned, 3>, 2> idx = {
      std::array<unsigned, 3>{0, 1, 2}, std::array<unsigned, 3>{0, 2, 3}};
  const std::array<vec3, 4> uv = {};
  auto geom = anari::newObject<anari::Geometry>(d, "triangle");
  anari::setParameterArray1D(
      d, geom, "vertex.position", pos.data(), pos.size());
  anari::setParameterArray1D(
      d, geom, "primitive.index", idx.data(), idx.size());
  anari::setParameterArray1D(
      d, geom, "vertex.attribute0", uv.data(), uv.size());
  anari::commitParameters(d, geom);
  auto mat = anari::newObject<anari::Material>(d, "matte");
  anari::setParameter(d, mat, "color", sampler);
  anari::setParameter(d, mat, "alphaMode", "opaque");
  anari::commitParameters(d, mat);
  auto surface = anari::newObject<anari::Surface>(d);
  anari::setAndReleaseParameter(d, surface, "geometry", geom);
  anari::setAndReleaseParameter(d, surface, "material", mat);
  anari::commitParameters(d, surface);
  return surface;
}

struct TexturedScene
{
  TexturedScene(anari::Device d,
      int dim,
      TexFormat format,
      Ownership ownership = Ownership::MANAGED,
      int numSamplers = 1,
      size_t width = 2);
  ~TexturedScene();
  TexturedScene(const TexturedScene &) = delete;
  TexturedScene &operator=(const TexturedScene &) = delete;

  anari::Device device;
  int dimension;
  TexFormat format;
  size_t texelCount;
  std::vector<float> appMemory;
  ANARIArray image{};
  std::vector<anari::Sampler> samplers;
  anari::World world{};
  anari::Camera camera{};
  anari::Renderer renderer{};
  anari::Frame frame{};
  int viewedSampler = 0;
};

static anari::Frame makeFrame(const TexturedScene &s)
{
  auto frame = anari::newObject<anari::Frame>(s.device);
  anari::setParameter(s.device, frame, "size", IMAGE_SIZE);
  anari::setParameter(s.device, frame, "channel.color", ANARI_FLOAT32_VEC4);
  anari::setParameter(s.device, frame, "world", s.world);
  anari::setParameter(s.device, frame, "camera", s.camera);
  anari::setParameter(s.device, frame, "renderer", s.renderer);
  anari::commitParameters(s.device, frame);
  return frame;
}

TexturedScene::TexturedScene(anari::Device d,
    int dim,
    TexFormat fmt,
    Ownership ownership,
    int numSamplers,
    size_t width)
    : device(d), dimension(dim), format(fmt), texelCount(width << (dim - 1))
{
  if (ownership == Ownership::SHARED) {
    const size_t bytes =
        texelCount * channels(format) * (isFloat(format) ? 4 : 1);
    appMemory.resize((bytes + sizeof(float) - 1) / sizeof(float));
    writeTexels(appMemory.data(), format, std::vector<vec3>(texelCount, GREEN));
    image = newImage(d, dim, anariFormat(format), width, appMemory.data());
  } else {
    image = newImage(d, dim, anariFormat(format), width);
    rewriteImage(d, image, format, std::vector<vec3>(texelCount, GREEN));
  }
  world = anari::newObject<anari::World>(d);
  std::vector<anari::Surface> surfaces;
  for (int i = 0; i < numSamplers; ++i) {
    const std::string subtype = "image" + std::to_string(dim) + "D";
    auto sampler = anari::newObject<anari::Sampler>(d, subtype.c_str());
    anari::setParameter(d, sampler, "inAttribute", "attribute0");
    anari::setParameter(d, sampler, "inOffset", vec4{0.75f, 0.5f, 0.5f, 0.f});
    anari::setParameter(d, sampler, "filter", "nearest");
    setImage(d, sampler, "image", image, dim);
    samplers.push_back(sampler);
    surfaces.push_back(makeQuad(d, sampler, 5.f * i));
  }
  if (!surfaces.empty())
    anari::setParameterArray1D(
        d, world, "surface", surfaces.data(), surfaces.size());
  for (auto surface : surfaces)
    anari::release(d, surface);
  anari::commitParameters(d, world);
  // Parallel rays remove the debug baseColor renderer's angle falloff.
  camera = anari::newObject<anari::Camera>(d, "orthographic");
  anari::setParameter(d, camera, "position", vec3{0.f, 0.f, -3.f});
  anari::setParameter(d, camera, "direction", vec3{0.f, 0.f, 1.f});
  anari::setParameter(d, camera, "up", vec3{0.f, 1.f, 0.f});
  anari::setParameter(d, camera, "height", 2.f);
  anari::setParameter(d, camera, "aspect", 1.f);
  anari::commitParameters(d, camera);
  renderer = anari::newObject<anari::Renderer>(d, "debug");
  anari::setParameter(d, renderer, "method", "baseColor");
  anari::setParameter(d, renderer, "background", vec4{0.f, 0.f, 0.f, 1.f});
  anari::setParameter(d, renderer, "ambientColor", vec3{1.f, 1.f, 1.f});
  anari::commitParameters(d, renderer);
  frame = makeFrame(*this);
}

TexturedScene::~TexturedScene()
{
  anari::release(device, frame);
  anari::release(device, world);
  anari::release(device, camera);
  anari::release(device, renderer);
  for (auto sampler : samplers)
    anari::release(device, sampler);
  if (image)
    anariRelease(device, image);
}

static vec3 renderMeanColor(TexturedScene &s, int samplerIndex = 0)
{
  if (s.viewedSampler != samplerIndex) {
    anari::setParameter(
        s.device, s.camera, "position", vec3{5.f * samplerIndex, 0.f, -3.f});
    anari::commitParameters(s.device, s.camera);
    s.viewedSampler = samplerIndex;
  }
  anari::render(s.device, s.frame);
  anari::wait(s.device, s.frame);
  uint32_t width = 0, height = 0;
  ANARIDataType type = ANARI_UNKNOWN;
  auto *pixels = static_cast<const vec4 *>(anariMapFrame(
      s.device, s.frame, "channel.color", &width, &height, &type));
  if (!pixels || width != IMAGE_SIZE[0] || height != IMAGE_SIZE[1]
      || type != ANARI_FLOAT32_VEC4) {
    fprintf(stderr, "FAIL: invalid framebuffer mapping\n");
    std::exit(1);
  }
  std::array<double, 3> sum{};
  for (size_t i = 0; i < size_t(width) * height; ++i) {
    for (float value : pixels[i]) {
      if (!std::isfinite(value)) {
        fprintf(stderr, "FAIL: non-finite framebuffer component\n");
        std::exit(1);
      }
    }
    for (int c = 0; c < 3; ++c)
      sum[c] += pixels[i][c];
  }
  anari::unmap(s.device, s.frame, "channel.color");
  return {float(sum[0] / (width * height)),
      float(sum[1] / (width * height)),
      float(sum[2] / (width * height))};
}

static void check(const std::string &label,
    const vec3 &actual,
    const vec3 &expected,
    bool &ok)
{
  bool pass = true;
  for (int c = 0; c < 3; ++c)
    pass &= std::isfinite(actual[c])
        && std::abs(actual[c] - expected[c]) <= COLOR_TOLERANCE;
  printf("%s got=(%.3f %.3f %.3f) want=(%.3f %.3f %.3f) %s\n",
      label.c_str(),
      actual[0],
      actual[1],
      actual[2],
      expected[0],
      expected[1],
      expected[2],
      pass ? "ok" : "FAIL");
  ok &= pass;
}

static void checkAll(
    TexturedScene &s, const std::string &label, const vec3 &expected, bool &ok)
{
  for (size_t i = 0; i < s.samplers.size(); ++i)
    check(label + "/sampler" + std::to_string(i),
        renderMeanColor(s, int(i)),
        expected,
        ok);
}

static void testWindows(
    anari::Device d, TexFormat format, Ownership ownership, bool &ok)
{
  TexturedScene s(d, 1, format, ownership, 3, 8);
  const std::vector<vec3> colors = {GREEN,
      RED,
      BLUE,
      {0.f, 1.f, 1.f},
      {1.f, 0.f, 1.f},
      {1.f, 1.f, 0.f},
      MID,
      {0.8f, 0.1f, 0.3f}};
  rewriteImage(d, s.image, format, colors);
  const std::string prefix = "window/" + formatName(format)
      + (ownership == Ownership::MANAGED ? "/managed" : "/shared");
  struct Window
  {
    uint64_t begin, end;
    size_t expectedIndex;
    const char *name;
  };
  // At normalized x=.75, nearest samples floor(.75 * (end-begin)) + begin.
  const Window windows[] = {{1, 5, 4, "nonzero-begin"},
      {2, 6, 5, "equal-length-shift"},
      {3, 5, 4, "shrink"},
      {1, 8, 6, "growth"}};
  for (const auto &window : windows) {
    anari::setParameter(d, s.image, "begin", window.begin);
    anari::setParameter(d, s.image, "end", window.end);
    anari::commitParameters(d, s.image);
    checkAll(s, prefix + "/" + window.name, colors[window.expectedIndex], ok);
  }
}

static void testRepeatedWrites(anari::Device d,
    int dim,
    TexFormat format,
    Ownership ownership,
    int numSamplers,
    bool &ok)
{
  TexturedScene s(d, dim, format, ownership, numSamplers);
  const std::string prefix = "repeated/image" + std::to_string(dim) + "D/"
      + formatName(format)
      + (ownership == Ownership::MANAGED ? "/managed/" : "/shared/")
      + std::to_string(numSamplers);
  checkAll(s, prefix + "/initial", GREEN, ok);
  for (const auto &color : {RED, GREEN, BLUE, MID}) {
    rewriteImage(d, s.image, format, std::vector<vec3>(s.texelCount, color));
    checkAll(s, prefix + "/write", color, ok);
  }
  // A new frame must see the update too, without recommitting the samplers.
  rewriteImage(d, s.image, format, std::vector<vec3>(s.texelCount, RED));
  anari::release(d, s.frame);
  s.frame = makeFrame(s);
  checkAll(s, prefix + "/fresh-frame", RED, ok);
}

static void testReplacement(anari::Device d, int dim, int numSamplers, bool &ok)
{
  TexturedScene s(
      d, dim, TexFormat::SRGB_RGBA, Ownership::MANAGED, numSamplers);
  const std::string prefix = "replacement/image" + std::to_string(dim) + "D/"
      + std::to_string(numSamplers);
  checkAll(s, prefix + "/initial", GREEN, ok);
  auto next = newImage(d, dim, anariFormat(s.format), 2);
  rewriteImage(d, next, s.format, std::vector<vec3>(s.texelCount, BLUE));
  for (auto sampler : s.samplers)
    setImage(d, sampler, "image", next, dim);
  // Drop the application reference before any render can finalize the
  // rebinding.
  anariRelease(d, s.image);
  s.image = next;
  checkAll(s, prefix + "/release-before-finalize", BLUE, ok);
}

static void checkValid(anari::Device d,
    anari::Sampler sampler,
    const std::string &label,
    bool expected,
    bool &ok)
{
  bool valid = !expected;
  const bool found = anari::getProperty(d, sampler, "valid", valid, ANARI_WAIT);
  const bool pass = found && valid == expected;
  printf("%s/valid got=%d want=%d property=%d %s\n",
      label.c_str(),
      int(valid),
      int(expected),
      int(found),
      pass ? "ok" : "FAIL");
  ok &= pass;
}

static void testInvalidTransitions(
    anari::Device d, int dim, bool unset, bool &ok)
{
  TexturedScene s(d, dim, TexFormat::SRGB_RGBA, Ownership::MANAGED, 3);
  const std::string prefix = "invalid/image" + std::to_string(dim) + "D/"
      + (unset ? "unset" : "unsupported");
  checkAll(s, prefix + "/initial", GREEN, ok);
  ANARIArray bad = unset ? nullptr : newImage(d, dim, ANARI_INT32, 2);
  if (bad) {
    std::memset(anariMapArray(d, bad), 0, s.texelCount * sizeof(int32_t));
    anariUnmapArray(d, bad);
  }
  // Invalidate one consumer at a time; peers must still sample the shared
  // image.
  for (size_t i = 0; i < s.samplers.size(); ++i) {
    if (unset) {
      anari::unsetParameter(d, s.samplers[i], "image");
      anari::commitParameters(d, s.samplers[i]);
    } else
      setImage(d, s.samplers[i], "image", bad, dim);
    if (i + 1 == s.samplers.size()) {
      anariRelease(d, s.image);
      s.image = nullptr;
    }
    for (size_t j = 0; j < s.samplers.size(); ++j) {
      const auto label =
          prefix + "/step" + std::to_string(i) + "/sampler" + std::to_string(j);
      // Keep the material-bound surface visible. Invalid appearance is
      // unspecified, but rendering must succeed with finite pixels and no ERROR
      // diagnostics.
      const auto color = renderMeanColor(s, int(j));
      checkValid(d, s.samplers[j], label, j > i, ok);
      if (j > i)
        check(label + "/unaffected-peer", color, GREEN, ok);
    }
  }
  s.image = newImage(d, dim, anariFormat(s.format), 2);
  rewriteImage(d, s.image, s.format, std::vector<vec3>(s.texelCount, BLUE));
  for (auto sampler : s.samplers)
    setImage(d, sampler, "image", s.image, dim);
  checkAll(s, prefix + "/recovery", BLUE, ok);
  for (auto sampler : s.samplers)
    checkValid(d, sampler, prefix + "/recovery", true, ok);
  if (bad)
    anariRelease(d, bad);
}

static void testBackground(anari::Device d, bool hdri, bool &ok)
{
  const auto format = hdri ? TexFormat::FLOAT_RGB : TexFormat::SRGB_RGBA;
  TexturedScene s(d, 2, format, Ownership::MANAGED, 0);
  const std::string prefix = hdri ? "hdri" : "background";
  auto light = hdri ? anari::newObject<anari::Light>(d, "hdri") : nullptr;
  ANARIObject consumer = hdri ? ANARIObject(light) : ANARIObject(s.renderer);
  const char *parameter = hdri ? "radiance" : "background";
  if (hdri) {
    anari::setParameterArray1D(d, s.world, "light", &light, 1);
    anari::commitParameters(d, s.world);
  }
  setImage(d, consumer, parameter, s.image, 2);
  check(prefix + "/initial", renderMeanColor(s), GREEN, ok);
  for (const auto &color : {RED, MID}) {
    rewriteImage(d, s.image, format, std::vector<vec3>(s.texelCount, color));
    check(prefix + "/write", renderMeanColor(s), color, ok);
  }
  auto next = newImage(d, 2, anariFormat(format), 2);
  rewriteImage(d, next, format, std::vector<vec3>(s.texelCount, BLUE));
  setImage(d, consumer, parameter, next, 2);
  anariRelease(d, s.image);
  s.image = next;
  check(prefix + "/replacement", renderMeanColor(s), BLUE, ok);
  anariUnsetParameter(d, consumer, parameter);
  anariCommitParameters(d, consumer);
  anariRelease(d, s.image);
  s.image = nullptr;
  check(prefix + "/unset", renderMeanColor(s), vec3{0.f, 0.f, 0.f}, ok);
  if (light)
    anari::release(d, light);
}

static void testAllocationFailure(anari::Device d, bool &ok)
{
  // 1D storage is a height-1 2D cudaArray, whose width limit is 2^17 on every
  // CUDA GPU to date; 2^20 exceeds it with only 4 MiB of host data.
  constexpr size_t WIDTH = 1 << 20;
  TexturedScene s(d, 1, TexFormat::SRGB_RGBA, Ownership::MANAGED, 3, WIDTH);
  anari::setParameter(d, s.image, "end", uint64_t(2));
  anari::commitParameters(d, s.image);
  checkAll(s, "allocation-failure/initial-small-window", GREEN, ok);

  anari::setParameter(d, s.image, "end", uint64_t(WIDTH));
  anari::commitParameters(d, s.image);
  const auto previousErrors = uploadErrors;
  expectUploadError = true;
  renderMeanColor(s);
  expectUploadError = false;
  if (uploadErrors == previousErrors) {
    fprintf(
        stderr, "FAIL: oversized CUDA texture did not report upload failure\n");
    ok = false;
  }
  for (auto sampler : s.samplers)
    checkValid(d, sampler, "allocation-failure/invalid", false, ok);

  // Recovery changes the extent, not the source bytes.
  anari::setParameter(d, s.image, "end", uint64_t(3));
  anari::commitParameters(d, s.image);
  checkAll(s, "allocation-failure/recovery", GREEN, ok);
}

static void testAccumulation(anari::Device d, int sampleLimit, bool &ok)
{
  TexturedScene s(d, 2, TexFormat::FLOAT_RGBA, Ownership::MANAGED, 0);
  auto renderer = anari::newObject<anari::Renderer>(d, "quality");
  anari::setParameter(d, renderer, "sampleLimit", sampleLimit);
  anari::setParameter(d, renderer, "pixelSamples", 1);
  anari::setParameter(d, renderer, "denoise", false);
  setImage(d, renderer, "background", s.image, 2);
  anari::setParameter(d, s.frame, "renderer", renderer);
  anari::commitParameters(d, s.frame);
  anari::release(d, s.renderer);
  s.renderer = renderer;

  // Complete scene initialization before measuring steady accumulation.
  renderMeanColor(s);
  for (int i = 0; i < 16; ++i)
    renderMeanColor(s);
  int samples = 0;
  anari::getProperty(d, s.frame, "numSamples", samples, ANARI_WAIT);
  if ((sampleLimit > 0 && samples != sampleLimit) || samples < 8) {
    fprintf(
        stderr, "FAIL: accumulation did not settle (%d samples)\n", samples);
    ok = false;
  }
  const auto label = "accumulation/limit" + std::to_string(sampleLimit);
  // Only unmap the image: a renderer/camera/frame recommit would mask a
  // missing array-driven accumulation reset.
  rewriteImage(d, s.image, s.format, std::vector<vec3>(s.texelCount, RED));
  check(label + "/first-frame-after-write", renderMeanColor(s), RED, ok);
  anari::getProperty(d, s.frame, "numSamples", samples, ANARI_WAIT);
  if (samples != 1) {
    fprintf(stderr, "FAIL: sample count did not reset (%d)\n", samples);
    ok = false;
  }
}

int main()
{
  setvbuf(stdout, nullptr, _IOLBF, 0);
  auto device = makeVisRTXDevice(statusFunc);
  bool ok = true;
  for (auto format : {TexFormat::SRGB_RGBA, TexFormat::SRGB_RGB})
    for (auto ownership : {Ownership::MANAGED, Ownership::SHARED})
      testWindows(device, format, ownership, ok);
  for (int dim : {1, 2, 3}) {
    for (int numSamplers : {1, 3}) {
      for (auto format : {TexFormat::SRGB_RGBA, TexFormat::FLOAT_RGBA})
        for (auto ownership : {Ownership::MANAGED, Ownership::SHARED})
          testRepeatedWrites(device, dim, format, ownership, numSamplers, ok);
      testReplacement(device, dim, numSamplers, ok);
    }
    testInvalidTransitions(device, dim, false, ok);
    testInvalidTransitions(device, dim, true, ok);
  }
  testAccumulation(device, 0, ok);
  testAccumulation(device, 8, ok);
  testAllocationFailure(device, ok);
  testBackground(device, false, ok);
  testBackground(device, true, ok);
  anari::release(device, device);
  return ok ? 0 : 1;
}
