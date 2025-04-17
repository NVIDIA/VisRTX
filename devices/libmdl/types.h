#pragma once

#include <cstdint>
#include <string>

namespace visrtx::libmdl {

enum class ColorSpace
{
  Auto,
  Linear,
  sRGB,
};

enum class Shape
{
  Unknown,
  TwoD,
  ThreeD,
  Cube,
  PTex,
  BsdfData,
};

struct TextureDescriptor
{
  uint64_t knownIndex;

  std::string url;
  ColorSpace colorSpace{ColorSpace::sRGB};
  Shape shape{Shape::TwoD};
  struct
  {
    const float *data = {};
    std::uint64_t dims[3] = {};
    const char *pixelFormat = {};
  } bsdf;
};

} // namespace visrtx::libmdl