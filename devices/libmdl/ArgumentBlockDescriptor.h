#pragma once

#include "libmdl/Core.h"

#include <mi/base/handle.h>
#include <mi/neuraylib/argument_editor.h>
#include <mi/neuraylib/icompiled_material.h>
#include <mi/neuraylib/imdl_backend.h>
#include <mi/neuraylib/imdl_factory.h>
#include <mi/neuraylib/itransaction.h>
#include <mi/neuraylib/ivalue.h>

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace visrtx::libmdl {

struct ArgumentBlockDescriptor
{
  enum class ArgumentType
  {
    Bool,
    Int,
    Int2,
    Int3,
    Int4,
    Float,
    Float2,
    Float3,
    Float4,
    Color,
    Texture,
  };

  struct Argument
  {
    std::string name;
    ArgumentType type;
  };

  struct TextureDescriptor
  {
    enum class ColorSpace
    {
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

    std::string url;
    ColorSpace colorSpace{ColorSpace::sRGB};
    Shape shape{Shape::TwoD};
    struct
    {
      const float *data = {};
      std::uint64_t dims[3] = {};
    } bsdf;
  };

  ArgumentBlockDescriptor() = default;

  ArgumentBlockDescriptor(libmdl::Core *core,
      const mi::neuraylib::ICompiled_material *compiledMaterial,
      const mi::neuraylib::ITarget_code *targetCode,
      const std::vector<TextureDescriptor> &textureDescs,
      std::uint32_t argumentBlockIndex = 0);

  mi::base::Handle<const mi::neuraylib::ITarget_code> m_targetCode;
  mi::base::Handle<const mi::neuraylib::ITarget_value_layout>
      m_argumentBlockLayout;
  mi::base::Handle<const mi::neuraylib::ITarget_argument_block> m_argumentBlock;

  std::vector<Argument> m_arguments;
  std::vector<TextureDescriptor> m_defaultAndBodyTextureDescriptors;
  std::unordered_map<std::string, mi::neuraylib::Target_value_layout_state>
      m_nameToLayout;
};

} // namespace visrtx::libmdl
