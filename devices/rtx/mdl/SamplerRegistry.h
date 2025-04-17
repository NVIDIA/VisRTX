#pragma once

#include <libmdl/Core.h>
#include <libmdl/ArgumentBlockDescriptor.h>

#include <anari/anari_cpp.hpp>

#include <mi/neuraylib/itransaction.h>

#include <string>
#include <unordered_map>

namespace visrtx {
class DeviceGlobalState;
class Sampler;
} // namespace visrtx

namespace visrtx::mdl {

class SamplerRegistry
{
 public:
  SamplerRegistry(libmdl::Core *core, DeviceGlobalState *deviceState);
  ~SamplerRegistry();

  Sampler *acquireSampler(const std::string &filePath, libmdl::ArgumentBlockDescriptor::ColorSpace colorSpace = libmdl::ArgumentBlockDescriptor::ColorSpace::Auto);
  Sampler *acquireSampler(const libmdl::ArgumentBlockDescriptor::TextureDescriptor &textureDesc);

  bool releaseSampler(const Sampler *);

 private:
  libmdl::Core *m_core = {};
  DeviceGlobalState *m_deviceState = {};

  std::unordered_map<std::string, Sampler *> m_dbToSampler;

  Sampler* loadFromFile(const std::string_view& filePath, libmdl::ArgumentBlockDescriptor::ColorSpace colorSpace = libmdl::ArgumentBlockDescriptor::ColorSpace::Auto);

  Sampler* loadFromDDS(const std::string_view& filePath, libmdl::ArgumentBlockDescriptor::ColorSpace colorSpace = libmdl::ArgumentBlockDescriptor::ColorSpace::Auto);
  Sampler* loadFromImage(const std::string_view& filePath, libmdl::ArgumentBlockDescriptor::ColorSpace colorSpace = libmdl::ArgumentBlockDescriptor::ColorSpace::Auto);

  Sampler* loadFromTextureDesc(const libmdl::ArgumentBlockDescriptor::TextureDescriptor &textureDesc);
};

} // namespace visrtx::mdl
