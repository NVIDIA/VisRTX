#pragma once

#include "libmdl/Core.h"

#include <anari/anari_cpp.hpp>

#include <mi/neuraylib/itransaction.h>

#include <string_view>
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

  // Material code
  Sampler *acquireSampler(std::string_view dbName,
      mi::neuraylib::ITarget_code::Texture_shape textureShape,
      mi::neuraylib::ITransaction *transaction);
  bool releaseSampler(const Sampler *);

 private:
  libmdl::Core *m_core = {};
  DeviceGlobalState *m_deviceState = {};

  std::unordered_map<std::string, Sampler *> m_dbToSampler;
};

} // namespace visrtx::mdl
