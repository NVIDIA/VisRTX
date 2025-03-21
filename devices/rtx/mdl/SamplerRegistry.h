#pragma once

#include "libmdl/Core.h"

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

  // Material code
  Sampler *acquireSampler(
      const std::string &textureDesc, mi::neuraylib::ITransaction *transaction);
  bool releaseSampler(const Sampler *);

 private:
  libmdl::Core *m_core = {};
  DeviceGlobalState *m_deviceState = {};

  std::unordered_map<std::string, Sampler *> m_dbToSampler;

  Sampler *createSamplerFromDb(const std::string &textureDbName,
      mi::neuraylib::ITransaction *transaction);
};

} // namespace visrtx::mdl
