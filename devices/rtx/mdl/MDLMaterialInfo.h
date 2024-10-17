#pragma once

#include <mi/base/handle.h>
#include <mi/neuraylib/imdl_backend.h>

#include <vector>
#include "gpu/gpu_objects.h"

namespace visrtx {

class Sampler;

// Material information structure.
class MDLMaterialInfo
{
 public:
  MDLMaterialInfo(
    const mi::neuraylib::ITarget_code *targetCode,
    std::vector<const Sampler*> samplers)
      : m_samplers(samplers)
  {
    if (auto argblockCount = targetCode->get_argument_block_count();
        argblockCount == 1) {
      m_argblock = targetCode->get_argument_block(0);
    }
  }

  std::vector<char> getArgumentBlockData() const {
    return m_argblock ? std::vector(m_argblock->get_data(), m_argblock->get_data() + m_argblock->get_size()) : std::vector<char>{};
  }

  std::vector<const Sampler*> getSamplers() const { return m_samplers; }

 private:
  mi::base::Handle<mi::neuraylib::ITarget_argument_block const> m_argblock;
  std::vector<const Sampler*> m_samplers;
};

} // namespace visrtx
