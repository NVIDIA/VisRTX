#pragma once

#include <mi/base/handle.h>
#include <mi/neuraylib/imdl_backend.h>

#include <vector>

namespace visrtx {

// Material information structure.
class MDLMaterialInfo
{
 public:
  MDLMaterialInfo(const mi::neuraylib::ITarget_argument_block *argblock)
      : m_argblock(argblock)
  {}

  std::vector<char> getArgumentBlockData() const
  {
    if (m_argblock) {
      return std::vector(m_argblock->get_data(),
          m_argblock->get_data() + m_argblock->get_size());
    }
    return {};
  }

 private:
  mi::base::Handle<mi::neuraylib::ITarget_argument_block const> m_argblock;
};

} // namespace visrtx
