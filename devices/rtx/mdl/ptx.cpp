#include "ptx.h"

#include "MDLShaderEvalSurfaceMaterial_ptx.h"
#include "MDLTexture_ptx.h"

#include <array>

namespace visrtx::mdl::ptx {

ptx_blob MDLShaderEvalSurfaceMaterial{
    std::data(MDLShaderEvalSurfaceMaterial_ptx),
    std::size(MDLShaderEvalSurfaceMaterial_ptx),
};

ptx_blob MDLTexture{
    std::data(MDLTexture_ptx),
    std::size(MDLTexture_ptx),
};

} // namespace visrtx::mdl::ptx
