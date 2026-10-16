// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

// MaterialX's data library as the build embedded it
// (external/materialx/CMakeLists.txt generates the definitions from the
// MaterialX the device links): every .mtlx file of its libraries/ directory
// but the shader generators' (gen*/) and targets/.

namespace halcyon::pathtracer {

struct MaterialXDefinitionFile
{
  // Relative to libraries/, e.g. "stdlib/stdlib_defs.mtlx"
  const char *path;
  // The file's XML
  const char *content;
};

extern const MaterialXDefinitionFile MATERIALX_DEFINITION_FILES[];
extern const int MATERIALX_DEFINITION_FILE_COUNT;

} // namespace halcyon::pathtracer
