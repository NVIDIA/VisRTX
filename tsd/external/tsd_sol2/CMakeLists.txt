## Copyright 2026 NVIDIA Corporation
## SPDX-License-Identifier: Apache-2.0

# Sol3 (sol2 v3.x) - Modern C++ Lua Bindings
# Header-only library, downloads sol3 headers

project(tsd_ext_sol2 LANGUAGES CXX)

anari_sdk_fetch_project(
  NAME ${PROJECT_NAME}
  URL https://github.com/ThePhD/sol2/archive/refs/tags/v3.5.0.zip
  MD5 2a52be066fbb16a44ad9568a6ebecc84
)

project_add_library(INTERFACE)

project_include_directories(
  INTERFACE
    $<BUILD_INTERFACE:${tsd_ext_sol2_LOCATION}/include>
)

# Link against Lua
project_link_libraries(INTERFACE tsd_ext_lua)

# Sol3 configuration
project_compile_definitions(INTERFACE SOL_ALL_SAFETIES_ON=1 SOL_USING_CXX_LUA=0)
