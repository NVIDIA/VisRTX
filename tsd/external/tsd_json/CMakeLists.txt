## Copyright 2026 NVIDIA Corporation
## SPDX-License-Identifier: Apache-2.0

project(tsd_ext_json LANGUAGES CXX)

anari_sdk_fetch_project(
  NAME ${PROJECT_NAME}
  URL https://github.com/nlohmann/json/archive/refs/tags/v3.11.3.zip
  MD5 80c21506b0e4c6c35ea52c4665e7e726
)

set(JSON_BuildTests OFF CACHE INTERNAL "")
set(JSON_Install OFF CACHE INTERNAL "")

add_library(${PROJECT_NAME} INTERFACE)

target_include_directories(${PROJECT_NAME}
  INTERFACE
    $<BUILD_INTERFACE:${tsd_ext_json_LOCATION}/include>
)
