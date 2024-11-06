## Copyright 2024 NVIDIA Corporation
## SPDX-License-Identifier: Apache-2.0

if (NOT TSD_ENABLE_SERIALIZATION)
  return()
endif()

project(tsd_conduit LANGUAGES CXX)
find_package(Conduit REQUIRED)
project_add_library(INTERFACE)
project_link_libraries(INTERFACE conduit::conduit)
project_compile_definitions(INTERFACE USE_CONDUIT)
