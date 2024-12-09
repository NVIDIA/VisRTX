## Copyright 2024 The Khronos Group
## SPDX-License-Identifier: Apache-2.0

project(libmdl LANGUAGES CXX)

find_package(mdl REQUIRED)

project_add_library(STATIC
  ArgumentBlockDescriptor.cpp
  ArgumentBlockInstance.cpp
  Core.cpp
  helpers.cpp
  ptx.cpp
  TimeStamp.cpp
)

project_link_libraries(
  PUBLIC mdl::mdl_sdk
         nonstd::span-lite nonstd::expected-lite
  PRIVATE fmt::fmt nonstd::scope-lite
)

project_include_directories(
  PUBLIC ..
)
