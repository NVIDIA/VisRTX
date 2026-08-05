# Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

project(libmdl LANGUAGES CXX)

list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)
find_package(MDL_SDK MODULE REQUIRED)

project_add_library(STATIC
  ArgumentBlockDescriptor.cpp
  ArgumentBlockInstance.cpp
  Core.cpp
  EmissionFold.cpp
  EmissionIR.cpp
  helpers.cpp
  ptx.cpp
  TimeStamp.cpp
)

project_link_libraries(
PRIVATE
  fmt::fmt
  nonstd::scope-lite
PUBLIC
  nonstd::span-lite
  nonstd::expected-lite
  MDL_SDK::MDL_SDK
)

project_include_directories(PUBLIC ..)

# The MDL backend takes its SM target from a fixed set (imdl_backend.h), which
# has no entry for every compute capability. Pick the highest one the floor
# covers so generated material PTX never outranks the GPU it links into. This
# list mirrors imdl_backend.h's allowed "sm_version" values at or above the
# OptiX floor (50); keep it in sync if the MDL SDK grows new entries. A floor
# with no exact match (e.g. 89) rounds down to the next lower entry (86), which
# is safe — sm_86 PTX runs on Ada — just not maximally matched.
set(MDL_SM_VERSIONS 50 52 60 61 62 70 75 80 86)
set(MDL_SM_VERSION 50)
foreach(sm IN LISTS MDL_SM_VERSIONS)
  if(NOT sm GREATER VISRTX_MIN_ARCH)
    set(MDL_SM_VERSION ${sm})
  endif()
endforeach()
target_compile_definitions(libmdl PRIVATE VISRTX_MDL_SM_VERSION="${MDL_SM_VERSION}")
