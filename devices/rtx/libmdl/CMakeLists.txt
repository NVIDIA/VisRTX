# Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

project(libmdl LANGUAGES CXX)

list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)
find_package(MDL_SDK MODULE REQUIRED)

project_add_library(STATIC
  ArgumentBlockDescriptor.cpp
  ArgumentBlockInstance.cpp
  Core.cpp
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
