# Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Host-only unit tests for the pure analytic primitive solvers. No CUDA/OptiX/GPU
# — these run on CPU and exercise devices/rtx/device/gpu/intersectPrimitives.h
# directly, so the intersection math is testable without a device.

project(rtx_test_Intersectors LANGUAGES CXX)

add_executable(${PROJECT_NAME} test_Intersectors.cpp)
target_include_directories(${PROJECT_NAME} PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../../../device)
target_link_libraries(${PROJECT_NAME} PRIVATE visrtx::glm)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)

add_test(NAME rtx::Intersectors COMMAND ${PROJECT_NAME})

project(rtx_test_IsosurfaceLinear LANGUAGES CXX)

add_executable(${PROJECT_NAME} test_IsosurfaceLinear.cpp)
target_include_directories(${PROJECT_NAME} PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../../../device)
target_link_libraries(${PROJECT_NAME} PRIVATE visrtx::glm)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)

add_test(NAME rtx::IsosurfaceLinear COMMAND ${PROJECT_NAME})
