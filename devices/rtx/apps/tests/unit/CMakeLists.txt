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

project(rtx_test_IntersectorsOracle LANGUAGES CXX)

add_executable(${PROJECT_NAME} test_IntersectorsOracle.cpp)
target_include_directories(${PROJECT_NAME} PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../../../device)
target_link_libraries(${PROJECT_NAME} PRIVATE visrtx::glm)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)

add_test(NAME rtx::IntersectorsOracle COMMAND ${PROJECT_NAME})

project(rtx_test_IsosurfaceLinear LANGUAGES CXX)

add_executable(${PROJECT_NAME} test_IsosurfaceLinear.cpp)
target_include_directories(${PROJECT_NAME} PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../../../device)
target_link_libraries(${PROJECT_NAME} PRIVATE visrtx::glm)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)

add_test(NAME rtx::IsosurfaceLinear COMMAND ${PROJECT_NAME})

# lightPickPower operates on the GPU LightGPUData struct, so this host test pulls
# the OptiX/CUDA/nanovdb/anari headers its definition transitively needs.
project(rtx_test_LightPickPower LANGUAGES CXX)

add_executable(${PROJECT_NAME} test_LightPickPower.cpp)
target_include_directories(${PROJECT_NAME} PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../../../device
  ${CMAKE_CURRENT_LIST_DIR}/../../../external/nanovdb/include
  ${OptiX_ROOT_DIR}/include)
target_link_libraries(${PROJECT_NAME} PRIVATE
  visrtx::glm
  anari::anari_headers
  CUDA::cudart)
target_compile_definitions(${PROJECT_NAME} PRIVATE GLM_ENABLE_EXPERIMENTAL)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)

add_test(NAME rtx::LightPickPower COMMAND ${PROJECT_NAME})

# The shared rect/ring area-light leaves (ADR 0009). lightGeometry.h is kept
# CUDA-free precisely so it can be tested here; like lightPickPower it still
# needs the OptiX/CUDA/nanovdb/anari headers that GPU struct definitions pull in.
project(rtx_test_LightGeometry LANGUAGES CXX)

add_executable(${PROJECT_NAME} test_LightGeometry.cpp)
target_include_directories(${PROJECT_NAME} PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../../../device
  ${CMAKE_CURRENT_LIST_DIR}/../../../external/nanovdb/include
  ${OptiX_ROOT_DIR}/include)
target_link_libraries(${PROJECT_NAME} PRIVATE
  visrtx::glm
  anari::anari_headers
  CUDA::cudart)
target_compile_definitions(${PROJECT_NAME} PRIVATE GLM_ENABLE_EXPERIMENTAL)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)

add_test(NAME rtx::LightGeometry COMMAND ${PROJECT_NAME})

# classifyEmission lives in libmdl, a standalone static lib with no CUDA/OptiX/GPU
# (the MDL SDK is dlopen'd at runtime), so the classifier is testable on the host.
# Gated on MDL support, since libmdl only exists then.
if (VISRTX_ENABLE_MDL_SUPPORT)
  project(rtx_test_MdlEmissionClassifier LANGUAGES CXX)

  add_executable(${PROJECT_NAME} test_MdlEmissionClassifier.cpp)
  # libmdl's Core.h includes <fmt/core.h> but links fmt PRIVATE, so a consumer
  # that only links libmdl must pull fmt itself.
  target_link_libraries(${PROJECT_NAME} PRIVATE libmdl fmt::fmt)
  target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)

  add_test(NAME rtx::MdlEmissionClassifier COMMAND ${PROJECT_NAME})
  # The test SKIPs (not fails) when the MDL SDK shared library is not loadable.
  set_tests_properties(rtx::MdlEmissionClassifier PROPERTIES SKIP_RETURN_CODE 77)
endif()
