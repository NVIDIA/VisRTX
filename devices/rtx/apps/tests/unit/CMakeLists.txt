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

# The analytic ray/rect solver against an independent fp64 two-triangle oracle —
# the triangulated proxy ADR 0009 rejected as a representation, used here only as
# a test oracle.
project(rtx_test_LightGeometryOracle LANGUAGES CXX)

add_executable(${PROJECT_NAME} test_LightGeometryOracle.cpp)
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

add_test(NAME rtx::LightGeometryOracle COMMAND ${PROJECT_NAME})

# classifyEmission lives in libmdl, a standalone static lib with no CUDA/OptiX/GPU
# (the MDL SDK is dlopen'd at runtime), so the classifier is testable on the host.
# Gated on MDL support, since libmdl only exists then.
if (VISRTX_ENABLE_MDL_SUPPORT)
  find_package(Threads REQUIRED)
  add_executable(rtx_test_MdlCompileCoordinator
    test_MdlCompileCoordinator.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../../../device/mdl/MdlCompileCoordinator.cpp)
  target_include_directories(rtx_test_MdlCompileCoordinator PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../../../device)
  target_link_libraries(rtx_test_MdlCompileCoordinator PRIVATE Threads::Threads)
  target_compile_features(rtx_test_MdlCompileCoordinator PRIVATE cxx_std_17)
  add_test(NAME rtx::MdlCompileCoordinator COMMAND rtx_test_MdlCompileCoordinator)
  set_tests_properties(rtx::MdlCompileCoordinator PROPERTIES
    ENVIRONMENT "VISRTX_MDL_COMPILE_THREADS=2" TIMEOUT 30)
  add_test(NAME rtx::MdlCompileCoordinatorReserveFailure
    COMMAND rtx_test_MdlCompileCoordinator --reserve-failure)
  set_tests_properties(rtx::MdlCompileCoordinatorReserveFailure PROPERTIES
    ENVIRONMENT "VISRTX_MDL_COMPILE_THREADS=256" TIMEOUT 30)

  # Exercise the real registry directly (its symbols are private to the device
  # DSO). Reuse the device's embedded MDL/PTX, without creating a GPU device.
  set(mdl_device_binary_dir ${CMAKE_CURRENT_BINARY_DIR}/../../../device)
  set(mdl_registry_resources
    ${mdl_device_binary_dir}/mdl/default.mdl.cpp
    ${mdl_device_binary_dir}/mdl/physically_based.mdl.cpp)
  set_source_files_properties(${mdl_registry_resources} PROPERTIES GENERATED TRUE)
  add_executable(rtx_test_MdlMaterialRegistry
    test_MdlMaterialRegistry.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../../../device/mdl/MaterialRegistry.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../../../device/mdl/MdlCompileCoordinator.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../../../device/mdl/ptx.cpp
    ${mdl_registry_resources})
  target_link_libraries(rtx_test_MdlMaterialRegistry PRIVATE
    anari_library_visrtx libmdl fmt::fmt nonstd::scope-lite
    visrtx::glm OptiX::OptiX anari::helium
    CUDA::cudart Threads::Threads)
  target_include_directories(rtx_test_MdlMaterialRegistry PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../../../external/nanovdb/include)
  target_compile_definitions(rtx_test_MdlMaterialRegistry PRIVATE USE_MDL)
  target_compile_features(rtx_test_MdlMaterialRegistry PRIVATE cxx_std_17)
  add_test(NAME rtx::MdlMaterialRegistry COMMAND rtx_test_MdlMaterialRegistry)
  set_tests_properties(rtx::MdlMaterialRegistry PROPERTIES
    ENVIRONMENT "VISRTX_MDL_COMPILE_THREADS=4" TIMEOUT 120)

  project(rtx_test_MdlEmissionClassifier LANGUAGES CXX)

  add_executable(${PROJECT_NAME} test_MdlEmissionClassifier.cpp)
  # libmdl's Core.h includes <fmt/core.h> but links fmt PRIVATE, so a consumer
  # that only links libmdl must pull fmt itself.
  target_link_libraries(${PROJECT_NAME} PRIVATE libmdl fmt::fmt)
  target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)

  add_test(NAME rtx::MdlEmissionClassifier COMMAND ${PROJECT_NAME})
  # The test SKIPs (not fails) when the MDL SDK shared library is not loadable.
  set_tests_properties(rtx::MdlEmissionClassifier PROPERTIES SKIP_RETURN_CODE 77)

  project(rtx_test_MdlSceneDataDescriptor LANGUAGES CXX)
  add_executable(${PROJECT_NAME} test_MdlSceneDataDescriptor.cpp)
  target_link_libraries(${PROJECT_NAME} PRIVATE libmdl fmt::fmt)
  target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)
  add_test(NAME rtx::MdlSceneDataDescriptor COMMAND ${PROJECT_NAME})
  set_tests_properties(rtx::MdlSceneDataDescriptor PROPERTIES SKIP_RETURN_CODE 77)

  # Binding resolution uses only owned host data: no CUDA or SDK runtime calls.
  project(rtx_test_MdlSceneDataInstance LANGUAGES CXX)
  add_executable(${PROJECT_NAME}
    test_MdlSceneDataInstance.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../../../device/mdl/SceneDataInstance.cpp)
  target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../../../device)
  target_link_libraries(${PROJECT_NAME} PRIVATE libmdl)
  target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)
  add_test(NAME rtx::MdlSceneDataInstance COMMAND ${PROJECT_NAME})
endif()
