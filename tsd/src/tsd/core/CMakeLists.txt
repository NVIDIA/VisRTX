## Copyright 2024-2026 NVIDIA Corporation
## SPDX-License-Identifier: Apache-2.0

project(tsd_core)

project_add_library(STATIC)

project_sources(
PRIVATE
  algorithms/computeScalarRange.cpp
  algorithms/detail/computeScalarRange_fixed8.cpp
  algorithms/detail/computeScalarRange_fixed16.cpp
  algorithms/detail/computeScalarRange_ufixed8.cpp
  algorithms/detail/computeScalarRange_ufixed16.cpp
  algorithms/detail/computeScalarRange_float32.cpp
  algorithms/detail/computeScalarRange_float64.cpp
  scene/Animation.cpp
  scene/AnyObjectUsePtr.cpp
  scene/Layer.cpp
  scene/Object.cpp
  scene/Scene.cpp
  scene/UpdateDelegate.cpp
  scene/objects/Array.cpp
  scene/objects/Camera.cpp
  scene/objects/Geometry.cpp
  scene/objects/Light.cpp
  scene/objects/Material.cpp
  scene/objects/Renderer.cpp
  scene/objects/Sampler.cpp
  scene/objects/SpatialField.cpp
  scene/objects/Surface.cpp
  scene/objects/Volume.cpp
  AnariObjectCache.cpp
  ColorMapUtil.cpp
  DataStream.cpp
  Logging.cpp
  Parameter.cpp
  Token.cpp
)

project_include_directories(
PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../..>
)

project_link_libraries(PUBLIC anari::helium fmt::fmt)

if (WIN32)
  project_link_libraries(PUBLIC anari::anari_static)
else()
  project_link_libraries(PUBLIC anari::anari)
endif()

if (TSD_USE_CUDA)
  project_compile_definitions(PUBLIC -DTSD_USE_CUDA=1)
  project_link_libraries(PUBLIC CUDA::cudart)
  if(TARGET CCCL::CCCL)
    project_link_libraries(PUBLIC CCCL::CCCL)
  endif()
  set_source_files_properties(
    algorithms/detail/computeScalarRange_fixed8.cpp
    algorithms/detail/computeScalarRange_fixed16.cpp
    algorithms/detail/computeScalarRange_ufixed8.cpp
    algorithms/detail/computeScalarRange_ufixed16.cpp
    algorithms/detail/computeScalarRange_float32.cpp
    algorithms/detail/computeScalarRange_float64.cpp
    objects/SpatialField.cpp
    PROPERTIES
      COMPILE_OPTIONS "--extended-lambda;--expt-relaxed-constexpr;-Xcudafe=--diag_suppress=20012;-Wno-deprecated-gpu-targets"
      LANGUAGE CUDA
  )
else()
  project_compile_definitions(PUBLIC -DTSD_USE_CUDA=0)
endif()
