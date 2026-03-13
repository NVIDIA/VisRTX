## Copyright 2024-2026 NVIDIA Corporation
## SPDX-License-Identifier: Apache-2.0

project(tsd_core)

project_add_library(STATIC)

set(TSD_CORE_CUDA_SOURCES
  algorithms/detail/computeScalarRange_fixed16.cpp
  algorithms/detail/computeScalarRange_fixed8.cpp
  algorithms/detail/computeScalarRange_float32.cpp
  algorithms/detail/computeScalarRange_float64.cpp
  algorithms/detail/computeScalarRange_ufixed16.cpp
  algorithms/detail/computeScalarRange_ufixed8.cpp
)

if (TSD_USE_CUDA)
  list(APPEND TSD_CORE_CUDA_SOURCES algorithms/detail/vort_cuda.cu)
endif()

set(TSD_CORE_CXX_SOURCES
  algorithms/computeScalarRange.cpp
  AnariObjectCache.cpp
  ColorMapUtil.cpp
  DataStream.cpp
  Logging.cpp
  Parameter.cpp
  scene/Animation.cpp
  scene/AnyObjectUsePtr.cpp
  scene/Layer.cpp
  scene/LayerNodeData.cpp
  scene/Object.cpp
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
  scene/Scene.cpp
  scene/UpdateDelegate.cpp
  Token.cpp
)

project_sources(PRIVATE ${TSD_CORE_CUDA_SOURCES} ${TSD_CORE_CXX_SOURCES})

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
  set_source_files_properties(${TSD_CORE_CUDA_SOURCES}
    PROPERTIES
      COMPILE_OPTIONS "--extended-lambda;--expt-relaxed-constexpr;-Xcudafe=--diag_suppress=20012;-Wno-deprecated-gpu-targets"
      LANGUAGE CUDA
  )
else()
  project_compile_definitions(PUBLIC -DTSD_USE_CUDA=0)
endif()

# Set MSVC /Zc:preprocessor (needed for VS generator).
if(MSVC)
  set_source_files_properties(${TSD_CORE_CXX_SOURCES}
    PROPERTIES COMPILE_OPTIONS "/Zc:preprocessor"
  )
endif()
