## Copyright 2026 NVIDIA Corporation
## SPDX-License-Identifier: Apache-2.0

project(tsd_algorithms)

project_add_library(STATIC)

# CPU backend (always present)
project_sources(PRIVATE
  cpu/toneMap.cpp
  cpu/autoExposure.cpp
  cpu/outputTransform.cpp
  cpu/visualizeAOV.cpp
  cpu/boxOutline.cpp
  cpu/outline.cpp
  cpu/clearBuffers.cpp
  cpu/convertColorBuffer.cpp
  cpu/depthCompositeFrame.cpp
)

project_include_directories(
PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>/../..
)

project_link_libraries(PUBLIC tsd_core)

# CPU internal: TBB or serial
if (TSD_USE_TBB)
  find_package(TBB REQUIRED)
  project_link_libraries(PRIVATE TBB::tbb)
  project_compile_definitions(PRIVATE ENABLE_TBB)
endif()

# GPU backend: CUDA
if (TSD_USE_CUDA)
  find_package(thrust REQUIRED HINTS ${CUDAToolkit_LIBRARY_DIR}/cmake)
  project_link_libraries(PUBLIC CUDA::cudart Thrust::Thrust)
  project_compile_definitions(PUBLIC TSD_ALGORITHMS_HAS_CUDA)

  set(TSD_ALGORITHMS_CUDA_SOURCES
    cuda/toneMap.cu
    cuda/autoExposure.cu
    cuda/outputTransform.cu
    cuda/visualizeAOV.cu
    cuda/boxOutline.cu
    cuda/outline.cu
    cuda/clearBuffers.cu
    cuda/convertColorBuffer.cu
    cuda/depthCompositeFrame.cu
  )
  project_sources(PRIVATE ${TSD_ALGORITHMS_CUDA_SOURCES})
  set_source_files_properties(${TSD_ALGORITHMS_CUDA_SOURCES}
    PROPERTIES
      COMPILE_OPTIONS "--extended-lambda;--expt-relaxed-constexpr;-Xcudafe=--diag_suppress=20012;-Wno-deprecated-gpu-targets"
      LANGUAGE CUDA
  )
endif()
