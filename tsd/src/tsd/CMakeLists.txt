## Copyright 2024-2025 NVIDIA Corporation
## SPDX-License-Identifier: Apache-2.0

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
  authoring/importers/detail/HDRImage.cpp
  authoring/importers/detail/importer_common.cpp
  authoring/importers/import_ASSIMP.cpp
  authoring/importers/import_DLAF.cpp
  authoring/importers/import_FLASH.cpp
  authoring/importers/import_HDRI.cpp
  authoring/importers/import_HSMESH.cpp
  authoring/importers/import_MHD.cpp
  authoring/importers/import_NBODY.cpp
  authoring/importers/import_NVDB.cpp
  authoring/importers/import_OBJ.cpp
  authoring/importers/import_USD.cpp
  authoring/importers/import_PDB.cpp
  authoring/importers/import_PLY.cpp
  authoring/importers/import_PT.cpp
  authoring/importers/import_RAW.cpp
  authoring/importers/import_SWC.cpp
  authoring/importers/import_volume.cpp
  authoring/importers/import_XYZDP.cpp
  authoring/importers/import_VTU.cpp
  authoring/procedural/generate_cylinders.cpp
  authoring/procedural/generate_hdri_dome.cpp
  authoring/procedural/generate_hdri_test_image.cpp
  authoring/procedural/generate_material_orb.cpp
  authoring/procedural/generate_monkey.cpp
  authoring/procedural/generate_noiseVolume.cpp
  authoring/procedural/generate_randomSpheres.cpp
  authoring/procedural/generate_rtow.cpp
  authoring/serialization/serialization_conduit.cpp
  authoring/serialization/serialization_datatree.cpp
  core/AnariObjectCache.cpp
  core/Context.cpp
  core/Layer.cpp
  core/Logging.cpp
  core/Object.cpp
  core/Parameter.cpp
  core/Token.cpp
  core/UpdateDelegate.cpp
  objects/Array.cpp
  objects/Geometry.cpp
  objects/Light.cpp
  objects/Material.cpp
  objects/Sampler.cpp
  objects/SpatialField.cpp
  objects/Surface.cpp
  objects/Volume.cpp
  rendering/RenderIndex.cpp
  rendering/RenderIndexFlatRegistry.cpp
  rendering/RenderIndexAllLayers.cpp
)

project_include_directories(
PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/..>
)

project_link_libraries(
PUBLIC
  anari::helium
  fmt::fmt
PRIVATE
  $<BUILD_INTERFACE:tsd_mikktspace>
  $<BUILD_INTERFACE:tsd_nanovdb>
  $<BUILD_INTERFACE:stb_image>
  $<BUILD_INTERFACE:tsd_tinyexr>
  $<BUILD_INTERFACE:tsd_tiny_obj_loader>
  $<BUILD_INTERFACE:tsd_tinyply>
)

if (WIN32)
  project_link_libraries(PUBLIC anari::anari_static)
else()
  project_link_libraries(PUBLIC anari::anari)
endif()

if (MSVC)
  set_source_files_properties(
    authoring/procedural/generate_material_orb.cpp
    PROPERTIES COMPILE_FLAGS /bigobj)
endif()


# Set TSD_USE_TORCH for OptiX 9.0
if(TSD_USE_TORCH)
  find_package(Torch REQUIRED)
  if (MSVC)
    set_source_files_properties(
      authoring/importers/import_PT.cpp
      PROPERTIES COMPILE_OPTIONS "/arch:AVX"
    )
  else()
    set_source_files_properties(
      authoring/importers/import_PT.cpp
      PROPERTIES COMPILE_OPTIONS "-mf16c"
    )
  endif()
  project_link_libraries(PRIVATE ${TORCH_LIBRARIES})
  project_compile_definitions(PUBLIC TSD_USE_TORCH=1)
else()
  project_compile_definitions(PUBLIC TSD_USE_TORCH=0)
endif()

if (TSD_USE_CUDA)
  project_compile_definitions(PUBLIC -DTSD_USE_CUDA=1)
  project_link_libraries(PUBLIC CUDA::cudart)
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

if (TSD_ENABLE_SERIALIZATION)
  project_link_libraries(PRIVATE tsd_conduit)
  set_source_files_properties(authoring/serialization/serialization_conduit.cpp
    PROPERTIES
      COMPILE_DEFINITIONS TSD_ENABLE_SERIALIZATION=1
  )
else()
  set_source_files_properties(authoring/serialization/serialization_conduit.cpp
    PROPERTIES
      COMPILE_DEFINITIONS TSD_ENABLE_SERIALIZATION=0
  )
endif()

if (TSD_USE_HDF5)
  find_package(HDF5 REQUIRED COMPONENTS CXX)
  project_compile_definitions(PRIVATE -DTSD_USE_HDF5=1)
  project_link_libraries(PRIVATE HDF5::HDF5)
else()
  project_compile_definitions(PRIVATE -DTSD_USE_HDF5=0)
endif()

if (TSD_USE_ASSIMP)
  find_package(assimp REQUIRED)
  project_compile_definitions(PRIVATE -DTSD_USE_ASSIMP=1)
  project_link_libraries(PRIVATE assimp::assimp)
else()
  project_compile_definitions(PRIVATE -DTSD_USE_ASSIMP=0)
endif()

if (TSD_USE_USD)
  find_package(OpenGL)
  find_package(X11)
  find_package(pxr REQUIRED usd usdGeom tf gf usdShade usdLux)
  project_compile_definitions(PRIVATE -DTSD_USE_USD=1)
  project_link_libraries(PRIVATE ${PXR_LIBRARIES})
else()
  project_compile_definitions(PRIVATE -DTSD_USE_USD=0)
endif()

if (TSD_USE_VTK)
  find_package(VTK REQUIRED)
  include(${VTK_USE_FILE})
  project_link_libraries(PRIVATE ${VTK_LIBRARIES})
  project_compile_definitions(PRIVATE -DTSD_USE_VTK=1)
else()
  project_compile_definitions(PRIVATE -DTSD_USE_VTK=0)
endif()
