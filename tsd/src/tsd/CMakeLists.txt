## Copyright 2024 NVIDIA Corporation
## SPDX-License-Identifier: Apache-2.0

project_add_library(STATIC)

project_sources(
PRIVATE
  algorithms/computeScalarRange.cpp
  authoring/importers/detail/HDRImage.cpp
  authoring/importers/detail/importer_common.cpp
  authoring/importers/import_ASSIMP.cpp
  authoring/importers/import_DLAF.cpp
  authoring/importers/import_FLASH.cpp
  authoring/importers/import_HDRI.cpp
  authoring/importers/import_NBODY.cpp
  authoring/importers/import_NVDB.cpp
  authoring/importers/import_OBJ.cpp
  authoring/importers/import_PLY.cpp
  authoring/importers/import_RAW.cpp
  authoring/importers/import_volume.cpp
  authoring/procedural/generate_cylinders.cpp
  authoring/procedural/generate_material_orb.cpp
  authoring/procedural/generate_monkey.cpp
  authoring/procedural/generate_noiseVolume.cpp
  authoring/procedural/generate_randomSpheres.cpp
  authoring/procedural/generate_rtow.cpp
  authoring/serialization/serialization.cpp
  core/AnariObjectCache.cpp
  core/Context.cpp
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
  rendering/RenderIndexTreeHierarchy.cpp
)

project_include_directories(
PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/..>
)

project_link_libraries(
PUBLIC
  anari::helium
PRIVATE
  $<BUILD_INTERFACE:tsd_nanovdb>
  $<BUILD_INTERFACE:tsd_stb>
  $<BUILD_INTERFACE:tsd_tiny_obj_loader>
  $<BUILD_INTERFACE:tsd_tinyply>
)

if (WIN32)
  project_link_libraries(PUBLIC anari::anari_static)
else()
  project_link_libraries(PUBLIC anari::anari)
endif()

if (MSVC)
  project_compile_options(PRIVATE /bigobj)
endif()

if (TSD_USE_CUDA)
  project_compile_definitions(PUBLIC -DTSD_USE_CUDA=1)
  project_link_libraries(PUBLIC CUDA::cudart)
  set_source_files_properties(
    algorithms/computeScalarRange.cpp
    objects/SpatialField.cpp
    PROPERTIES
      COMPILE_FLAGS "--extended-lambda --expt-relaxed-constexpr"
      LANGUAGE CUDA
  )
else()
  project_compile_definitions(PUBLIC -DTSD_USE_CUDA=0)
endif()

if (TSD_ENABLE_SERIALIZATION)
  project_link_libraries(PRIVATE tsd_conduit)
  set_source_files_properties(authoring/serialization/serialization.cpp
    PROPERTIES
      COMPILE_DEFINITIONS TSD_ENABLE_SERIALIZATION=1
  )
else()
  set_source_files_properties(authoring/serialization/serialization.cpp
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
