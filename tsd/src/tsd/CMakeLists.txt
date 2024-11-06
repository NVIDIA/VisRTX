## Copyright 2024 NVIDIA Corporation
## SPDX-License-Identifier: Apache-2.0

project_add_library(STATIC)

project_sources(
PRIVATE
  authoring/procedural/generate_cylinders.cpp
  authoring/procedural/generate_material_orb.cpp
  authoring/procedural/generate_monkey.cpp
  authoring/procedural/generate_noiseVolume.cpp
  authoring/procedural/generate_randomSpheres.cpp
  authoring/procedural/generate_rtow.cpp
  authoring/importers/detail/HDRImage.cpp
  authoring/importers/detail/importer_common.cpp
  authoring/importers/import_ASSIMP.cpp
  authoring/importers/import_DLAF.cpp
  authoring/importers/import_HDRI.cpp
  authoring/importers/import_NBODY.cpp
  authoring/importers/import_OBJ.cpp
  authoring/importers/import_PLY.cpp
  authoring/serialization/serialization.cpp
  core/AnariObjectCache.cpp
  core/Context.cpp
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
  assimp::assimp
PRIVATE
  $<BUILD_INTERFACE:tsd_stb>
  $<BUILD_INTERFACE:tsd_tiny_obj_loader>
  $<BUILD_INTERFACE:tsd_tinyply>
)

if (TSD_ENABLE_SERIALIZATION)
  project_link_libraries(PRIVATE $<BUILD_INTERFACE:tsd_conduit>)
  set_source_files_properties(authoring/serialization/serialization.cpp
    PROPERTIES
      COMPILE_DEFINITIONS TSD_ENABLE_SERIALIZATION
  )
endif()
