## Copyright 2024-2025 NVIDIA Corporation
## SPDX-License-Identifier: Apache-2.0

project(tsd_io)

project_add_library(STATIC)

project_sources(
PRIVATE
  importers/detail/HDRImage.cpp
  importers/detail/importer_common.cpp
  importers/detail/usd/MaterialCommon.cpp
  importers/detail/usd/OmniPbrMaterial.cpp
  importers/import_AGX.cpp
  importers/import_ASSIMP.cpp
  importers/import_AXYZ.cpp
  importers/import_DLAF.cpp
  importers/import_E57XYZ.cpp
  importers/import_FLASH.cpp
  importers/import_GLTF.cpp
  importers/import_HDRI.cpp
  importers/import_HSMESH.cpp
  importers/import_MHD.cpp
  importers/import_NBODY.cpp
  importers/import_NVDB.cpp
  importers/import_OBJ.cpp
  importers/import_USD.cpp
  importers/import_USD2.cpp
  importers/import_PDB.cpp
  importers/import_PLY.cpp
  importers/import_POINTSBIN.cpp
  importers/import_PT.cpp
  importers/import_RAW.cpp
  importers/import_SILO.cpp
  importers/import_SMESH.cpp
  importers/import_SWC.cpp
  importers/import_TRK.cpp
  importers/import_volume.cpp
  importers/import_VTI.cpp
  importers/import_VTU.cpp
  importers/import_XYZDP.cpp
  procedural/generate_cylinders.cpp
  procedural/generate_default_lights.cpp
  procedural/generate_hdri_dome.cpp
  procedural/generate_hdri_test_image.cpp
  procedural/generate_material_orb.cpp
  procedural/generate_monkey.cpp
  procedural/generate_noiseVolume.cpp
  procedural/generate_randomSpheres.cpp
  procedural/generate_rtow.cpp
  procedural/generate_sphereSetVolume.cpp
  serialization/export_RegularVolumeToNanoVDB.cpp
  serialization/export_SceneToUSD.cpp
  serialization/serialization_datatree.cpp
)

project_include_directories(
PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../..>
)

project_link_libraries(
PUBLIC
  tsd_core
PRIVATE
  $<BUILD_INTERFACE:tsd_ext_agx>
  $<BUILD_INTERFACE:tsd_ext_mikktspace>
  $<BUILD_INTERFACE:tsd_ext_nanovdb>
  $<BUILD_INTERFACE:tsd_ext_tinyexr>
  $<BUILD_INTERFACE:tsd_ext_tinyexr_impl>
  $<BUILD_INTERFACE:tsd_ext_tinyobjloader>
  $<BUILD_INTERFACE:tsd_ext_tinyply>
  $<BUILD_INTERFACE:tsd_ext_tinygltf>
  $<BUILD_INTERFACE:stb_image>
)

# Always use nanovdb
find_package(tsd_nanovdb MODULE REQUIRED)
project_link_libraries(PRIVATE tsd::nanovdb)


## Setup optional dependencies ##

if (TSD_USE_ASSIMP)
  find_package(assimp REQUIRED)
  project_compile_definitions(PRIVATE -DTSD_USE_ASSIMP=1)
  project_link_libraries(PRIVATE assimp::assimp)
else()
  project_compile_definitions(PRIVATE -DTSD_USE_ASSIMP=0)
endif()

if (TSD_USE_HDF5)
  find_package(HDF5 REQUIRED COMPONENTS CXX)
  project_compile_definitions(PRIVATE -DTSD_USE_HDF5=1)
  project_link_libraries(PRIVATE HDF5::HDF5)
else()
  project_compile_definitions(PRIVATE -DTSD_USE_HDF5=0)
endif()

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
  project_link_libraries(PRIVATE ${VTK_LIBRARIES})
  project_compile_definitions(PRIVATE -DTSD_USE_VTK=1)
else()
  project_compile_definitions(PRIVATE -DTSD_USE_VTK=0)
endif()

if (TSD_USE_SILO)
  find_package(tsd_silo MODULE REQUIRED)
  project_link_libraries(PRIVATE tsd::silo)
  project_compile_definitions(PRIVATE -DTSD_USE_SILO=1)
else()
  project_compile_definitions(PRIVATE -DTSD_USE_SILO=0)
endif()
