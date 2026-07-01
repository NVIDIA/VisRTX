## Copyright 2024-2026 NVIDIA Corporation
## SPDX-License-Identifier: Apache-2.0

if (NOT BUILD_TESTING)
  return()
endif()

project(tsdTests LANGUAGES CXX)
project_add_executable(
  test_App.cpp
  test_AnimationManager.cpp
  test_AnimationArchive.cpp
  test_AnimationManagerArchive.cpp
  test_ArchiveCompatibility.cpp
  test_CameraRendererArchive.cpp
  test_Algorithms.cpp
  test_AnyArray.cpp
  test_Array.cpp
  test_ComponentSerialization.cpp
  test_DataTree.cpp
  test_FlatMap.cpp
  test_Forest.cpp
  test_Geometry.cpp
  test_Importers.cpp
  test_LayerSubtreeArchive.cpp
  test_Manipulator.cpp
  test_ObjectPool.cpp
  test_Material.cpp
  test_Math.cpp
  test_Object.cpp
  test_ObjectArchive.cpp
  test_ObjectUsePtr.cpp
  test_Parameter.cpp
  test_Scene.cpp
  test_SceneArchive.cpp
  test_Token.cpp
)
if (TARGET tsd_scivis_studio_model)
  target_sources(${PROJECT_NAME} PRIVATE
    test_ProjectAssetTransaction.cpp
    test_SciVisStudio.cpp
  )
endif()
project_link_libraries(
PRIVATE
  tsd_algorithms
  tsd_animation
  tsd_app
  tsd_io
  tsd_rendering
  tsd_scene
  tsd_ext_catch2
)
if (TARGET tsd_scivis_studio_model)
  target_link_libraries(${PROJECT_NAME} PRIVATE tsd_scivis_studio_model)
endif()

add_test(NAME tsd::AnimationManager COMMAND ${PROJECT_NAME} "[AnimationManager]")
add_test(NAME tsd::App COMMAND ${PROJECT_NAME} "[App]")
add_test(NAME tsd::AnimationArchive COMMAND ${PROJECT_NAME} "[AnimationArchive]")
add_test(NAME tsd::AnimationManagerArchive
  COMMAND ${PROJECT_NAME} "[AnimationManagerArchive]")
add_test(NAME tsd::ArchiveCompatibility
  COMMAND ${PROJECT_NAME} "[ArchiveCompatibility]")
add_test(NAME tsd::CameraArchive COMMAND ${PROJECT_NAME} "[CameraArchive]")
add_test(NAME tsd::RendererArchive COMMAND ${PROJECT_NAME} "[RendererArchive]")
add_test(NAME tsd::Algorithms    COMMAND ${PROJECT_NAME} "[Algorithms]"   )
add_test(NAME tsd::AnyArray      COMMAND ${PROJECT_NAME} "[AnyArray]"     )
add_test(NAME tsd::Array         COMMAND ${PROJECT_NAME} "[Array]"        )
add_test(NAME tsd::ComponentSerialization
  COMMAND ${PROJECT_NAME} "[ComponentSerialization]")
add_test(NAME tsd::DataTree      COMMAND ${PROJECT_NAME} "[DataTree]"     )
add_test(NAME tsd::FlatMap       COMMAND ${PROJECT_NAME} "[FlatMap]"      )
add_test(NAME tsd::Forest        COMMAND ${PROJECT_NAME} "[Forest]"       )
add_test(NAME tsd::Geometry      COMMAND ${PROJECT_NAME} "[Geometry]"     )
add_test(NAME tsd::Importers     COMMAND ${PROJECT_NAME} "[Importers]"    )
add_test(NAME tsd::LayerSubtreeArchive
  COMMAND ${PROJECT_NAME} "[LayerSubtreeArchive]")
add_test(NAME tsd::Manipulator   COMMAND ${PROJECT_NAME} "[Manipulator]"  )
add_test(NAME tsd::ObjectPool    COMMAND ${PROJECT_NAME} "[ObjectPool]"   )
add_test(NAME tsd::Material      COMMAND ${PROJECT_NAME} "[Material]"     )
add_test(NAME tsd::Math          COMMAND ${PROJECT_NAME} "[Math]"         )
add_test(NAME tsd::Object        COMMAND ${PROJECT_NAME} "[Object]"       )
add_test(NAME tsd::ObjectArchive COMMAND ${PROJECT_NAME} "[ObjectArchive]")
add_test(NAME tsd::ObjectUsePtr  COMMAND ${PROJECT_NAME} "[ObjectUsePtr]" )
add_test(NAME tsd::Parameter     COMMAND ${PROJECT_NAME} "[Parameter]"    )
add_test(NAME tsd::Scene         COMMAND ${PROJECT_NAME} "[Scene]"        )
add_test(NAME tsd::SceneArchive  COMMAND ${PROJECT_NAME} "[SceneArchive]" )
add_test(NAME tsd::Token         COMMAND ${PROJECT_NAME} "[Token]"        )
if (TARGET tsd_scivis_studio_model)
  add_test(NAME tsd::SciVisStudio COMMAND ${PROJECT_NAME} "[SciVisStudio]")
endif()
