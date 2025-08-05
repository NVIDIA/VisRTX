## Copyright 2024-2025 NVIDIA Corporation
## SPDX-License-Identifier: Apache-2.0

if (NOT BUILD_TESTING)
  return()
endif()

project(tsdTests LANGUAGES CXX)
project_add_executable(
  test_Array.cpp
  test_DataTree.cpp
  test_FlatMap.cpp
  test_Forest.cpp
  test_Geometry.cpp
  test_IndexedVector.cpp
  test_Material.cpp
  test_Math.cpp
  test_Object.cpp
  test_Parameter.cpp
  test_Token.cpp
)
project_link_libraries(PRIVATE tsd_core tsd_ext_catch2)

add_test(NAME tsd::Array         COMMAND ${PROJECT_NAME} "[Array]"        )
add_test(NAME tsd::DataTree      COMMAND ${PROJECT_NAME} "[DataTree]"     )
add_test(NAME tsd::FlatMap       COMMAND ${PROJECT_NAME} "[FlatMap]"      )
add_test(NAME tsd::Forest        COMMAND ${PROJECT_NAME} "[Forest]"       )
add_test(NAME tsd::Geometry      COMMAND ${PROJECT_NAME} "[Geometry]"     )
add_test(NAME tsd::IndexedVector COMMAND ${PROJECT_NAME} "[IndexedVector]")
add_test(NAME tsd::Material      COMMAND ${PROJECT_NAME} "[Material]"     )
add_test(NAME tsd::Math          COMMAND ${PROJECT_NAME} "[Math]"         )
add_test(NAME tsd::Object        COMMAND ${PROJECT_NAME} "[Object]"       )
add_test(NAME tsd::Parameter     COMMAND ${PROJECT_NAME} "[Parameter]"    )
add_test(NAME tsd::Token         COMMAND ${PROJECT_NAME} "[Token]"        )
