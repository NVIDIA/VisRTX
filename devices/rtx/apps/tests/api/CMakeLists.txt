# Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

macro(make_test TEST_NAME)
  project(${TEST_NAME} LANGUAGES CXX)
  project_add_executable(${PROJECT_NAME}.cpp)
  project_link_libraries(
  PRIVATE
    visrtx::stb_image
    anari_library_visrtx
    anari::anari_static
  )
  add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
endmacro()

if (VISRTX_ENABLE_MDL_SUPPORT)
  make_test(TestMdlSceneData)
endif()
make_test(TestQuadLightShaping)
make_test(TestQuadLightFocus)
make_test(TestQuadLightTexture)
make_test(TestDenoiserReuse)

if (WIN32)
  return()
endif()

make_test(TestDeviceInit)
make_test(TestEvictGPUArray)
make_test(TestMultiThreadedObjectCreation)
make_test(TestSpheres)
make_test(TestFrameChannelTypes)
make_test(TestCylinderCaps)
make_test(TestConeApex)
make_test(TestConeCylinderTransforms)
make_test(TestConeCylinderDegenerate)
make_test(TestConeCylinderTransparency)
make_test(TestRotatedSphereShading)
make_test(TestIsosurfaceRamp)
make_test(TestCellCentered)
make_test(TestPrimitiveInterior)
make_test(TestGroundSphereShadows)
make_test(TestPowerLightPick)
make_test(TestRayBufferCamera)
make_test(TestCameraNearFar)
make_test(TestMaterialSampling)
set_tests_properties(TestMaterialSampling PROPERTIES TIMEOUT 120)
make_test(TestSampleAddressing)
target_include_directories(TestSampleAddressing PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../../../device)
target_link_libraries(TestSampleAddressing PRIVATE CUDA::cudart)
set_tests_properties(TestSampleAddressing PROPERTIES TIMEOUT 120)
make_test(TestPowerLightPickEnv)
make_test(TestEmissiveGeometryLight)
add_executable(BenchmarkCudaOrdering BenchmarkCudaOrdering.cu)
target_link_libraries(BenchmarkCudaOrdering PRIVATE CUDA::cudart)
set_target_properties(BenchmarkCudaOrdering PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
add_executable(BenchmarkGeometryPreprocessing BenchmarkGeometryPreprocessing.cpp)
target_link_libraries(BenchmarkGeometryPreprocessing PRIVATE anari_library_visrtx anari::anari_static CUDA::cudart)
add_executable(TestPreparationStream TestPreparationStream.cpp)
target_link_libraries(TestPreparationStream PRIVATE CUDA::cudart Threads::Threads)
add_test(NAME TestPreparationStream COMMAND TestPreparationStream)
add_executable(TestRegistrySnapshot TestRegistrySnapshot.cpp
  ../../../device/utility/MemoryAllocation.cpp)
target_include_directories(TestRegistrySnapshot PRIVATE
  ../../../device ../../../external/nanovdb/include ${OptiX_ROOT_DIR}/include)
target_link_libraries(TestRegistrySnapshot PRIVATE
  CUDA::cudart anari::anari_static visrtx::glm)
target_compile_definitions(TestRegistrySnapshot PRIVATE GLM_ENABLE_EXPERIMENTAL)
add_test(NAME TestRegistrySnapshot COMMAND TestRegistrySnapshot)
make_test(TestDeferredGeometryPreparation)
target_link_libraries(TestDeferredGeometryPreparation PRIVATE CUDA::cudart)
make_test(TestGeometryPreparationReuse)
target_link_libraries(TestGeometryPreparationReuse PRIVATE CUDA::cudart)
make_test(TestSharedGeometryLifecycle)
target_link_libraries(TestSharedGeometryLifecycle PRIVATE CUDA::cudart)
make_test(TestGeometryPreprocessingFailure)
target_link_libraries(TestGeometryPreprocessingFailure PRIVATE CUDA::cudart)
add_test(NAME TestGeometryBoundsFailure COMMAND TestGeometryPreprocessingFailure --bounds)
# CUDA API interposition needs ELF dynamic symbols; static cudart calls cannot
# be intercepted. Configure with -DCMAKE_CUDA_RUNTIME_LIBRARY=Shared to run.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_CUDA_RUNTIME_LIBRARY STREQUAL "Shared")
  make_test(TestFramePublicationRecovery)
  target_link_libraries(TestFramePublicationRecovery PRIVATE CUDA::cudart ${CMAKE_DL_LIBS})
  set_target_properties(TestFramePublicationRecovery PROPERTIES ENABLE_EXPORTS ON)
  make_test(TestCudaPreparationFailure)
  target_link_libraries(TestCudaPreparationFailure PRIVATE CUDA::cudart ${CMAKE_DL_LIBS} Threads::Threads)
  target_include_directories(TestCudaPreparationFailure PRIVATE ${OptiX_ROOT_DIR}/include)
  set_target_properties(TestCudaPreparationFailure PROPERTIES ENABLE_EXPORTS ON)
  add_test(NAME TestIsosurfacePreparationRecovery
    COMMAND TestCudaPreparationFailure isosurface)
  foreach(geometry sphere sphere-indexed sphere-completion sphere-dependency
      sphere-undrained-completion
      sphere-readback-allocation sphere-batch-completion
      sphere-batch-upload-completion
      sphere-upload-allocation sphere-upload-copy sphere-upload-completion
      sphere-upload-partial)
    add_test(NAME TestSpherePreparationRecovery_${geometry}
      COMMAND TestCudaPreparationFailure ${geometry})
  endforeach()
  foreach(failure registry-copy registry-sync registry-partial
      instance-allocation instance-copy instance-sync world-copy world-sync)
    add_test(NAME TestSceneUploadRecovery_${failure}
      COMMAND TestCudaPreparationFailure ${failure})
  endforeach()
  foreach(failure bvh-copy bvh-sync tlas-sync)
    add_test(NAME TestBvhPreparationRecovery_${failure}
      COMMAND TestCudaPreparationFailure ${failure})
  endforeach()
  foreach(failure allocation copy work-sync copy-sync pending-retirement)
    add_test(NAME TestArrayPrivatizationFailure_${failure}
      COMMAND TestCudaPreparationFailure ${failure})
  endforeach()
endif()
make_test(TestDeviceGeometryVisibility)
target_link_libraries(TestDeviceGeometryVisibility PRIVATE CUDA::cudart)
add_test(NAME TestSharedGeometryInputs COMMAND TestDeviceGeometryVisibility --shared-inputs)
add_test(NAME TestDeviceArrayRetirement COMMAND TestDeviceGeometryVisibility --retire)
add_test(NAME TestDeviceSdfExpansion COMMAND TestDeviceGeometryVisibility sdf-expanded)
foreach(type triangle sphere curve sdf cone cylinder)
  add_test(NAME TestDeviceGeometryVisibility_${type} COMMAND TestDeviceGeometryVisibility ${type})
endforeach()
make_test(TestQuadLightSide)
make_test(TestVisibleAreaLight)
make_test(TestEmissiveSphereLight)
target_link_libraries(TestEmissiveSphereLight PRIVATE CUDA::cudart)
# Sampling radiometry retains host anchors and all-device integration;
# individual input roles use deterministic bounds/visibility above.
make_test(TestEmissiveInstanceScale)
make_test(TestEmissiveSphereInstanceScale)
make_test(TestEmissiveCylinderConeLight)
target_link_libraries(TestEmissiveCylinderConeLight PRIVATE CUDA::cudart)
# Each family keeps independent identity/nonuniform-transform references;
# cap residency is isolated at identity, with all-device coverage also scaled.
# These interleave attribute-driven emission cases, and MDL does not support
# attributes — native-PBR only. (Textured emission itself works on the MDL
# wrapper too; TestEmissiveMaterialParity covers it in that config.)
if (NOT VISRTX_USE_MDL_FOR_PHYSICALLY_BASED)
  make_test(TestEmissiveTexturedLight)
  make_test(TestEmissiveInstanceAttribute)
endif()
make_test(TestPbrSpecularDefault)
make_test(TestNormalMapDecode)
make_test(TestGeneratedTangents)
target_link_libraries(TestGeneratedTangents PRIVATE CUDA::cudart)
add_test(NAME TestDeviceGeneratedTangents COMMAND TestGeneratedTangents --device)
make_test(TestShadingNormalGrazing)
if (VISRTX_USE_MDL_FOR_PHYSICALLY_BASED)
  # Its closed-form reference models the backend's diffuse lobe.
  target_compile_definitions(
    TestShadingNormalGrazing PRIVATE VISRTX_TEST_PHYSICALLY_BASED_IS_MDL)
endif()
make_test(TestSamplerAlphaFill)
make_test(TestTextureDataUpdate)
make_test(TestAlphaCutoffMode)
make_test(TestRendererFog)
make_test(TestRendererFogDiscovery)
make_test(TestRendererFogDistance)
make_test(TestRendererFogBackground)
make_test(TestRendererFogInteractions)
make_test(TestRendererFogCoverage)
make_test(TestRendererFogTransport)
make_test(TestEmissionRadianceScale)
make_test(TestEmissiveRecommit)
# The raw `mdl` material type and the `physicallyBasedMDL` subtype only exist
# with MDL support compiled in.
if (VISRTX_ENABLE_MDL_SUPPORT)
  target_compile_definitions(
    TestMaterialSampling PRIVATE VISRTX_TEST_MDL_WRAPPER)
  target_compile_definitions(
    TestEmissiveRecommit PRIVATE VISRTX_TEST_MDL_WRAPPER)
  # Adds the physicallyBasedMDL leg, so the module's specular default is
  # exercised even when the physicallyBased backend is native CUDA.
  target_compile_definitions(
    TestPbrSpecularDefault PRIVATE VISRTX_TEST_MDL_WRAPPER)
  target_compile_definitions(
    TestNormalMapDecode PRIVATE VISRTX_TEST_MDL_WRAPPER)
  target_compile_definitions(
    TestGeneratedTangents PRIVATE VISRTX_TEST_MDL_WRAPPER)
  target_compile_definitions(
    TestShadingNormalGrazing PRIVATE VISRTX_TEST_MDL_WRAPPER)
  make_test(TestEmissiveMdlLight)
  make_test(TestEmissionDescriptorPolicy)
  make_test(TestMdlPipelineRebuild)
  make_test(TestMdlModuleReuse)
  set_tests_properties(TestMdlModuleReuse PROPERTIES TIMEOUT 120)
  add_test(NAME TestMdlModuleReuseSerial COMMAND TestMdlModuleReuse)
  set_tests_properties(TestMdlModuleReuseSerial PROPERTIES
    ENVIRONMENT "VISRTX_MDL_COMPILE_THREADS=1" TIMEOUT 120)
  make_test(TestMdlSearchPaths)
  set_tests_properties(TestMdlSearchPaths PROPERTIES TIMEOUT 300)
  make_test(TestMdlIneurayHandoff)
  make_test(TestMdlIneurayIntrospect)
  set_tests_properties(TestMdlIneurayHandoff TestMdlIneurayIntrospect
    PROPERTIES TIMEOUT 120)
  make_test(TestMdlLogging)
  set_tests_properties(TestMdlLogging PROPERTIES TIMEOUT 120)
  make_test(TestMdlMultiDevice)
  set_tests_properties(TestMdlMultiDevice PROPERTIES TIMEOUT 300)
  make_test(TestMdlMaterialRegistryRelease)
  make_test(TestMdlConcurrentCompile)
  make_test(TestMdlCompileBenchmark)
  make_test(TestMdlPtxFingerprint)
  foreach(mode IN ITEMS invalid-source one-invalid-source)
    add_test(NAME TestMdlPtxFingerprintRejects-${mode}
      COMMAND TestMdlPtxFingerprint --${mode})
    set_tests_properties(TestMdlPtxFingerprintRejects-${mode} PROPERTIES
      WILL_FAIL TRUE TIMEOUT 120)
  endforeach()
  make_test(TestMdlSerialCompile)
  target_link_libraries(TestMdlSerialCompile PRIVATE
    libmdl fmt::fmt nonstd::scope-lite)
  set_tests_properties(TestMdlSerialCompile PROPERTIES
    ENVIRONMENT "VISRTX_MDL_COMPILE_THREADS=1" TIMEOUT 120)
  make_test(TestEmissiveMaterialParity)
  # Silent-miscompile gate (ADR 0009): the same scene must produce byte-identical
  # PTX whether compiled serially or across the worker pool. Runs the fingerprint
  # binary under 1 and 8 compile threads and asserts the printed hashes match.
  add_test(NAME TestMdlPtxIdentity COMMAND ${CMAKE_COMMAND}
    -DTEST_BIN=$<TARGET_FILE:TestMdlPtxFingerprint>
    -P ${CMAKE_CURRENT_LIST_DIR}/mdl_ptx_identity.cmake)
endif()
make_test(TestEmissiveInteractive)
make_test(TestEmissiveIncremental)


make_test(TestEmissiveInstanceRotatedScale)

if(VISRTX_ENABLE_MATERIALX_SUPPORT)
  make_test(TestMaterialXStandardSurfaceInline)
  make_test(TestMaterialXSearchPaths)
  target_compile_definitions(TestMaterialXSearchPaths PRIVATE
    MATERIALX_LIBRARIES_DIR="${MATERIALX_LIBRARIES_DIR}")
  make_test(TestMaterialXRootChangeTexture)
  target_compile_definitions(TestMaterialXRootChangeTexture PRIVATE
    MATERIALX_LIBRARIES_DIR="${MATERIALX_LIBRARIES_DIR}")
  make_test(TestMaterialXOutageRecovery)
  target_compile_definitions(TestMaterialXOutageRecovery PRIVATE
    MATERIALX_LIBRARIES_DIR="${MATERIALX_LIBRARIES_DIR}")
  make_test(TestMaterialXNoDeviceCommit)
  make_test(TestMaterialXParamOverride)
  target_compile_definitions(TestMaterialXParamOverride PRIVATE
    MATERIALX_TEST_DATA_DIR="${CMAKE_CURRENT_LIST_DIR}/data")
  make_test(TestMaterialXMaterial)
  # USE_MATERIALX gates the VISRTX_MATERIAL_MATERIALX flag in
  # fillExtensionStruct (visrtx_extensions.h); the test asserts the device
  # advertises it, so it must be compiled MaterialX-aware.
  target_compile_definitions(TestMaterialXMaterial PRIVATE
    USE_MATERIALX
    MATERIALX_TEST_DATA_DIR="${CMAKE_CURRENT_LIST_DIR}/data")
  make_test(TestMaterialXMultiAndConcurrent)
  target_compile_definitions(TestMaterialXMultiAndConcurrent PRIVATE
    MATERIALX_TEST_DATA_DIR="${CMAKE_CURRENT_LIST_DIR}/data")
  make_test(TestMaterialXTranscoder)
  make_test(TestMaterialXRender)
  make_test(TestMaterialXTexturedInput)
  make_test(TestMaterialXWiredTexture)
  make_test(TestMaterialXNodeDefAutoInstantiate)
  target_compile_definitions(TestMaterialXNodeDefAutoInstantiate PRIVATE
    MATERIALX_LIBRARIES_DIR="${MATERIALX_LIBRARIES_DIR}")
  target_compile_definitions(TestMaterialXRender PRIVATE
    MATERIALX_TEST_DATA_DIR="${CMAKE_CURRENT_LIST_DIR}/data")
  # Compile Transcoder.cpp directly INTO the test. anari_library_visrtx is built
  # SHARED with hidden visibility (device/CMakeLists.txt CMAKE_CXX_VISIBILITY_PRESET
  # hidden), and the transcoder free functions carry no VISRTX_DEVICE_INTERFACE
  # export macro, so they are NOT linkable from the shared object. Building the .cpp
  # into the test keeps it self-contained.
  target_sources(TestMaterialXTranscoder PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../../../device/materialx/Transcoder.cpp)
  target_compile_definitions(TestMaterialXTranscoder PRIVATE
    MATERIALX_TEST_DATA_DIR="${CMAKE_CURRENT_LIST_DIR}/data"
    MATERIALX_LIBRARIES_DIR="${MATERIALX_LIBRARIES_DIR}")
  target_link_libraries(TestMaterialXTranscoder PRIVATE
    MaterialXCore MaterialXFormat MaterialXGenShader MaterialXGenMdl
    nonstd::span-lite)
  target_include_directories(TestMaterialXTranscoder PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../../../device)
  make_test(TestMaterialXAutoInstTexture)
  target_compile_definitions(TestMaterialXAutoInstTexture PRIVATE
    MATERIALX_LIBRARIES_DIR="${MATERIALX_LIBRARIES_DIR}")
  make_test(TestMaterialXInlineDocument)
  make_test(TestEmissiveMaterialXParity)
endif()
