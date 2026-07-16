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

if (WIN32)
  return()
endif()

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

make_test(TestDeviceInit)
make_test(TestEvictGPUArray)
make_test(TestMultiThreadedObjectCreation)
make_test(TestSpheres)
make_test(TestCylinderCaps)
make_test(TestConeApex)
make_test(TestConeCylinderTransforms)
make_test(TestConeCylinderDegenerate)
make_test(TestConeCylinderTransparency)
make_test(TestRotatedSphereShading)
make_test(TestIsosurfaceRamp)
make_test(TestPrimitiveInterior)
make_test(TestGroundSphereShadows)
make_test(TestPowerLightPick)
make_test(TestPowerLightPickEnv)
make_test(TestEmissiveGeometryLight)
make_test(TestEmissiveSphereLight)
make_test(TestEmissiveInstanceScale)
make_test(TestEmissiveSphereInstanceScale)
make_test(TestEmissiveCylinderConeLight)
# These interleave attribute-driven emission cases, and MDL does not support
# attributes — native-PBR only. (Textured emission itself works on the MDL
# wrapper too; TestEmissiveMaterialParity covers it in that config.)
if (NOT VISRTX_USE_MDL_FOR_PHYSICALLY_BASED)
  make_test(TestEmissiveTexturedLight)
  make_test(TestEmissiveInstanceAttribute)
endif()
make_test(TestPbrSpecularDefault)
make_test(TestSamplerAlphaFill)
make_test(TestEmissionRadianceScale)
make_test(TestEmissiveRecommit)
# The raw `mdl` material type and the `physicallyBasedMDL` subtype only exist
# with MDL support compiled in.
if (VISRTX_ENABLE_MDL_SUPPORT)
  target_compile_definitions(
    TestEmissiveRecommit PRIVATE VISRTX_TEST_MDL_WRAPPER)
  # Adds the physicallyBasedMDL leg, so the module's specular default is
  # exercised even when the physicallyBased backend is native CUDA.
  target_compile_definitions(
    TestPbrSpecularDefault PRIVATE VISRTX_TEST_MDL_WRAPPER)
  make_test(TestEmissiveMdlLight)
  make_test(TestEmissionDescriptorPolicy)
  make_test(TestMdlPipelineRebuild)
  make_test(TestMdlMaterialRegistryRelease)
  make_test(TestEmissiveMaterialParity)
endif()
make_test(TestEmissiveInteractive)
make_test(TestEmissiveIncremental)


make_test(TestEmissiveInstanceRotatedScale)
