## Copyright 2024-2026 NVIDIA Corporation
## SPDX-License-Identifier: Apache-2.0

add_subdirectory(animatedParticles)
add_subdirectory(animatedVolume)
add_subdirectory(arrayInstancing)
add_subdirectory(manualAccumulationReset)
if (TSD_USE_NETWORKING)
  add_subdirectory(network)
endif()
option(BUILD_VISKORES_DEMO "Build Viskores demo viewer" OFF)
if (BUILD_VISKORES_DEMO)
  add_subdirectory(viskores)
endif()
