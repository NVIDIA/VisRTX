# Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

if(VISRTX_ENABLE_MDL_SUPPORT)
    # Stage per file with configure_file, NOT file(COPY): configure_file
    # registers each source as a configure dependency, so a plain build after
    # editing a module re-stages and re-embeds it. file(COPY) staged only at
    # configure time — an edited .mdl kept shipping the STALE embedded copy
    # until the next explicit reconfigure, silently inverting any test verdict
    # on the edit. CONFIGURE_DEPENDS extends the same guarantee to files being
    # added or removed.
    file(GLOB VISRTX_MDL_RESOURCES
        RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        CONFIGURE_DEPENDS
        visrtx/*)
    foreach(resource ${VISRTX_MDL_RESOURCES})
        configure_file(${resource} ${resource} COPYONLY)
    endforeach()
endif()