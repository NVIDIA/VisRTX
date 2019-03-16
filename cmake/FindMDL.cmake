#
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

# Try to find MDL SDK library and include dir.
# Once done this will define
#
# MDL_FOUND        - true if MDL has been found
# MDL_INCLUDE_DIR  - where the mi/mdl_sdk.h can be found
# MDL_LIBRARIES    - all MDL related libraries: libmdl_sdk, nv_freeimage, dds

set(MDL_INSTALL_DIR "" CACHE PATH "Path to MDL SDK installed location.")

if (NOT MDL_INCLUDE_DIR)
    find_path(MDL_INCLUDE_DIR mi/mdl_sdk.h PATHS ${MDL_INSTALL_DIR}/include NO_DEFAULT_PATH)
endif()


macro(MDL_find_library name)
    if(WIN32)
        set(MDL_LIB_DIR ${MDL_INSTALL_DIR}/nt-x86-64/lib)

        find_file(${name}_LIBRARY
            NAMES ${name}.dll
            PATHS ${MDL_LIB_DIR}
            NO_DEFAULT_PATH
        )
        find_file(${name}_LIBRARY
            NAMES ${name}.dll
        )
    else()
        set(MDL_LIB_DIR ${MDL_INSTALL_DIR}/linux-x86-64/lib)

        find_library(${name}_LIBRARY
            NAMES ${name}.so
            PATHS ${MDL_LIB_DIR}
            NO_DEFAULT_PATH
        )
        find_library(${name}_LIBRARY
            NAMES ${name}
        )
    endif()
endmacro()


MDL_find_library(libmdl_sdk)
MDL_find_library(dds)
MDL_find_library(nv_freeimage)

set(MDL_LIBRARIES ${libmdl_sdk_LIBRARY} ${dds_LIBRARY} ${nv_freeimage_LIBRARY})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MDL DEFAULT_MSG MDL_LIBRARIES MDL_INCLUDE_DIR)
