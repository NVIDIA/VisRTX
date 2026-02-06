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

# Script to embed a resource file as a C++ array
# Usage: cmake -DRESOURCE_FILE=<input> -DSOURCE_FILE=<output> -DVARIABLE_NAME=<name> -P embed_resource.cmake

if(NOT RESOURCE_FILE)
  message(FATAL_ERROR "RESOURCE_FILE not specified")
endif()

if(NOT SOURCE_FILE)
  message(FATAL_ERROR "SOURCE_FILE not specified")
endif()

if(NOT VARIABLE_NAME)
  message(FATAL_ERROR "VARIABLE_NAME not specified")
endif()

# Read the input file as binary
file(READ ${RESOURCE_FILE} file_content HEX)

# Get the file size
file(SIZE ${RESOURCE_FILE} file_size)

# Convert hex string to C array format
string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," array_content ${file_content})
# Remove trailing comma
string(REGEX REPLACE ",$" "" array_content ${array_content})

# Generate the new C++ source content
set(new_content
"// Auto-generated file - do not edit
// Generated from: ${RESOURCE_FILE}

#include <cstddef>

extern \"C\" const unsigned char ${VARIABLE_NAME}[] = {
${array_content}
};

extern \"C\" const std::size_t ${VARIABLE_NAME}_size = ${file_size};
")

# Check if output file exists and read current content
set(write_file TRUE)
if(EXISTS ${SOURCE_FILE})
  file(READ ${SOURCE_FILE} current_content)
  if("${current_content}" STREQUAL "${new_content}")
    set(write_file FALSE)
    message(STATUS "Skipping ${SOURCE_FILE} - content unchanged")
  endif()
endif()

# Only write the file if content has changed
if(write_file)
  file(WRITE ${SOURCE_FILE} "${new_content}")
  message(STATUS "Generated ${SOURCE_FILE} with ${file_size} bytes from ${RESOURCE_FILE}")
endif()
