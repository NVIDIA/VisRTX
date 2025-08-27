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
