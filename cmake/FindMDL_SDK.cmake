#Looks for environment variable:
#MDL_SDK_PATH

#Sets the variables:
#MDL_SDK_INCLUDE_DIRS
#MDL_SDK_LIBRARIES
#MDL_SDK_FOUND

set(MDL_SDK_PATH $ENV{MDL_SDK_PATH})

#If there was no environment variable override for the MDL_SDK_PATH
#try finding it inside the local 3rdparty path.
if ("${MDL_SDK_PATH}" STREQUAL "")
  set(MDL_SDK_PATH "${LOCAL_3RDPARTY}/MDL_SDK")
endif()

message(VERBOSE "MDL_SDK_PATH = " "${MDL_SDK_PATH}")

find_path( MDL_SDK_INCLUDE_DIRS "mi/mdl_sdk.h"
  PATHS /usr/include ${MDL_SDK_PATH}/include )

message(VERBOSE "MDL_SDK_INCLUDE_DIRS = " "${MDL_SDK_INCLUDE_DIRS}")

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(MDL_SDK DEFAULT_MSG MDL_SDK_INCLUDE_DIRS)

mark_as_advanced(MDL_SDK_INCLUDE_DIRS)

 message(STATUS "MDL_SDK_FOUND = " "${MDL_SDK_FOUND}")
