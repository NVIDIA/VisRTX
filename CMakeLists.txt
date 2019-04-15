cmake_minimum_required(VERSION 3.10)

set(VISRTX_VERSION_MAJOR 0)
set(VISRTX_VERSION_MINOR 1)
set(VISRTX_VERSION_PATCH 5)

set(VISRTX_VERSION ${VISRTX_VERSION_MAJOR}.${VISRTX_VERSION_MINOR}.${VISRTX_VERSION_PATCH})


project(VisRTX VERSION ${VISRTX_VERSION} LANGUAGES C CXX CUDA)

# Set a default build type if none was specified
set(default_build_type "Release")
if(EXISTS "${CMAKE_SOURCE_DIR}/.git")
    set(default_build_type "Debug")
endif()
 
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
    set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()


# Language features
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)


# Dependencies
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

if(NOT WIN32)
    set(OpenGL_GL_PREFERENCE GLVND)
endif()

find_package(CUDA 8.0 REQUIRED) # Still required even with CUDA language support so we can link against CUDA from a cpp-only target
find_package(OptiX REQUIRED)
find_package(MDL REQUIRED)
find_package(OpenGL REQUIRED)
find_package(OpenMP)


# Usage report verbosity
set(VISRTX_USAGE_REPORT_VERBOSITY "0" CACHE STRING "Verbosity of OptiX usage report (0 disables reporting).")
set_property(CACHE VISRTX_USAGE_REPORT_VERBOSITY PROPERTY STRINGS 0 1 2 3)

option(VISRTX_USE_DEBUG_EXCEPTIONS "Enables advanced exception handling and error checking for debugging purposes." OFF)


# Compile pathtracer kernels to PTX
if (VISRTX_USE_DEBUG_EXCEPTIONS)
    set(CMAKE_CUDA_FLAGS "-arch sm_30 --use_fast_math --keep-device-functions -lineinfo")
else()
    set(CMAKE_CUDA_FLAGS "-arch sm_30 --use_fast_math --keep-device-functions")
endif()

add_library(PathtracerPTX OBJECT 
    src/Pathtracer/TriangleMesh.cu
    src/Pathtracer/Sphere.cu
    src/Pathtracer/Cylinder.cu
    src/Pathtracer/Disk.cu
    src/Pathtracer/Pathtracer.cu
    src/MDL/Textures.cu
)
set_property(TARGET PathtracerPTX PROPERTY CUDA_PTX_COMPILATION ON)
target_include_directories(PathtracerPTX PUBLIC ${OptiX_INCLUDE} ${MDL_INCLUDE_DIR})
target_include_directories(PathtracerPTX PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/include)


# Retrieve bin2c util
get_filename_component(cuda_compiler_bin "${CMAKE_CUDA_COMPILER}" DIRECTORY)
find_program(bin_to_c
    NAMES bin2c
    PATHS ${cuda_compiler_bin}
    )
if(NOT bin_to_c)
    message(FATAL_ERROR
        "bin2c not found:\n"
        "  CMAKE_CUDA_COMPILER='${CMAKE_CUDA_COMPILER}'\n"
        "  cuda_compiler_bin='${cuda_compiler_bin}'\n"
    )
endif()


# Embed PTX in header
set(headerPathPTX ${CMAKE_CURRENT_BINARY_DIR}/include/PTX.h)
add_custom_command(
    OUTPUT "${headerPathPTX}"
    COMMAND ${CMAKE_COMMAND}
    "-DBIN_TO_C_COMMAND=${bin_to_c}"
    "-DOBJECTS=$<TARGET_OBJECTS:PathtracerPTX>"
    "-DOUTPUT=${headerPathPTX}"
    -P ${PROJECT_SOURCE_DIR}/cmake/bin2c_wrapper.cmake
    VERBATIM
    DEPENDS $<TARGET_OBJECTS:PathtracerPTX>
    COMMENT "Converting PTX files to a C header"
)


# Headers
configure_file(${PROJECT_SOURCE_DIR}/src/VisRTX.h.in ${CMAKE_CURRENT_BINARY_DIR}/include/VisRTX.h)
configure_file(${PROJECT_SOURCE_DIR}/src/Config.h.in ${CMAKE_CURRENT_BINARY_DIR}/include/Config.h)


# glad OpenGL loader
add_library(Glad STATIC libs/glad/src/glad.c)
target_include_directories(Glad PUBLIC libs/glad/include)
target_link_libraries(Glad PUBLIC ${CMAKE_DL_LIBS})
set_property(TARGET Glad PROPERTY POSITION_INDEPENDENT_CODE ON)


# VisRTX shared library
 add_library(VisRTX SHARED 
    src/VisRTX.cpp
    src/OptiX.cpp
	src/ProgramLoader.cpp
    src/Renderer.cpp
    src/FrameBuffer.cpp
    src/Camera.cpp
    src/Light.cpp
    src/Geometry.cpp
    src/Material.cpp
    src/Texture.cpp
    src/Material.cpp
    src/Model.cpp
    src/MDL/MDL.cpp
    ${headerPathPTX}
)

add_dependencies(VisRTX PathtracerPTX)

target_compile_definitions(VisRTX PRIVATE "_CRT_SECURE_NO_WARNINGS")

target_include_directories(VisRTX PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/include>
)    
target_include_directories(VisRTX PRIVATE 
    ${CMAKE_CURRENT_BINARY_DIR}/include 
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} 
    ${OptiX_INCLUDE}
    ${MDL_INCLUDE_DIR}
)

target_link_libraries(VisRTX PRIVATE ${CUDA_LIBRARIES} ${optix_LIBRARY} Glad)
if(WIN32)    
    target_include_directories(VisRTX PRIVATE ${OPENGL_INCLUDE_DIR})
    target_link_libraries(VisRTX PRIVATE ${OPENGL_gl_LIBRARY})
else()
    target_include_directories(VisRTX PRIVATE ${OPENGL_INCLUDE_DIR} ${OPENGL_EGL_INCLUDE_DIRS})
    target_link_libraries(VisRTX PRIVATE ${OPENGL_opengl_LIBRARY} ${OPENGL_egl_LIBRARY} ${OPENGL_glx_LIBRARY})
endif()
if(OPENMP_FOUND)
    target_link_libraries(VisRTX PRIVATE OpenMP::OpenMP_CXX)
endif()

# Header-only target for dynamic loading use case
add_library(VisRTX_DynLoad INTERFACE)
target_include_directories(VisRTX_DynLoad INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/include>
)   
target_link_libraries(VisRTX_DynLoad INTERFACE ${CMAKE_DL_LIBS})


# Install / export
include(GNUInstallDirs)
set_target_properties(VisRTX PROPERTIES VERSION ${VISRTX_VERSION})

install(TARGETS VisRTX VisRTX_DynLoad EXPORT VisRTXConfig
    ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(FILES 
    ${CMAKE_CURRENT_BINARY_DIR}/include/VisRTX.h 
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT VisRTXConfig DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/VisRTX-${VISRTX_VERSION})
export(TARGETS VisRTX VisRTX_DynLoad FILE VisRTXConfig.cmake)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(VisRTXConfigVersion.cmake VERSION ${VISRTX_VERSION} COMPATIBILITY AnyNewerVersion)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/VisRTXConfigVersion.cmake" DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/VisRTX-${VISRTX_VERSION})


# Sample app
option(VISRTX_BUILD_SAMPLE "Builds the sample applications." ON)
if(VISRTX_BUILD_SAMPLE)
    
    option(VISRTX_SAMPLE_USE_DYNAMIC_LOADING "Uses dynamic loading instead of dynamic linking to access VisRTX in the samples." OFF)
    option(VISRTX_SAMPLE_WITH_GLFW "Enables GLFW-based OpenGL windows in the samples." ON)    
    option(VISRTX_SAMPLE_WITH_EGL "Enables EGL-based OpenGL offscreen interop in the samples (Linux only)." ON)

    if(WIN32)
        set(VISRTX_SAMPLE_WITH_EGL OFF)
    endif()

    set(CMAKE_CXX_STANDARD 11)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)

    if(VISRTX_SAMPLE_WITH_GLFW)
        # GLFW
        set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
        set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
        set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
        add_subdirectory(sample/libs/glfw-3.2.1 EXCLUDE_FROM_ALL)

        # ImGUI
        add_definitions(-DIMGUI_IMPL_OPENGL_LOADER_GLAD)

        include_directories(sample/libs/ImGui-1.67-WIP)
        include_directories(sample/libs/ImGui-1.67-WIP/examples)
        add_library(ImGui STATIC 
            sample/libs/ImGui-1.67-WIP/imgui_demo.cpp 
            sample/libs/ImGui-1.67-WIP/imgui_draw.cpp 
            sample/libs/ImGui-1.67-WIP/imgui_widgets.cpp 
            sample/libs/ImGui-1.67-WIP/imgui.cpp
            sample/libs/ImGui-1.67-WIP/examples/imgui_impl_glfw.cpp 
            sample/libs/ImGui-1.67-WIP/examples/imgui_impl_opengl3.cpp
        )
        target_link_libraries(ImGui glfw Glad)
        target_include_directories(ImGui PRIVATE ${OPENGL_INCLUDE_DIR})
        if(WIN32)            
            target_link_libraries(ImGui ${OPENGL_gl_LIBRARY})
        else()
            target_link_libraries(ImGui ${OPENGL_opengl_LIBRARY})
        endif()

        list(APPEND VISRTX_SAMPLE_LIBS ImGui)
        list(APPEND VISRTX_SAMPLE_DEFINES "VISRTX_SAMPLE_WITH_GLFW=1")
    endif()

    if(VISRTX_SAMPLE_WITH_EGL)
        list(APPEND VISRTX_SAMPLE_INCLUDES ${OPENGL_EGL_INCLUDE_DIRS})
        list(APPEND VISRTX_SAMPLE_LIBS Glad ${OPENGL_egl_LIBRARY})
        list(APPEND VISRTX_SAMPLE_DEFINES "VISRTX_SAMPLE_WITH_EGL=1")
    endif()

    # Tinyobjloader
    include_directories(sample/libs/tinyobjloader)

    # Embed ospray.mdl in header
    set(OSPRayMDLFiles ${PROJECT_SOURCE_DIR}/sample/data/OSPRay.mdl)
    set(headerPathOSPRayMDL ${CMAKE_CURRENT_BINARY_DIR}/include/OSPRayMDL.h)
    add_custom_command(
      OUTPUT "${headerPathOSPRayMDL}"
      COMMAND ${CMAKE_COMMAND}
        "-DBIN_TO_C_COMMAND=${bin_to_c}"
        "-DOBJECTS=${OSPRayMDLFiles}"
        "-DOUTPUT=${headerPathOSPRayMDL}"
        -P ${PROJECT_SOURCE_DIR}/cmake/bin2c_wrapper.cmake
      VERBATIM
      COMMENT "Converting OSPRay MDL materials to a C header"
    )

    # Copy sample data to build folder
    file(COPY ${PROJECT_SOURCE_DIR}/sample/data/cow.obj DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
	file(COPY ${PROJECT_SOURCE_DIR}/sample/data/NVIDIA_Logo.jpg DESTINATION ${CMAKE_CURRENT_BINARY_DIR})	

    # Samples        
    if (VISRTX_SAMPLE_USE_DYNAMIC_LOADING)
        # Dynamic loading
        list(APPEND VISRTX_SAMPLE_LIBS VisRTX_DynLoad)
        list(APPEND VISRTX_SAMPLE_DEFINES "VISRTX_DYNLOAD=1")    
    else()
        # Dynamic linking
        list(APPEND VISRTX_SAMPLE_LIBS VisRTX)
    endif()

    add_executable(visRtxSampleBasic sample/SampleBasic.cpp ${headerPathOSPRayMDL})
    add_executable(visRtxSampleMDL sample/SampleMDL.cpp ${headerPathOSPRayMDL})
    add_executable(visRtxSampleLightTest sample/SampleLightTest.cpp ${headerPathOSPRayMDL})
    add_executable(visRtxSampleMeshViewer sample/SampleMeshViewer.cpp ${headerPathOSPRayMDL})
	add_executable(visRtxSampleVolume sample/SampleVolume.cpp ${headerPathOSPRayMDL})

    target_compile_definitions(visRtxSampleBasic PRIVATE ${VISRTX_SAMPLE_DEFINES})
    target_compile_definitions(visRtxSampleMDL PRIVATE ${VISRTX_SAMPLE_DEFINES})
    target_compile_definitions(visRtxSampleLightTest PRIVATE ${VISRTX_SAMPLE_DEFINES})
    target_compile_definitions(visRtxSampleMeshViewer PRIVATE ${VISRTX_SAMPLE_DEFINES})
	target_compile_definitions(visRtxSampleVolume PRIVATE ${VISRTX_SAMPLE_DEFINES})

    target_include_directories(visRtxSampleBasic PRIVATE ${VISRTX_SAMPLE_INCLUDES})
    target_include_directories(visRtxSampleMDL PRIVATE ${VISRTX_SAMPLE_INCLUDES})
    target_include_directories(visRtxSampleLightTest PRIVATE ${VISRTX_SAMPLE_INCLUDES})
    target_include_directories(visRtxSampleMeshViewer PRIVATE ${VISRTX_SAMPLE_INCLUDES})
	target_include_directories(visRtxSampleVolume PRIVATE ${VISRTX_SAMPLE_INCLUDES})

    target_link_libraries(visRtxSampleBasic ${VISRTX_SAMPLE_LIBS})        
    target_link_libraries(visRtxSampleMDL ${VISRTX_SAMPLE_LIBS})        
    target_link_libraries(visRtxSampleLightTest ${VISRTX_SAMPLE_LIBS})        
    target_link_libraries(visRtxSampleMeshViewer ${VISRTX_SAMPLE_LIBS})
	target_link_libraries(visRtxSampleVolume ${VISRTX_SAMPLE_LIBS})
endif()
