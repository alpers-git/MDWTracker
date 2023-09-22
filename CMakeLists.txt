cmake_policy(SET CMP0048 NEW)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

project(rqs-renderer VERSION 0.0.1 LANGUAGES C CXX CUDA)

cmake_minimum_required(VERSION 3.16)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


if(${CMAKE_VERSION} VERSION_GREATER_EQUAL 3.18)
cmake_policy(SET CMP0104 NEW)
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

find_package(OpenGL REQUIRED)

#--------------------GLFW---------------------#
set(GLFW_BUILD_DOCS OFF CACHE STRING "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE STRING "" FORCE)
set(GLFW_BUILD_TESTS    OFF CACHE STRING "" FORCE)
set(GLFW_INSTALL        OFF CACHE STRING "" FORCE)
add_subdirectory(submodules/glfw)
set_property(TARGET glfw PROPERTY FOLDER "submodules/GLFW3" )
#----------------------------------------------#

#--------------------- owl --------------------------
add_subdirectory(submodules/owl EXCLUDE_FROM_ALL)
#----------------------------------------------------

#------------------ device code ---------------------
include(embed_ptx)

embed_ptx(
  OUTPUT_TARGET
    deviceCode_ptx
  EMBEDDED_SYMBOL_NAMES
    deviceCode_ptx
  PTX_LINK_LIBRARIES
    owl::owl
  SOURCES
    deviceCode.cu
)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --generate-line-info")

#-------------------- viewer ------------------------
add_executable(rqsViewer
  viewer.cpp
  renderer.h
  renderer.cpp
  stb_image_write.h)

target_link_libraries(rqsViewer
  PRIVATE 
  deviceCode_ptx
  PUBLIC
  ${OPENGL_LIBRARIES}
  owl::owl
  glfw)
#----------------------------------------------------