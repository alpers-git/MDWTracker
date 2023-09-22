cmake_policy(SET CMP0048 NEW)
project(rqs-renderer VERSION 0.0.1 LANGUAGES C CXX CUDA)

cmake_minimum_required(VERSION 3.16)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


if(${CMAKE_VERSION} VERSION_GREATER_EQUAL 3.18)
  cmake_policy(SET CMP0104 NEW)
endif()

#--------------------- owl --------------------------
add_subdirectory(submodules/owl EXCLUDE_FROM_ALL)
#----------------------------------------------------

#------------------ device code ---------------------

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
  owl::owl)
#----------------------------------------------------