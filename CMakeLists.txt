cmake_policy(SET CMP0048 NEW)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

project(MDWTracker VERSION 0.0.1 LANGUAGES C CXX CUDA)

cmake_minimum_required(VERSION 3.16)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


if(${CMAKE_VERSION} VERSION_GREATER_EQUAL 3.18)
cmake_policy(SET CMP0104 NEW)
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

#--------------compile options----------------#
# Set the default value to false
# option(OFFLINE_VIEWER "Enable offline viewer" OFF)

# if (OFFLINE_VIEWER)
# add_compile_definitions(OFFLINE_VIEWER=1)
# else()
# add_compile_definitions(OFFLINE_VIEWER=0)
# endif()
#---------------------------------------------#

# if linux
if(UNIX)
  find_package(SDL2 REQUIRED)
endif()
find_package(OpenGL REQUIRED)

#---------------- argparse -------------------#
add_subdirectory(submodules/argparse)
#---------------------------------------------#

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
set(OWL_BUILD_SAMPLES OFF CACHE BOOL "" FORCE)
#----------------------------------------------------

#-------------------- umesh -------------------------
add_subdirectory(submodules/umesh EXCLUDE_FROM_ALL)
add_definitions(${UMESH_DEFINITIONS})
include_directories(submodules/umesh)
set(UMESH_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
#----------------------------------------------------

#-------------------- imgui -------------------------
# Set where the ImGui files are stored
set(IMGUI_PATH  submodules/imgui)

# Compile as static library 
file(GLOB IMGUI_SOURCES ${IMGUI_PATH}/*.cpp)
add_library("ImGui" STATIC ${IMGUI_SOURCES}
            ${IMGUI_PATH}/backends/imgui_impl_glfw.cpp
            ${IMGUI_PATH}/backends/imgui_impl_opengl3.cpp)
target_include_directories("ImGui" PUBLIC ${IMGUI_PATH} ${IMGUI_PATH}/backends glfw)
#link with glfw and opengl
target_link_libraries("ImGui" PUBLIC glfw)
include_directories(submodules/imgui)
#----------------------------------------------------
#--------------------imgui-tfn-----------------------
set(IMGUI_TFN_PATH  submodules/imgui-transfer-function)

# Compile as static library
file(GLOB IMGUI_TFN_SOURCES ${IMGUI_TFN_PATH}/*.cpp ${IMGUI_TFN_PATH}/*.c)
add_library("ImGuiTFN" STATIC ${IMGUI_TFN_SOURCES})
target_include_directories("ImGuiTFN" PUBLIC ${IMGUI_TFN_PATH})
include_directories(submodules/imgui-transfer-function)
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
add_executable(mdwtViewer
  dda.h
  glfwHandler.cpp
  hilbert.h
  renderer.cpp
  renderer.cu
  stb_image_write.h
  rawFile.cpp
  viewer.cpp)

target_link_libraries(mdwtViewer PRIVATE -lGL -lGLX)
target_link_libraries(mdwtViewer
  PRIVATE
  deviceCode_ptx
  argparse
  owl::owl
  PUBLIC
  ${OPENGL_LIBRARIES}
  glfw
  umesh
  ImGui
  ImGuiTFN)
 target_compile_definitions(mdwtViewer PRIVATE OFFLINE_VIEWER=false)
#----------------------------------------------------
#---------------- offline viewer --------------------
add_executable(mdwtOffline
  dda.h
  glfwHandler.cpp
  hilbert.h
  renderer.cpp
  renderer.cu
  stb_image_write.h
  rawFile.cpp
  viewer.cpp)

target_link_libraries(mdwtOffline PRIVATE -lGL -lGLX)
target_link_libraries(mdwtOffline
  PRIVATE
  deviceCode_ptx
  argparse
  owl::owl
  PUBLIC
  ${OPENGL_LIBRARIES}
  glfw
  umesh
  ImGui
  ImGuiTFN)
 target_compile_definitions(mdwtOffline PRIVATE OFFLINE_VIEWER=true)
#----------------------------------------------------

# if windows copy owl.dll to the current build/<build_type> folder
 # Find the directory where owl.dll is located
 if(WIN32)
 foreach(target mdwtViewer mdwtOffline)
   add_custom_command(TARGET ${target} POST_BUILD
     COMMAND ${CMAKE_COMMAND} -E copy_if_different
       $<TARGET_FILE:owl::owl>
       $<TARGET_FILE_DIR:${target}>
     COMMENT "Copying owl.dll to build output directory for ${target}"
   )
 endforeach()
endif()