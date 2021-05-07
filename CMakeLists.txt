cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(train_unet_darknet)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_VERBOSE_MAKEFILE ON)

if("${CUSTOM_TORCH_BUILD_PATH}" STREQUAL "")
   find_package(Torch REQUIRED)
else()
   find_package(Torch REQUIRED PATHS ${CUSTOM_TORCH_BUILD_PATH} NO_DEFAULT_PATH)
endif()

if("${CUSTOM_OPENCV_BUILD_PATH}" STREQUAL "")
   find_package(OpenCV REQUIRED)
else()
   find_package(OpenCV REQUIRED PATHS ${CUSTOM_OPENCV_BUILD_PATH} NO_DEFAULT_PATH)
endif()

find_package(Boost REQUIRED)

file(GLOB_RECURSE PUBLIC_INCLUDES include/*.hpp)

include_directories(
   include
   ${Boost_INCLUDE_DIRS}
   ${OpenCV_INCLUDE_DIRS})

add_library(${PROJECT_NAME}2dl
   ${PUBLIC_INCLUDES}
   src/train_unet2d.cpp
   src/UNetDataset.cpp
   src/DarknetParser.cpp)

add_executable(${PROJECT_NAME}2d
   src/main.cpp)

target_compile_features(${PROJECT_NAME}2d PUBLIC cxx_range_for)

set(STD_FILESYSTEM)
if (NOT APPLE AND NOT MSVC)
   set(STD_FILESYSTEM stdc++fs)
endif()

if(NOT APPLE AND NOT MSVC)
   set(TIFF_LIB ${TIFF_LIBRARY})
endif()

target_link_libraries(${PROJECT_NAME}2dl
   PRIVATE
   ${TORCH_LIBRARIES}
   ${OpenCV_LIBS}
   ${TIFF_LIB}
   ${STD_FILESYSTEM})

target_link_libraries(${PROJECT_NAME}2d
   ${PROJECT_NAME}2dl)

add_executable(grubcut
   src/grubcut.cpp)

target_link_libraries(grubcut
   ${OpenCV_LIBS}
   ${TIFF_LIB}
   ${STD_FILESYSTEM})

# Tanks windows for this unneeded workaround
if(MSVC)
   file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
   message(STATUS TORCH_DLLS=${TORCH_DLLS})
   message(STATUS DESTINATION=${CMAKE_BINARY_DIR})
   add_custom_command(TARGET ${PROJECT_NAME}2d
      POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
      ${TORCH_DLLS}
      ${CMAKE_BINARY_DIR})
endif(MSVC)