project(DpGMMs)
cmake_minimum_required(VERSION 2.8.1)

if(COMMAND cmake_policy)
  cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

set(CMAKE_COLOR_MAKEFILE ON)
set(CMAKE_BUILD_TYPE Debug) # Release; debug; ...
set(CMAKE_BUILD_TYPE Release) # Release; debug; ...
  #ENDIF()

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# add flags depending on build type
set(CMAKE_CXX_FLAGS_RELEASE "-g -Wall -fopenmp -Wno-deprecated -O2 -finline-functions -DNDEBUG -DCUDA") # -DNDEBUG
set(CMAKE_CXX_FLAGS_DEBUG "-g -Wall -Wno-deprecated -DCUDA")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=\"sm_35,compute_35\")

set(CMAKE_EXPORT_COMPILE_COMMANDS 1)


find_package (Eigen3 REQUIRED)
find_package (Boost REQUIRED)
find_package (OpenMP REQUIRED)
find_package (CUDA REQUIRED)

message(${EIGEN3_INCLUDE_DIR})

IF(BOOST_OLD)
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DBOOST_OLD")
  include_directories(
    ./include
    ./cuda
    ${EIGEN3_INCLUDE_DIR}
    /usr/include
    #/usr/include/eigen3
    )
  link_directories(
#    $ENV{WORKSPACE_HOME}/3rdparty/lib
    /usr/lib
    )
  # no sampling from chi_squared
  cuda_add_library(dpMMlowVar SHARED
    ./src/sphere.cpp
    ./cuda/vectorSum_kernel.cu
    ./cuda/ddpvMFlabels_kernel.cu
    ./cuda/copy_kernel.cu
    )
  target_link_libraries(dpMMlowVar
    boost_system
    boost_random
    boost_program_options
    )
ELSE(BOOST_OLD)
  include_directories(
    ./include
    ./cuda
    ${EIGEN3_INCLUDE_DIR}
    /usr/include
    #/usr/include/eigen3
    $ENV{WORKSPACE_HOME}/3rdparty/include
    )
  link_directories(
    $ENV{WORKSPACE_HOME}/3rdparty/lib
    )
  cuda_add_library(dpMMlowVar SHARED
    ./src/sphere.cpp
    ./cuda/vectorSum_kernel.cu
    ./cuda/ddpvMFlabels_kernel.cu
    ./cuda/copy_kernel.cu
    )
  target_link_libraries(dpMMlowVar
    boost_system
    boost_random
    boost_program_options
    )

#  cuda_add_executable(testSPKM ./test/spkm.cpp
#    ./src/cat.cpp
#    ./src/mult.cpp
#    ./src/dir.cpp
#    ./src/sphere.cpp
#    ./src/normal.cpp
#    )
#  target_link_libraries(testSPKM
#    boost_system
#    boost_random
#    boost_unit_test_framework
#    )
#
#  cuda_add_executable(testvMFmeans ./test/vMFmeans.cpp
#    ./src/cat.cpp
#    ./src/mult.cpp
#    ./src/dir.cpp
#    ./src/sphere.cpp
#    ./src/normal.cpp
#    ./cuda/vectorSum_kernel.cu
#    ./cuda/ddpvMFlabels_kernel.cu
#    ./cuda/copy_kernel.cu
#    )
#  target_link_libraries(testvMFmeans
#    boost_system
#    boost_random
#    boost_unit_test_framework
#    )

ENDIF(BOOST_OLD)



INSTALL(TARGETS dpMMlowVar #dpStickGMM dpDirMM
  #  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  # ARCHIVE DESTINATION lib
  COMPONENT compLib
  )

install(DIRECTORY include/ DESTINATION include/                                 
  FILES_MATCHING PATTERN "*.h")                                         
install(DIRECTORY cuda/ DESTINATION include/                                 
  FILES_MATCHING PATTERN "*.h")                                         
install(DIRECTORY include/ DESTINATION include/                                 
  FILES_MATCHING PATTERN "*.hpp") 
