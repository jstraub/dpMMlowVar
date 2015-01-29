project(DpGMMs)
cmake_minimum_required(VERSION 2.8.1)

message("${CMAKE_MODULE_PATH}")

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

#IF(BOOST_OLD)
#  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DBOOST_OLD")
#  include_directories(
#    ./include
#    ./cuda
#    ${EIGEN3_INCLUDE_DIR}
##    /usr/include
#    #/usr/include/eigen3
#    ${Boost_INCLUDE_DIRS}
#    ${CUDA_TOOLKIT_INCLUDE}
#    )
#  link_directories(
#    ${Boost_LIBRARY_DIRS}
##    $ENV{WORKSPACE_HOME}/3rdparty/lib
##    /usr/lib
#    ${CUDA_LIBRARIES}
##    /usr/local/cuda-6.5/targets/x86_64-linux/lib/
#    )
#  # no sampling from chi_squared
#  cuda_add_library(dpMMlowVar SHARED
#    ./cuda/vectorSum_kernel.cu
#    ./cuda/ddpvMFlabels_kernel.cu
#    ./cuda/copy_kernel.cu
#    ./cuda/choiceMult_kernel.cu
#    ./cuda/labelMap_kernel.cu
#    ./cuda/ddpLabelsSpecial_kernel.cu
#    )
#  target_link_libraries(dpMMlowVar
#    boost_system
#    boost_random
#    boost_program_options
#    )
#ELSE(BOOST_OLD)
  include_directories(
    ./include
    ./cuda
    ${EIGEN3_INCLUDE_DIR}
    /usr/include
    #/usr/include/eigen3
#    $ENV{WORKSPACE_HOME}/3rdparty/include
    ${Boost_INCLUDE_DIRS}
    ${CUDA_TOOLKIT_INCLUDE}
    )
  link_directories(
    ${Boost_LIBRARY_DIRS}
    ${CUDA_LIBRARIES}
#    $ENV{WORKSPACE_HOME}/3rdparty/lib
    )
  cuda_add_library(dpMMlowVar SHARED
    ./cuda/vectorSum_kernel.cu
    ./cuda/ddpvMFlabels_kernel.cu
    ./cuda/ddpLabels_kernel.cu
    ./cuda/copy_kernel.cu
    ./cuda/choiceMult_kernel.cu
    ./cuda/labelMap_kernel.cu
    ./cuda/ddpLabelsSpecial_kernel.cu
    )
  target_link_libraries(dpMMlowVar
    boost_system
    boost_random
    boost_program_options
    )
#
#ENDIF(BOOST_OLD)

add_executable(testSO3 ./test/SO3.cpp
  )
target_link_libraries(testSO3
  boost_system
  boost_random
  boost_unit_test_framework
  )

add_executable(testClusterer ./test/kmeans.cpp
  )
target_link_libraries(testClusterer
  boost_system
  boost_random
  boost_unit_test_framework
  )

add_executable(ddpImageCluster ./src/ddpImageCluster.cpp
  )
target_link_libraries(ddpImageCluster
  dpMMlowVar
  boost_system
  boost_random
  boost_program_options
  opencv_core
  opencv_imgproc
  opencv_highgui
  )

add_executable(ddpSphereCluster ./src/ddpSphereCluster.cpp
  )
target_link_libraries(ddpSphereCluster
  dpMMlowVar
  boost_system
  boost_random
  boost_program_options
  opencv_core
  opencv_imgproc
  opencv_highgui
  )

INSTALL(TARGETS dpMMlowVar ddpImageCluster #dpStickGMM dpDirMM
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  # ARCHIVE DESTINATION lib
  COMPONENT compLib
  )

install(DIRECTORY src/ DESTINATION src/                                 
  FILES_MATCHING PATTERN "*.cpp")                                         
install(DIRECTORY include/ DESTINATION include/                                 
  FILES_MATCHING PATTERN "*.h")                                         
install(DIRECTORY cuda/ DESTINATION include/                                 
  FILES_MATCHING PATTERN "*.h")                                         
install(DIRECTORY include/ DESTINATION include/                                 
  FILES_MATCHING PATTERN "*.hpp") 
