project(DpGMMs)
cmake_minimum_required(VERSION 2.8.1)

set(POD_NAME dpMMlowVar)
include(cmake/pods.cmake)

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

include_directories(
  ./include
  ${EIGEN3_INCLUDE_DIR}
  ${Boost_INCLUDE_DIRS}
  ${CUDA_TOOLKIT_INCLUDE}
  )
link_directories(
  ${Boost_LIBRARY_DIRS}
  ${CUDA_LIBRARIES}
  )
cuda_add_library(dpMMlowVar SHARED
#  ./cuda/vectorSum_kernel.cu
  ./cuda/ddpvMFlabels_kernel.cu
  ./cuda/ddpLabels_kernel.cu
#  ./cuda/copy_kernel.cu
#  ./cuda/choiceMult_kernel.cu
#  ./cuda/labelMap_kernel.cu
  ./cuda/ddpLabelsSpecial_kernel.cu
  ./cuda/ddpvMFlabelsSpecial_kernel.cu
  ./cuda/spkmLabels_kernel.cu
  ./cuda/kmeansLabels_kernel.cu
  )
target_link_libraries(dpMMlowVar
  boost_system
  boost_random
  boost_program_options
  )

add_executable(testSO3 ./test/SO3.cpp
  )
target_link_libraries(testSO3
  boost_system
  boost_random
  boost_unit_test_framework
  )

add_executable(testSpkmCUDA ./test/spkmCUDA.cpp
  )
target_link_libraries(testSpkmCUDA
  jsCore dpMMlowVar
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

add_executable(dpMMlowVarCluster ./src/dpMMlowVarCluster.cpp
  )
target_link_libraries(dpMMlowVarCluster
  jsCore dpMMlowVar
  boost_system
  boost_random
  boost_program_options
  )

add_executable(ddpImageCluster ./src/ddpImageCluster.cpp
  )
target_link_libraries(ddpImageCluster
  jsCore dpMMlowVar 
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
  jsCore dpMMlowVar
  boost_system
  boost_random
  boost_program_options
  opencv_core
  opencv_imgproc
  opencv_highgui
  )

#install(DIRECTORY include/dpMMlowVar/ DESTINATION include/dpMMlowVar/
#  FILES_MATCHING PATTERN "*.h")
##install(DIRECTORY cuda/ DESTINATION include/dpMMlowVar/
##  FILES_MATCHING PATTERN "*.h")
#install(DIRECTORY include/dpMMlowVar/ DESTINATION include/dpMMlowVar/
#  FILES_MATCHING PATTERN "*.hpp")

pods_install_executables(ddpImageCluster dpMMlowVarCluster
  testClusterer testSpkmCUDA testSO3)
pods_install_libraries(dpMMlowVar)
pods_install_headers(
  ./include/dpMMlowVar/clusterer.hpp
  ./include/dpMMlowVar/ddpmeans.hpp
  ./include/dpMMlowVar/ddpmeansCUDA.hpp
  ./include/dpMMlowVar/dpmeans.hpp
  ./include/dpMMlowVar/ddpvMF_cuda_helper.h
  ./include/dpMMlowVar/euclideanData.hpp
  ./include/dpMMlowVar/sphericalData.hpp
  ./include/dpMMlowVar/spline.h
  ./include/dpMMlowVar/kmeans.hpp
  ./include/dpMMlowVar/kmeansCUDA.hpp
  ./include/dpMMlowVar/opencvHelper.hpp
  ./include/dpMMlowVar/random.hpp
  ./include/dpMMlowVar/SO3.hpp
  DESTINATION dpMMlowVar)
