cmake_minimum_required(VERSION 4.0.0)
set(CMAKE_BUILD_TYPE Debug)
project(21-5 VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_C_COMPILER "/usr/bin/gcc")
set(CMAKE_CXX_COMPILER "/usr/bin/g++")

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB SRC_FILES "src/*.cpp")
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_BINARY_DIR}/Release/generators")

find_package(Eigen3 REQUIRED)
find_package(dlib REQUIRED)
add_executable(dlibCluster src/dlib-cluster.cpp)
#add_library(ETL SHARED ${SRC_FILES})



target_include_directories(dlibCluster PUBLIC ${Eigen3_INCLUDE_DIR})
target_include_directories(dlibCluster PUBLIC /usr/include)
target_include_directories(dlibCluster PUBLIC /usr/include/plotcpp)
target_include_directories(dlibCluster PUBLIC /usr/local/include)

target_link_libraries(dlibCluster PRIVATE Eigen3::Eigen)
target_link_libraries(dlibCluster PRIVATE  armadillo)
target_link_libraries(dlibCluster PRIVATE  dlib::dlib)
#target_link_libraries(ai ${OpenCV_LIBS} )