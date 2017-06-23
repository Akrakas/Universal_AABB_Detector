cmake_minimum_required(VERSION 2.8)
project( BB_Finder )

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Build/)
set(SOURCE_FILES ${PROJECT_SOURCE_DIR}/Sources/main.cpp ${PROJECT_SOURCE_DIR}/Sources/Rect_fuser.cpp)
set(EXECUTABLE BB_Finder)

# add modules
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/Modules")

# find packages
find_package(TensorFlow REQUIRED)
find_package(Eigen REQUIRED)
find_package(Protobuf REQUIRED)
find_package( OpenCV REQUIRED )
set(PROJECT_INCLUDE_DIRS ${TensorFlow_INCLUDE_DIRS} ${Eigen_INCLUDE_DIRS} ${PROTOBUF_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
set(PROJECT_LIBRARIES ${TensorFlow_LIBRARIES} ${PROTOBUF_LIBRARIES} ${OpenCV_LIBS})

include_directories( ${PROJECT_INCLUDE_DIRS} )
add_executable(${EXECUTABLE} ${SOURCE_FILES})
target_link_libraries(${EXECUTABLE} ${PROJECT_LIBRARIES})
