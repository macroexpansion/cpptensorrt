cmake_minimum_required(VERSION 3.10)
project(TensorRT)

set(CMAKE_WARN_DEPRECATED OFF CACHE BOOL "" FORCE)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}" ${CMAKE_MODULE_PATH})

find_package(Threads)
find_package(CUDA REQUIRED)
find_package(OpenCV 4.5.1 REQUIRED)
find_package(TensorRT REQUIRED)

set(EXECUTABLE_OUTPUT_PATH "../bin")
add_executable(inference inference.cpp)
add_executable(serialize serialize_model.cpp)

set(INCLUDE_LIBS ${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIRS})
set(LIBS ${OpenCV_LIBS} ${CUDA_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${TensorRT_LIBRARIES})

target_include_directories(inference PUBLIC ${INCLUDE_LIBS})
target_link_libraries(inference PUBLIC ${LIBS})
target_include_directories(serialize PUBLIC ${INCLUDE_LIBS})
target_link_libraries(serialize PUBLIC ${LIBS})