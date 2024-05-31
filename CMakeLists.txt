cmake_minimum_required(VERSION 3.16)
project(stars VERSION 0.1.0 LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 20)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(stars main.cpp)

target_link_libraries( stars ${OpenCV_LIBS} )