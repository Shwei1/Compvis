cmake_minimum_required(VERSION 3.16)
project(hw3 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 26)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Ceres REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(t01
	t01.cpp
	hw3utils.cpp
)

add_executable(t02
	t02.cpp
)

add_executable(t03
	t03.cpp
	hw3utils.cpp
)

add_executable(t04
	t04.cpp
	hw3utils.cpp
)

set(MY_EXECUTABLES
	t01
	t02
	t03
)


foreach(target ${MY_EXECUTABLES})
	target_link_libraries(${target} PRIVATE Ceres::ceres)
endforeach()

target_include_directories(t03 PRIVATE ${OpenCV_INCLUDE_DIRECTORIES})
target_link_libraries(t03 PRIVATE ${OpenCV_LIBS})
target_compile_definitions(t03 PRIVATE USE_OPENCV)
