cmake_minimum_required(VERSION 3.16)
project(hw4 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 26)

find_package(Ceres REQUIRED)
find_package(OpenCV REQUIRED)
find_package(CGAL REQUIRED)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_BUILD_TYPE "Release")

add_executable(t01
	t01.cpp
)

add_executable(t02
	t02.cpp
)

add_executable(t03
	t03.cpp
)

add_executable(t04
	t04.cpp
)

set(MY_EXECUTABLES
	t01
	t02
	t03
	t04
)

foreach(target ${MY_EXECUTABLES})
	target_include_directories(${target} PRIVATE ${OpenCV_INCLUDE_DIRECTORIES})
	target_link_libraries(${target} PRIVATE Ceres::ceres)
	target_link_libraries(${target} PRIVATE ${OpenCV_LIBS})
endforeach()

target_link_libraries(t04 PRIVATE CGAL::CGAL)

