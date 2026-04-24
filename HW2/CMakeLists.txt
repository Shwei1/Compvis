cmake_minimum_required(VERSION 3.16)
project(hw2 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 26)

find_package(OpenCV REQUIRED)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(t01
	t01.cpp
	hw2utils.cpp
)

add_executable(t02
	t02.cpp
	hw2utils.cpp
)

add_executable(t03
	t03.cpp
	hw2utils.cpp
)

add_executable(mc
	minecraft.cpp
)

set(MY_EXECUTABLES
	t01
	t02
	t03
)

foreach(target ${MY_EXECUTABLES})
	target_link_libraries(${target} PRIVATE ${OpenCV_LIBS})
	target_include_directories(${target} PRIVATE ${OpenCV_INCLUDE_DIRECTORIES})
endforeach()
