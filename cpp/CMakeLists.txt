cmake_minimum_required(VERSION 3.21)
project(
	CPPSAM
	VERSION 0.0.0
	LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)

add_subdirectory(cppsam)
add_subdirectory(vino_executor)
add_subdirectory(test_app)
