# CMake instructions to make the static lib
set(target_name cppsam)

file(GLOB_RECURSE headers CONFIGURE_DEPENDS "../include/cppsam/*.h")
file(GLOB_RECURSE sources CONFIGURE_DEPENDS "*.cpp")
file(GLOB_RECURSE headers_intern CONFIGURE_DEPENDS "*.h")

find_package(OpenCV 4.2 REQUIRED opencv_core opencv_imgproc opencv_imgcodecs)

add_library( ${target_name} STATIC ${headers} ${headers_intern} ${sources} )
set_target_properties(${target_name} PROPERTIES CXX_STANDARD 20)
target_include_directories(
  ${target_name} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
)

target_link_libraries(${target_name} PUBLIC opencv_core opencv_imgproc opencv_imgcodecs)
