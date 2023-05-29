# CMake instructions to make the static lib
set(target_name vino_executor)

file(GLOB_RECURSE headers CONFIGURE_DEPENDS "../include/vino_executor/*.h")
file(GLOB_RECURSE sources CONFIGURE_DEPENDS "*.cpp")
file(GLOB_RECURSE headers_intern CONFIGURE_DEPENDS "*.h")

find_package(OpenCV 4.2 REQUIRED opencv_core opencv_imgproc opencv_imgcodecs)
find_package(OpenVINO REQUIRED)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

add_library( ${target_name} STATIC ${headers} ${headers_intern} ${sources} )
set_target_properties(${target_name} PROPERTIES CXX_STANDARD 20)
target_include_directories(
  ${target_name} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
)

target_link_libraries(${target_name} PUBLIC opencv_core opencv_imgproc opencv_imgcodecs openvino::frontend::onnx openvino::runtime cppsam)
