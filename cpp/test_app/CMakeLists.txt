set(target_name test_app)

find_package(OpenCV 4.2 REQUIRED opencv_core opencv_imgproc opencv_imgcodecs)
find_package(OpenVINO REQUIRED Runtime ONNX)

add_executable(
  ${target_name}
  main.cpp
)

set_target_properties(${target_name} PROPERTIES CXX_STANDARD 20)
target_include_directories(${target_name} PRIVATE "../code")
target_link_libraries(
  ${target_name}
  PRIVATE opencv_core opencv_imgproc opencv_imgcodecs openvino::runtime cppsam vino_executor
)

get_filename_component(OpenVINO_ROOT "${OpenVINO_DIR}/../" ABSOLUTE CACHE)
if (MSVC)
get_target_property(opencv_dbg opencv_world IMPORTED_LOCATION_DEBUG)
get_target_property(opencv_release opencv_world IMPORTED_LOCATION_RELEASE)
file(GLOB openvino_dbg "${OpenVINO_ROOT}/bin/intel64/Debug")
file(GLOB openvino_release "${OpenVINO_ROOT}/bin/intel64/Release")
file(GLOB tbb_dir "${OpenVINO_ROOT}/3rdparty/tbb/bin")

message("tbb_dir = ${tbb_dir}")
message("openvino_files = ${openvino_dbg}")

add_custom_command(
  TARGET ${target_name}
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different "$<$<CONFIG:debug>:${opencv_dbg}>$<$<CONFIG:release>:${opencv_release}>" $<TARGET_FILE_DIR:${target_name}>
  COMMAND ${CMAKE_COMMAND} -E copy_directory "$<$<CONFIG:debug>:${openvino_dbg}>$<$<CONFIG:release>:${openvino_release}>" $<TARGET_FILE_DIR:${target_name}>
  COMMAND ${CMAKE_COMMAND} -E copy_directory "${tbb_dir}" $<TARGET_FILE_DIR:${target_name}>
)
endif (MSVC)

