cmake_minimum_required(VERSION 3.24)
project(obs-face-mosaic VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

set(OBS_SOURCE_DIR "${CMAKE_SOURCE_DIR}/third_party/obs-studio" CACHE PATH "obs-studio 源码（只用 libobs 头文件）")
set(OBS_APP "/Applications/OBS.app" CACHE PATH "已安装的 OBS.app，用来链接 libobs")
set(ORT_DIR "${CMAKE_SOURCE_DIR}/third_party/onnxruntime" CACHE PATH "onnxruntime 预编译包（include + lib）")

set(ORT_DYLIB "${ORT_DIR}/lib/libonnxruntime.dylib")

# ---- 检测器（插件和测试工具共用）----
add_library(face_detector STATIC src/face-detector.cpp src/nude-detector.cpp src/skin-detector.cpp
  src/text-detector.cpp)
target_include_directories(face_detector PUBLIC src "${ORT_DIR}/include")
target_link_libraries(face_detector PUBLIC "${ORT_DYLIB}")
set_target_properties(face_detector PROPERTIES POSITION_INDEPENDENT_CODE ON)

# ---- 离线测试工具 ----
add_executable(detect-test tools/detect-test.cpp)
target_include_directories(detect-test PRIVATE third_party/stb)
target_link_libraries(detect-test PRIVATE face_detector)
set_target_properties(detect-test PROPERTIES BUILD_RPATH "${ORT_DIR}/lib")
# dylib 的 install name 是 libonnxruntime.1.dylib，补一个同名链接
add_custom_command(TARGET detect-test POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E create_symlink libonnxruntime.dylib "${ORT_DIR}/lib/libonnxruntime.1.dylib")

add_executable(nude-test tools/nude-test.cpp tools/stb-impl-test.c)
target_include_directories(nude-test PRIVATE third_party/stb)
target_link_libraries(nude-test PRIVATE face_detector)
set_target_properties(nude-test PROPERTIES BUILD_RPATH "${ORT_DIR}/lib")

add_executable(skin-test tools/skin-test.cpp tools/stb-impl-test.c)
target_include_directories(skin-test PRIVATE third_party/stb)
target_link_libraries(skin-test PRIVATE face_detector)
set_target_properties(skin-test PROPERTIES BUILD_RPATH "${ORT_DIR}/lib")

add_executable(text-test tools/text-test.cpp tools/stb-impl-test.c)
target_include_directories(text-test PRIVATE third_party/stb)
target_link_libraries(text-test PRIVATE face_detector)
set_target_properties(text-test PROPERTIES BUILD_RPATH "${ORT_DIR}/lib")

# ---- OBS 插件 ----
add_library(obs-face-mosaic MODULE src/plugin-main.cpp src/face-mosaic-filter.cpp src/stb-impl.c)
find_path(SIMDE_INCLUDE_DIR simde/x86/sse2.h PATHS /opt/homebrew/include /usr/local/include REQUIRED)
target_include_directories(obs-face-mosaic PRIVATE include third_party/stb "${OBS_SOURCE_DIR}/libobs" "${SIMDE_INCLUDE_DIR}")
target_compile_options(obs-face-mosaic PRIVATE -Wall -Wno-unused-parameter)
target_link_options(obs-face-mosaic PRIVATE "-F${OBS_APP}/Contents/Frameworks")
target_link_libraries(obs-face-mosaic PRIVATE face_detector "-framework libobs")
set_target_properties(obs-face-mosaic PROPERTIES
  BUNDLE TRUE
  BUNDLE_EXTENSION plugin
  MACOSX_BUNDLE_INFO_PLIST "${CMAKE_SOURCE_DIR}/cmake/Info.plist.in"
  MACOSX_BUNDLE_GUI_IDENTIFIER "com.obsai.obs-face-mosaic"
  MACOSX_BUNDLE_BUNDLE_VERSION "${PROJECT_VERSION}"
  MACOSX_BUNDLE_SHORT_VERSION_STRING "${PROJECT_VERSION}"
  INSTALL_RPATH "@loader_path/../Frameworks"
  BUILD_WITH_INSTALL_RPATH TRUE)

# 把模型、语言文件、onnxruntime 放进 bundle
set(BUNDLE_DIR "$<TARGET_BUNDLE_DIR:obs-face-mosaic>")
add_custom_command(TARGET obs-face-mosaic POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory "${BUNDLE_DIR}/Contents/Resources/locale" "${BUNDLE_DIR}/Contents/Frameworks"
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_SOURCE_DIR}/data/face_detection_yunet_2023mar.onnx" "${BUNDLE_DIR}/Contents/Resources/"
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_SOURCE_DIR}/data/seg/pphumanseg_2023mar.onnx" "${BUNDLE_DIR}/Contents/Resources/"
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_SOURCE_DIR}/data/ocr/ch_PP-OCRv4_det_infer.onnx" "${CMAKE_SOURCE_DIR}/data/ocr/ch_PP-OCRv4_rec_infer.onnx" "${BUNDLE_DIR}/Contents/Resources/"
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_SOURCE_DIR}/data/nudenet/320n.onnx" "${CMAKE_SOURCE_DIR}/data/nudenet/640m.onnx" "${BUNDLE_DIR}/Contents/Resources/"
  COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_SOURCE_DIR}/data/locale" "${BUNDLE_DIR}/Contents/Resources/locale"
  COMMAND ${CMAKE_COMMAND} -E copy "${ORT_DYLIB}" "${BUNDLE_DIR}/Contents/Frameworks/libonnxruntime.1.dylib"
  COMMAND codesign --force --sign - "${BUNDLE_DIR}"
  VERBATIM)

# 安装到当前用户的 OBS 插件目录
set(OBS_USER_PLUGIN_DIR "$ENV{HOME}/Library/Application Support/obs-studio/plugins")
add_custom_target(install-plugin
  COMMAND ${CMAKE_COMMAND} -E rm -rf "${OBS_USER_PLUGIN_DIR}/obs-face-mosaic.plugin"
  COMMAND ${CMAKE_COMMAND} -E copy_directory "${BUNDLE_DIR}" "${OBS_USER_PLUGIN_DIR}/obs-face-mosaic.plugin"
  DEPENDS obs-face-mosaic
  VERBATIM)
