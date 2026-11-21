cmake_minimum_required(VERSION 3.24)
project(obs-face-mosaic VERSION 0.3.1 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

set(OBS_SOURCE_DIR "${CMAKE_SOURCE_DIR}/third_party/obs-studio" CACHE PATH "obs-studio 源码（只用 libobs 头文件）")
set(ORT_DIR "${CMAKE_SOURCE_DIR}/third_party/onnxruntime" CACHE PATH "onnxruntime 预编译包（include + lib）")

# ONNX Runtime 的 C++ API 函数表由 ort_runtime_init 设置（Windows 上要先按完整路径加载 DLL）
add_compile_definitions(ORT_API_MANUAL_INIT)
# 版本号只在 project() 里写一处，插件日志从这里取（方便从用户日志判断装的是哪个版本）
add_compile_definitions(PLUGIN_VERSION="${PROJECT_VERSION}")
if(MSVC)
  # 源码是 UTF-8（含中文字符串）；关掉 CRT 的「不安全函数」告警
  add_compile_options(/utf-8 /W3)
  add_compile_definitions(_CRT_SECURE_NO_WARNINGS NOMINMAX)
endif()

set(MODELS
  "${CMAKE_SOURCE_DIR}/data/face_detection_yunet_2023mar.onnx"
  "${CMAKE_SOURCE_DIR}/data/seg/pphumanseg_2023mar.onnx"
  "${CMAKE_SOURCE_DIR}/data/ocr/ch_PP-OCRv4_det_infer.onnx"
  "${CMAKE_SOURCE_DIR}/data/ocr/ch_PP-OCRv4_rec_infer.onnx"
  "${CMAKE_SOURCE_DIR}/data/nudenet/320n.onnx"
  "${CMAKE_SOURCE_DIR}/data/nudenet/640m.onnx"
  "${CMAKE_SOURCE_DIR}/data/scene/clip_vision_int8.onnx"
  "${CMAKE_SOURCE_DIR}/data/scene/clip_prompts.bin")

# ---- 检测器（插件和测试工具共用）----
add_library(face_detector STATIC src/face-detector.cpp src/nude-detector.cpp src/skin-detector.cpp
  src/text-detector.cpp src/scene-detector.cpp src/ort-runtime.cpp)
target_include_directories(face_detector PUBLIC src "${ORT_DIR}/include")
set_target_properties(face_detector PROPERTIES POSITION_INDEPENDENT_CODE ON)

# ---- OBS 插件 ----
add_library(obs-face-mosaic MODULE src/plugin-main.cpp src/face-mosaic-filter.cpp src/stb-impl.c)
target_include_directories(obs-face-mosaic PRIVATE include third_party/stb "${OBS_SOURCE_DIR}/libobs")
target_link_libraries(obs-face-mosaic PRIVATE face_detector)

if(APPLE)
  set(OBS_APP "/Applications/OBS.app" CACHE PATH "已安装的 OBS.app，用来链接 libobs")
  set(ORT_DYLIB "${ORT_DIR}/lib/libonnxruntime.dylib")
  target_link_libraries(face_detector PUBLIC "${ORT_DYLIB}")

  # ---- 离线测试工具（只在 macOS 构建）----
  foreach(tool detect-test nude-test skin-test text-test scene-test)
    if(tool STREQUAL "detect-test")
      add_executable(${tool} tools/${tool}.cpp)
    else()
      add_executable(${tool} tools/${tool}.cpp tools/stb-impl-test.c)
    endif()
    target_include_directories(${tool} PRIVATE third_party/stb)
    target_link_libraries(${tool} PRIVATE face_detector)
    set_target_properties(${tool} PROPERTIES BUILD_RPATH "${ORT_DIR}/lib")
  endforeach()
  # dylib 的 install name 是 libonnxruntime.1.dylib，补一个同名链接
  add_custom_command(TARGET detect-test POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E create_symlink libonnxruntime.dylib "${ORT_DIR}/lib/libonnxruntime.1.dylib")

  find_path(SIMDE_INCLUDE_DIR simde/x86/sse2.h PATHS /opt/homebrew/include /usr/local/include REQUIRED)
  target_include_directories(obs-face-mosaic PRIVATE "${SIMDE_INCLUDE_DIR}")
  target_compile_options(obs-face-mosaic PRIVATE -Wall -Wno-unused-parameter)
  target_link_options(obs-face-mosaic PRIVATE "-F${OBS_APP}/Contents/Frameworks")
  target_link_libraries(obs-face-mosaic PRIVATE "-framework libobs")
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
    COMMAND ${CMAKE_COMMAND} -E copy ${MODELS} "${BUNDLE_DIR}/Contents/Resources/"
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

elseif(WIN32)
  # libobs：从 OBS 发布包里的 obs.dll 生成的导入库（见 .github/workflows/build.yml）
  set(OBS_LIB "${CMAKE_SOURCE_DIR}/third_party/obs-win/obs.lib" CACHE FILEPATH "obs.dll 的导入库")
  # 两套 ONNX Runtime 运行库：DirectML（必需）和 CUDA（可选），运行时按完整路径加载，不链接
  set(ORT_DML_DIR "${CMAKE_SOURCE_DIR}/third_party/ort-win/dml" CACHE PATH "onnxruntime.dll（DirectML 版）和 DirectML.dll")
  set(ORT_CUDA_DIR "${CMAKE_SOURCE_DIR}/third_party/ort-win/cuda" CACHE PATH "onnxruntime*.dll（CUDA 版），可不存在")
  target_link_libraries(obs-face-mosaic PRIVATE "${OBS_LIB}")

  # 组装成 OBS 第三方插件目录：<包>/obs-face-mosaic/bin/64bit/*.dll、<包>/obs-face-mosaic/data/*
  set(PKG_DIR "${CMAKE_BINARY_DIR}/package/obs-face-mosaic")
  add_custom_command(TARGET obs-face-mosaic POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${PKG_DIR}/bin/64bit/ort-dml" "${PKG_DIR}/data/locale"
    COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_FILE:obs-face-mosaic>" "${PKG_DIR}/bin/64bit/"
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${ORT_DML_DIR}" "${PKG_DIR}/bin/64bit/ort-dml"
    COMMAND ${CMAKE_COMMAND} -E copy ${MODELS} "${PKG_DIR}/data/"
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_SOURCE_DIR}/data/locale" "${PKG_DIR}/data/locale"
    VERBATIM)
  if(EXISTS "${ORT_CUDA_DIR}")
    set(PKG_CUDA_DIR "${CMAKE_BINARY_DIR}/package-cuda/obs-face-mosaic/bin/64bit/ort-cuda")
    add_custom_command(TARGET obs-face-mosaic POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E make_directory "${PKG_CUDA_DIR}"
      COMMAND ${CMAKE_COMMAND} -E copy_directory "${ORT_CUDA_DIR}" "${PKG_CUDA_DIR}"
      VERBATIM)
  endif()
endif()
