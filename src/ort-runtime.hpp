#pragma once

// 所有 TU 都以 ORT_API_MANUAL_INIT 编译（见 CMakeLists.txt）：ONNX Runtime 的函数表由 ort_runtime_init 设置。
#include <onnxruntime_cxx_api.h>

#include <memory>
#include <string>

// 推理设备（滤镜设置）
enum class OrtAccel {
	Auto = 0,    // 有 GPU 方案就用（macOS：CoreML；Windows：CUDA 或 DirectML），失败退回 CPU
	CpuOnly = 1, // 只用 CPU
};

// 初始化 ONNX Runtime（进程内一次）。
// macOS：使用链接进来的 libonnxruntime。
// Windows：按完整路径加载插件目录下 ort-cuda\ 或 ort-dml\ 里的 onnxruntime.dll
//   （避开 Windows 11 System32 里自带的旧版 onnxruntime.dll）。有 ort-cuda 就先试 CUDA
//   （需要装好对应版本的 CUDA 和 cuDNN），CUDA 方案加不上就换 DirectML。
//   环境变量 FACE_MOSAIC_ORT=dml 可跳过 CUDA。
// 试 CUDA 时先按完整路径预加载 ort-cuda 里随包的 CUDA / cuDNN 运行库，再用 test_model 真正推理一次，
//   通过了才用 CUDA（驱动太旧、显卡不支持等都会在这里暴露），否则换 DirectML。
// plugin_dir：插件二进制所在目录（UTF-8）；test_model：用来实测的小模型路径（可空）。
// info 写入加载结果说明（不含用户数据）。
bool ort_runtime_init(const std::string &plugin_dir, std::string &info, const std::string &test_model = "");
bool ort_runtime_ready();

// 当前运行库可用的 GPU 方案名："CoreML"、"CUDA"、"DirectML"；没有时为 ""
const char *ort_gpu_provider();

// 会话设置：threads 个线程、不自旋；accel 为 Auto 时加上 GPU 方案。used 写入实际使用的方案（"CPU" 或 GPU 方案名）。
Ort::SessionOptions make_session_options(int threads, OrtAccel accel, std::string *used = nullptr);

// 创建会话：accel 为 Auto 时先用 GPU 方案，建不起来自动退回 CPU。used 写入实际使用的方案。
std::unique_ptr<Ort::Session> make_session(Ort::Env &env, const std::string &model_path, int threads, OrtAccel accel,
					   std::string *used = nullptr);

// UTF-8 路径 → ORT 需要的路径字符串（Windows 为宽字符）
std::basic_string<ORTCHAR_T> ort_path(const std::string &utf8);
