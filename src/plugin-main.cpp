#include <obs-module.h>

#include "ort-runtime.hpp"

#include <string>

OBS_DECLARE_MODULE()
OBS_MODULE_USE_DEFAULT_LOCALE("obs-face-mosaic", "en-US")

extern struct obs_source_info face_mosaic_filter_info;

bool obs_module_load(void)
{
	// 插件二进制所在目录（Windows 上从这里的 ort-cuda / ort-dml 子目录加载 onnxruntime.dll）
	const char *bin = obs_get_module_binary_path(obs_current_module());
	std::string dir = bin ? bin : "";
	size_t slash = dir.find_last_of("/\\");
	dir = slash == std::string::npos ? std::string(".") : dir.substr(0, slash);
	// 用最小的人脸模型实测 GPU 方案能不能真正跑起来
	char *test = obs_module_file("face_detection_yunet_2023mar.onnx");
	std::string test_model = test ? test : "";
	bfree(test);
	std::string info;
	bool ok = ort_runtime_init(dir, info, test_model);
	blog(ok ? LOG_INFO : LOG_ERROR, "[face-mosaic] ONNX Runtime：%s", info.c_str());

	obs_register_source(&face_mosaic_filter_info);
	blog(LOG_INFO, "[face-mosaic] 插件已加载，版本 0.2.0");
	return true;
}

const char *obs_module_description(void)
{
	return "Auto face mosaic filter (YuNet + ONNX Runtime)";
}
