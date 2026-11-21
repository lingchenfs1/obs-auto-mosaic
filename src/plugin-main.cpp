#include <obs-module.h>

OBS_DECLARE_MODULE()
OBS_MODULE_USE_DEFAULT_LOCALE("obs-face-mosaic", "en-US")

extern struct obs_source_info face_mosaic_filter_info;

bool obs_module_load(void)
{
	obs_register_source(&face_mosaic_filter_info);
	blog(LOG_INFO, "[face-mosaic] 插件已加载，版本 0.1.0");
	return true;
}

const char *obs_module_description(void)
{
	return "Auto face mosaic filter (YuNet + ONNX Runtime)";
}
