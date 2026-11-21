#include "ort-runtime.hpp"

#include <onnxruntime_session_options_config_keys.h>

#include <algorithm>
#include <cstdlib>
#include <mutex>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

namespace {

bool g_ready = false;
std::string g_gpu; // 当前运行库可用的 GPU 方案
std::mutex g_mutex;

#ifdef _WIN32
// dml_provider_factory.h 里 OrtDmlApi 的第一个成员（只用这一个，避免依赖 DirectML 包的头文件）
struct OrtDmlApiHead {
	OrtStatus *(ORT_API_CALL *SessionOptionsAppendExecutionProvider_DML)(OrtSessionOptions *options, int device_id);
};

std::wstring widen(const std::string &s)
{
	if (s.empty())
		return {};
	int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), (int)s.size(), nullptr, 0);
	std::wstring w(n, L'\0');
	MultiByteToWideChar(CP_UTF8, 0, s.data(), (int)s.size(), w.data(), n);
	return w;
}

bool file_exists(const std::wstring &p)
{
	DWORD a = GetFileAttributesW(p.c_str());
	return a != INVALID_FILE_ATTRIBUTES && !(a & FILE_ATTRIBUTE_DIRECTORY);
}

HMODULE g_ort = nullptr;

// 加载某个 onnxruntime.dll 并设置 C++ API 函数表
bool load_ort_dll(const std::wstring &path, std::string &err)
{
	HMODULE m = LoadLibraryExW(path.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
	if (!m) {
		err = "LoadLibrary 失败，错误码 " + std::to_string(GetLastError());
		return false;
	}
	using GetApiBase = const OrtApiBase *(ORT_API_CALL *)();
	auto get = reinterpret_cast<GetApiBase>(GetProcAddress(m, "OrtGetApiBase"));
	const OrtApi *api = get ? get()->GetApi(ORT_API_VERSION) : nullptr;
	if (!api) {
		err = "onnxruntime.dll 版本不匹配（需要 API 版本 " + std::to_string(ORT_API_VERSION) + "）";
		FreeLibrary(m);
		return false;
	}
	Ort::InitApi(api);
	g_ort = m;
	return true;
}

// 试着把 CUDA 方案加到一个临时会话设置上：CUDA / cuDNN 缺失或版本不对时这里就会报错。
// 期间关掉「找不到 DLL」的系统弹窗。
bool cuda_provider_works(std::string &err)
{
	DWORD old_mode = 0;
	SetThreadErrorMode(SEM_FAILCRITICALERRORS | SEM_NOOPENFILEERRORBOX, &old_mode);
	struct Restore {
		DWORD mode;
		~Restore() { SetThreadErrorMode(mode, nullptr); }
	} restore{old_mode};
	try {
		Ort::SessionOptions so;
		OrtCUDAProviderOptionsV2 *opts = nullptr;
		Ort::ThrowOnError(Ort::GetApi().CreateCUDAProviderOptions(&opts));
		so.AppendExecutionProvider_CUDA_V2(*opts);
		Ort::GetApi().ReleaseCUDAProviderOptions(opts);
		return true;
	} catch (const std::exception &e) {
		err = e.what();
		return false;
	}
}
#endif

} // namespace

bool ort_runtime_init(const std::string &plugin_dir, std::string &info)
{
	std::lock_guard<std::mutex> lock(g_mutex);
	if (g_ready) {
		info = "已初始化";
		return true;
	}
#ifdef _WIN32
	std::wstring dir = widen(plugin_dir);
	std::wstring cuda_dll = dir + L"\\ort-cuda\\onnxruntime.dll";
	std::wstring dml_dll = dir + L"\\ort-dml\\onnxruntime.dll";
	const char *force = std::getenv("FACE_MOSAIC_ORT");
	std::string want = force ? force : "";

	std::string err;
	bool try_cuda = want != "dml" && file_exists(cuda_dll);
	if (try_cuda) {
		if (load_ort_dll(cuda_dll, err)) {
			if (cuda_provider_works(err)) {
				g_gpu = "CUDA";
				g_ready = true;
				info = "使用 CUDA 运行库（ort-cuda）";
				return true;
			}
			// CUDA 方案加不上：卸掉这份运行库，换 DirectML
			FreeLibrary(g_ort);
			g_ort = nullptr;
			info = "CUDA 不可用（" + err + "），改用 DirectML；";
		} else {
			info = "加载 ort-cuda 失败（" + err + "），改用 DirectML；";
		}
	}
	if (!file_exists(dml_dll)) {
		info += "找不到 ort-dml\\onnxruntime.dll";
		return false;
	}
	if (!load_ort_dll(dml_dll, err)) {
		info += "加载 ort-dml 失败：" + err;
		return false;
	}
	g_gpu = "DirectML";
	g_ready = true;
	info += "使用 DirectML 运行库（ort-dml）";
	return true;
#else
	(void)plugin_dir;
	Ort::InitApi();
#ifdef __APPLE__
	g_gpu = "CoreML";
#endif
	g_ready = true;
	info = g_gpu.empty() ? "使用 CPU 运行库" : "使用内置运行库（可用 " + g_gpu + "）";
	return true;
#endif
}

bool ort_runtime_ready()
{
	return g_ready;
}

const char *ort_gpu_provider()
{
	return g_gpu.c_str();
}

Ort::SessionOptions make_session_options(int threads, OrtAccel accel, std::string *used)
{
	if (const char *v = std::getenv("FACE_MOSAIC_ORT_THREADS"); v && *v)
		threads = std::clamp(std::atoi(v), 1, 8);
	const char *spin = std::getenv("FACE_MOSAIC_ORT_SPIN");

	Ort::SessionOptions opts;
	opts.SetIntraOpNumThreads(threads);
	opts.SetInterOpNumThreads(1);
	opts.AddConfigEntry(kOrtSessionOptionsConfigAllowIntraOpSpinning, spin && spin[0] == '1' ? "1" : "0");
	opts.AddConfigEntry(kOrtSessionOptionsConfigAllowInterOpSpinning, "0");
	opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

	std::string provider = "CPU";
	if (accel == OrtAccel::Auto && !g_gpu.empty()) {
		try {
			if (g_gpu == "CoreML") {
				// MLProgram 格式、CPU + GPU：M4 上实测比交给 CoreML 自动分配（ALL）或用神经引擎都快，CPU 占用也低
				const char *units = std::getenv("FACE_MOSAIC_COREML_UNITS"); // 调试用：ALL / CPUAndGPU / CPUAndNeuralEngine
				opts.AppendExecutionProvider("CoreML", {{"ModelFormat", "MLProgram"},
									{"MLComputeUnits", units && *units ? units : "CPUAndGPU"}});
				provider = "CoreML";
			}
#ifdef _WIN32
			else if (g_gpu == "CUDA") {
				OrtCUDAProviderOptionsV2 *cuda = nullptr;
				Ort::ThrowOnError(Ort::GetApi().CreateCUDAProviderOptions(&cuda));
				opts.AppendExecutionProvider_CUDA_V2(*cuda);
				Ort::GetApi().ReleaseCUDAProviderOptions(cuda);
				provider = "CUDA";
			} else if (g_gpu == "DirectML") {
				// DirectML 要求关闭内存模式、顺序执行
				opts.DisableMemPattern();
				opts.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
				const void *api = nullptr;
				Ort::ThrowOnError(Ort::GetApi().GetExecutionProviderApi("DML", ORT_API_VERSION, &api));
				auto dml = static_cast<const OrtDmlApiHead *>(api);
				Ort::ThrowOnError(dml->SessionOptionsAppendExecutionProvider_DML(opts, 0));
				provider = "DirectML";
			}
#endif
		} catch (const std::exception &) {
			provider = "CPU"; // GPU 方案加不上就用 CPU（已加的选项不影响 CPU 执行）
		}
	}
	if (used)
		*used = provider;
	return opts;
}

std::basic_string<ORTCHAR_T> ort_path(const std::string &utf8)
{
#ifdef _WIN32
	return widen(utf8);
#else
	return utf8;
#endif
}
