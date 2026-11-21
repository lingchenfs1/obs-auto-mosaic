#include "ort-runtime.hpp"

#include <onnxruntime_session_options_config_keys.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <mutex>
#include <vector>

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

std::vector<HMODULE> g_nvidia; // 预加载的随包 NVIDIA 运行库

// 按完整路径预加载 ort-cuda 目录里随包的 CUDA / cuDNN 运行库。
// onnxruntime 之后按文件名找 cudnn64_9.dll 等时，Windows 直接用已加载的这份：
// 不依赖 PATH，也不会混用系统里装的其他版本。目录里没有这些 DLL（旧版扩展包）时什么都不做。
int preload_nvidia_dlls(const std::wstring &dir)
{
	std::vector<std::wstring> names;
	WIN32_FIND_DATAW fd;
	HANDLE h = FindFirstFileW((dir + L"\\*.dll").c_str(), &fd);
	if (h != INVALID_HANDLE_VALUE) {
		do {
			names.push_back(fd.cFileName);
		} while (FindNextFileW(h, &fd));
		FindClose(h);
	}
	// 被依赖的先加载；同一前缀内按文件名顺序
	std::sort(names.begin(), names.end());
	const wchar_t *order[] = {L"cudart64_", L"cublasLt64_",       L"cublas64_",      L"cufft64_",
				  L"nvrtc-builtins", L"nvrtc64_", L"cudnn_graph64_", L"cudnn"};
	std::vector<bool> done(names.size(), false);
	for (const wchar_t *prefix : order) {
		size_t len = wcslen(prefix);
		for (size_t i = 0; i < names.size(); i++) {
			if (done[i] || _wcsnicmp(names[i].c_str(), prefix, len) != 0)
				continue;
			done[i] = true;
			HMODULE m = LoadLibraryExW((dir + L"\\" + names[i]).c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
			if (m)
				g_nvidia.push_back(m);
		}
	}
	return (int)g_nvidia.size();
}

void free_nvidia_dlls()
{
	for (auto it = g_nvidia.rbegin(); it != g_nvidia.rend(); ++it)
		FreeLibrary(*it);
	g_nvidia.clear();
}

// 用 CUDA 真正加载一个模型并推理一次：驱动太旧、显卡不支持、cuDNN 有问题，都要到这一步才会暴露。
bool cuda_session_works(const std::string &test_model, std::string &err)
{
	try {
		Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "obs-face-mosaic-cuda-check");
		Ort::SessionOptions so;
		OrtCUDAProviderOptionsV2 *opts = nullptr;
		Ort::ThrowOnError(Ort::GetApi().CreateCUDAProviderOptions(&opts));
		so.AppendExecutionProvider_CUDA_V2(*opts);
		Ort::GetApi().ReleaseCUDAProviderOptions(opts);
		if (test_model.empty())
			return true; // 没有测试模型时只能检查到这一步
		Ort::Session session(env, widen(test_model).c_str(), so);

		// 按模型的输入形状造一个全 0 输入（可变维度取 1），跑一次
		Ort::AllocatorWithDefaultOptions alloc;
		auto in_name = session.GetInputNameAllocated(0, alloc);
		std::vector<int64_t> shape = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
		size_t count = 1;
		for (int64_t &d : shape) {
			if (d <= 0)
				d = 1;
			count *= (size_t)d;
		}
		std::vector<float> data(count, 0.0f);
		auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
		Ort::Value input = Ort::Value::CreateTensor<float>(mem, data.data(), data.size(), shape.data(), shape.size());
		std::vector<Ort::AllocatedStringPtr> out_names_alloc;
		std::vector<const char *> out_names;
		for (size_t i = 0; i < session.GetOutputCount(); i++) {
			out_names_alloc.push_back(session.GetOutputNameAllocated(i, alloc));
			out_names.push_back(out_names_alloc.back().get());
		}
		const char *in_names[] = {in_name.get()};
		session.Run(Ort::RunOptions{nullptr}, in_names, &input, 1, out_names.data(), out_names.size());
		return true;
	} catch (const std::exception &e) {
		err = e.what();
		return false;
	}
}
#endif

} // namespace

bool ort_runtime_init(const std::string &plugin_dir, std::string &info, const std::string &test_model)
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
		// 试 CUDA 期间关掉「找不到 DLL」之类的系统弹窗
		DWORD old_mode = 0;
		SetThreadErrorMode(SEM_FAILCRITICALERRORS | SEM_NOOPENFILEERRORBOX, &old_mode);
		int preloaded = preload_nvidia_dlls(dir + L"\\ort-cuda");
		std::string pre = preloaded ? "，预加载随包的 NVIDIA 运行库 " + std::to_string(preloaded) + " 个" : "";
		bool ok = false;
		if (load_ort_dll(cuda_dll, err)) {
			ok = cuda_session_works(test_model, err);
			if (!ok) {
				FreeLibrary(g_ort);
				g_ort = nullptr;
			}
		}
		SetThreadErrorMode(old_mode, nullptr);
		if (ok) {
			g_gpu = "CUDA";
			g_ready = true;
			info = "使用 CUDA 运行库（ort-cuda" + pre + "，实测推理通过）";
			return true;
		}
		// CUDA 不可用：卸掉这份运行库和预加载的 NVIDIA 库，换 DirectML
		free_nvidia_dlls();
		info = "CUDA 不可用（" + err + pre + "），改用 DirectML；";
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
	(void)test_model;
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

std::unique_ptr<Ort::Session> make_session(Ort::Env &env, const std::string &model_path, int threads, OrtAccel accel,
					   std::string *used)
{
	std::string provider;
	if (accel == OrtAccel::Auto && *ort_gpu_provider()) {
		try {
			Ort::SessionOptions opts = make_session_options(threads, accel, &provider);
			auto s = std::make_unique<Ort::Session>(env, ort_path(model_path).c_str(), opts);
			if (used)
				*used = provider;
			return s;
		} catch (const std::exception &) {
			// GPU 上建不起来（显存不够、算子不支持等）：退回 CPU，不让一个模型的问题变成整屏打码
		}
	}
	Ort::SessionOptions opts = make_session_options(threads, OrtAccel::CpuOnly, &provider);
	auto s = std::make_unique<Ort::Session>(env, ort_path(model_path).c_str(), opts);
	if (used)
		*used = accel == OrtAccel::Auto && *ort_gpu_provider() ? "CPU（GPU 加载失败）" : "CPU";
	return s;
}

std::basic_string<ORTCHAR_T> ort_path(const std::string &utf8)
{
#ifdef _WIN32
	return widen(utf8);
#else
	return utf8;
#endif
}
