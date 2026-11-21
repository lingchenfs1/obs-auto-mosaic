#pragma once

#include <onnxruntime_cxx_api.h>
#include <onnxruntime_session_options_config_keys.h>

#include <algorithm>
#include <cstdlib>

// 所有检测器共用的 ONNX Runtime 会话设置。
// 默认每个模型 1 个线程（三个通道本来就各在自己的线程里并行）：
// 2026-11-21 在 M4 上实测，2 线程 + 自旋时 OBS 约 2.3~2.7 个核；2 线程不自旋时 macOS 唤醒线程太慢，
// 每次检测拖到 100~300 ms；1 线程时约 1.3 个核，每次检测 15~35 ms，延迟对齐不受影响。
// 环境变量（调试用）：FACE_MOSAIC_ORT_THREADS=<每个模型的线程数>，FACE_MOSAIC_ORT_SPIN=1 打开自旋。
inline Ort::SessionOptions make_session_options(int default_threads)
{
	int threads = default_threads;
	if (const char *v = std::getenv("FACE_MOSAIC_ORT_THREADS"); v && *v)
		threads = std::clamp(std::atoi(v), 1, 8);
	const char *spin = std::getenv("FACE_MOSAIC_ORT_SPIN");

	Ort::SessionOptions opts;
	opts.SetIntraOpNumThreads(threads);
	opts.SetInterOpNumThreads(1);
	opts.AddConfigEntry(kOrtSessionOptionsConfigAllowIntraOpSpinning, spin && spin[0] == '1' ? "1" : "0");
	opts.AddConfigEntry(kOrtSessionOptionsConfigAllowInterOpSpinning, "0");
	opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
	return opts;
}
