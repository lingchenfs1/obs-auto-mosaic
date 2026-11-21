#include "face-detector.hpp"

#include "ort-runtime.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <mutex>

namespace {

Ort::Env &ort_env()
{
	static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "obs-face-mosaic");
	return env;
}

float iou(const FaceBox &a, const FaceBox &b)
{
	float x1 = std::max(a.x, b.x);
	float y1 = std::max(a.y, b.y);
	float x2 = std::min(a.x + a.w, b.x + b.w);
	float y2 = std::min(a.y + a.h, b.y + b.h);
	float inter = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
	float uni = a.w * a.h + b.w * b.h - inter;
	return uni > 0.0f ? inter / uni : 0.0f;
}

} // namespace

struct FaceDetector::Impl {
	std::unique_ptr<Ort::Session> session;
	std::vector<float> input;
	// 输出顺序：cls_8/16/32, obj_8/16/32, bbox_8/16/32
	std::array<const char *, 9> output_names = {"cls_8", "cls_16", "cls_32", "obj_8", "obj_16",
						     "obj_32", "bbox_8", "bbox_16", "bbox_32"};
};

FaceDetector::FaceDetector() : impl_(std::make_unique<Impl>()) {}
FaceDetector::~FaceDetector() = default;

bool FaceDetector::loaded() const
{
	return impl_->session != nullptr;
}

bool FaceDetector::load(const std::string &model_path, std::string &error, int accel)
{
	try {
		Ort::SessionOptions opts = make_session_options(1, accel == 0 ? OrtAccel::Auto : OrtAccel::CpuOnly, &provider_);
		impl_->session = std::make_unique<Ort::Session>(ort_env(), ort_path(model_path).c_str(), opts);
		impl_->input.assign(3 * kInputSize * kInputSize, 0.0f);
		return true;
	} catch (const std::exception &e) {
		error = e.what();
		impl_->session.reset();
		return false;
	}
}

std::vector<FaceBox> FaceDetector::detect(const uint8_t *rgba, int w, int h, int stride, float score_threshold,
					  float nms_threshold)
{
	std::vector<FaceBox> result;
	if (!impl_->session || w <= 0 || h <= 0 || w > kInputSize || h > kInputSize)
		return result;

	// RGBA -> BGR 平面（NCHW），其余区域补零
	const int plane = kInputSize * kInputSize;
	float *b = impl_->input.data();
	float *g = b + plane;
	float *r = g + plane;
	std::fill(impl_->input.begin(), impl_->input.end(), 0.0f);
	for (int y = 0; y < h; y++) {
		const uint8_t *row = rgba + (size_t)y * stride;
		int o = y * kInputSize;
		for (int x = 0; x < w; x++) {
			r[o + x] = row[x * 4 + 0];
			g[o + x] = row[x * 4 + 1];
			b[o + x] = row[x * 4 + 2];
		}
	}

	std::vector<Ort::Value> outputs;
	try {
		auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
		std::array<int64_t, 4> shape = {1, 3, kInputSize, kInputSize};
		Ort::Value tensor = Ort::Value::CreateTensor<float>(mem, impl_->input.data(), impl_->input.size(),
								    shape.data(), shape.size());
		const char *input_names[] = {"input"};
		outputs = impl_->session->Run(Ort::RunOptions{nullptr}, input_names, &tensor, 1,
					      impl_->output_names.data(), impl_->output_names.size());
	} catch (const std::exception &) {
		return result;
	}

	// 后处理与 OpenCV FaceDetectorYN（2023mar 模型）一致
	std::vector<FaceBox> candidates;
	const int strides[3] = {8, 16, 32};
	for (int i = 0; i < 3; i++) {
		const int s = strides[i];
		const int cols = kInputSize / s;
		const int rows = kInputSize / s;
		const float *cls = outputs[i].GetTensorData<float>();
		const float *obj = outputs[i + 3].GetTensorData<float>();
		const float *bbox = outputs[i + 6].GetTensorData<float>();
		for (int ry = 0; ry < rows; ry++) {
			for (int cx = 0; cx < cols; cx++) {
				int idx = ry * cols + cx;
				float cs = std::clamp(cls[idx], 0.0f, 1.0f);
				float os = std::clamp(obj[idx], 0.0f, 1.0f);
				float score = std::sqrt(cs * os);
				if (score < score_threshold)
					continue;
				float ccx = (cx + bbox[idx * 4 + 0]) * s;
				float ccy = (ry + bbox[idx * 4 + 1]) * s;
				float bw = std::exp(bbox[idx * 4 + 2]) * s;
				float bh = std::exp(bbox[idx * 4 + 3]) * s;
				candidates.push_back({ccx - bw / 2.0f, ccy - bh / 2.0f, bw, bh, score});
			}
		}
	}

	// NMS
	std::sort(candidates.begin(), candidates.end(),
		  [](const FaceBox &a, const FaceBox &b) { return a.score > b.score; });
	std::vector<bool> removed(candidates.size(), false);
	for (size_t i = 0; i < candidates.size(); i++) {
		if (removed[i])
			continue;
		result.push_back(candidates[i]);
		for (size_t j = i + 1; j < candidates.size(); j++) {
			if (!removed[j] && iou(candidates[i], candidates[j]) > nms_threshold)
				removed[j] = true;
		}
	}
	return result;
}
