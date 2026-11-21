#include "nude-detector.hpp"

#include "ort-runtime.hpp"

#include <algorithm>
#include <array>
#include <cmath>

namespace {

Ort::Env &ort_env()
{
	static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "obs-face-mosaic-nude");
	return env;
}

const char *kNames[NUDE_CLASS_COUNT] = {
	"FEMALE_GENITALIA_COVERED", "FACE_FEMALE",     "BUTTOCKS_EXPOSED",       "FEMALE_BREAST_EXPOSED",
	"FEMALE_GENITALIA_EXPOSED", "MALE_BREAST_EXPOSED", "ANUS_EXPOSED",        "FEET_EXPOSED",
	"BELLY_COVERED",            "FEET_COVERED",    "ARMPITS_COVERED",        "ARMPITS_EXPOSED",
	"FACE_MALE",                "BELLY_EXPOSED",   "MALE_GENITALIA_EXPOSED", "ANUS_COVERED",
	"FEMALE_BREAST_COVERED",    "BUTTOCKS_COVERED",
};

float iou(const NudeBox &a, const NudeBox &b)
{
	float x1 = std::max(a.x, b.x), y1 = std::max(a.y, b.y);
	float x2 = std::min(a.x + a.w, b.x + b.w), y2 = std::min(a.y + a.h, b.y + b.h);
	float inter = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
	float uni = a.w * a.h + b.w * b.h - inter;
	return uni > 0.0f ? inter / uni : 0.0f;
}

} // namespace

const char *nude_class_name(int cls)
{
	return cls >= 0 && cls < NUDE_CLASS_COUNT ? kNames[cls] : "?";
}

struct NudeDetector::Impl {
	std::unique_ptr<Ort::Session> session;
	int input_size = 640;
	std::vector<float> input;
};

NudeDetector::NudeDetector() : impl_(std::make_unique<Impl>()) {}
NudeDetector::~NudeDetector() = default;

bool NudeDetector::loaded() const
{
	return impl_->session != nullptr;
}

bool NudeDetector::load(const std::string &model_path, int input_size, std::string &error, int accel)
{
	try {
		Ort::SessionOptions opts = make_session_options(1, accel == 0 ? OrtAccel::Auto : OrtAccel::CpuOnly, &provider_);
		impl_->session = std::make_unique<Ort::Session>(ort_env(), ort_path(model_path).c_str(), opts);
		impl_->input_size = std::max(32, input_size / 32 * 32);
		return true;
	} catch (const std::exception &e) {
		error = e.what();
		impl_->session.reset();
		return false;
	}
}

std::vector<NudeBox> NudeDetector::detect(const uint8_t *rgba, int w, int h, int stride, float min_score,
					  float nms_threshold)
{
	std::vector<NudeBox> result;
	if (!impl_->session || w <= 0 || h <= 0)
		return result;

	// 长边缩放到 input_size，短边补到 32 的倍数
	const int L = impl_->input_size;
	const float scale = (float)L / (float)std::max(w, h);
	const int rw = std::clamp((int)std::lround(w * scale), 1, L);
	const int rh = std::clamp((int)std::lround(h * scale), 1, L);
	const int iw = (rw + 31) / 32 * 32;
	const int ih = (rh + 31) / 32 * 32;
	const size_t plane = (size_t)iw * ih;
	impl_->input.assign(plane * 3, 0.0f);
	float *pr = impl_->input.data();
	float *pg = pr + plane;
	float *pb = pg + plane;

	// 尺寸不变（插件里画面已在 GPU 上缩到 640 以内）：直接转成 RGB 平面并 /255
	if (rw == w && rh == h) {
		const float k = 1.0f / 255.0f;
		for (int y = 0; y < h; y++) {
			const uint8_t *row = rgba + (size_t)y * stride;
			size_t o = (size_t)y * iw;
			for (int x = 0; x < w; x++) {
				pr[o + x] = row[x * 4 + 0] * k;
				pg[o + x] = row[x * 4 + 1] * k;
				pb[o + x] = row[x * 4 + 2] * k;
			}
		}
	}
	// 否则双线性缩放（像素中心对齐），同时转成 RGB 平面并 /255
	const float sx = (float)w / rw, sy = (float)h / rh;
	for (int y = 0; y < rh && !(rw == w && rh == h); y++) {
		float fy = (y + 0.5f) * sy - 0.5f;
		int y0 = std::clamp((int)std::floor(fy), 0, h - 1);
		int y1 = std::min(y0 + 1, h - 1);
		float ty = std::clamp(fy - y0, 0.0f, 1.0f);
		const uint8_t *r0 = rgba + (size_t)y0 * stride;
		const uint8_t *r1 = rgba + (size_t)y1 * stride;
		for (int x = 0; x < rw; x++) {
			float fx = (x + 0.5f) * sx - 0.5f;
			int x0 = std::clamp((int)std::floor(fx), 0, w - 1);
			int x1 = std::min(x0 + 1, w - 1);
			float tx = std::clamp(fx - x0, 0.0f, 1.0f);
			float v[3];
			for (int c = 0; c < 3; c++) {
				float a = r0[x0 * 4 + c] * (1 - tx) + r0[x1 * 4 + c] * tx;
				float b = r1[x0 * 4 + c] * (1 - tx) + r1[x1 * 4 + c] * tx;
				v[c] = (a * (1 - ty) + b * ty) / 255.0f;
			}
			size_t o = (size_t)y * iw + x;
			pr[o] = v[0];
			pg[o] = v[1];
			pb[o] = v[2];
		}
	}

	std::vector<Ort::Value> outputs;
	try {
		auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
		std::array<int64_t, 4> shape = {1, 3, ih, iw};
		Ort::Value tensor = Ort::Value::CreateTensor<float>(mem, impl_->input.data(), impl_->input.size(),
								    shape.data(), shape.size());
		const char *in_names[] = {"images"};
		const char *out_names[] = {"output0"};
		outputs = impl_->session->Run(Ort::RunOptions{nullptr}, in_names, &tensor, 1, out_names, 1);
	} catch (const std::exception &) {
		return result;
	}

	// 输出 [1, 4 + 18, N]，按通道排列
	auto info = outputs[0].GetTensorTypeAndShapeInfo();
	auto oshape = info.GetShape();
	if (oshape.size() != 3 || oshape[1] != 4 + NUDE_CLASS_COUNT)
		return result;
	const int64_t n = oshape[2];
	const float *out = outputs[0].GetTensorData<float>();

	std::vector<NudeBox> cand;
	for (int64_t i = 0; i < n; i++) {
		int best = 0;
		float best_s = out[4 * n + i];
		for (int c = 1; c < NUDE_CLASS_COUNT; c++) {
			float s = out[(4 + c) * n + i];
			if (s > best_s) {
				best_s = s;
				best = c;
			}
		}
		if (best_s < min_score)
			continue;
		float cx = out[0 * n + i], cy = out[1 * n + i], bw = out[2 * n + i], bh = out[3 * n + i];
		NudeBox b;
		b.x = (cx - bw / 2) / scale;
		b.y = (cy - bh / 2) / scale;
		b.w = bw / scale;
		b.h = bh / scale;
		// 裁到原图范围
		float x2 = std::min(b.x + b.w, (float)w), y2 = std::min(b.y + b.h, (float)h);
		b.x = std::max(0.0f, b.x);
		b.y = std::max(0.0f, b.y);
		b.w = x2 - b.x;
		b.h = y2 - b.y;
		if (b.w <= 0 || b.h <= 0)
			continue;
		b.score = best_s;
		b.cls = best;
		cand.push_back(b);
	}

	// 按类别分别 NMS（官方是类别无关的；分开做可避免重叠的不同部位互相抑制）
	std::sort(cand.begin(), cand.end(), [](const NudeBox &a, const NudeBox &b) { return a.score > b.score; });
	std::vector<bool> removed(cand.size(), false);
	for (size_t i = 0; i < cand.size(); i++) {
		if (removed[i])
			continue;
		result.push_back(cand[i]);
		for (size_t j = i + 1; j < cand.size(); j++) {
			if (!removed[j] && cand[j].cls == cand[i].cls && iou(cand[i], cand[j]) > nms_threshold)
				removed[j] = true;
		}
	}
	return result;
}
