#include "scene-detector.hpp"

#include "ort-runtime.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <vector>

namespace {

Ort::Env &ort_env()
{
	static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "obs-face-mosaic-scene");
	return env;
}

constexpr int kSize = 224;
constexpr int kCrops = 3;
constexpr float kLogitScale = 100.0f; // CLIP 的 logit_scale（exp 后的值）
const float kMean[3] = {0.48145466f, 0.4578275f, 0.40821073f};
const float kStd[3] = {0.26862954f, 0.26130258f, 0.27577711f};

} // namespace

struct SceneDetector::Impl {
	std::unique_ptr<Ort::Session> session;
	int dim = 0, n_gore = 0, n_violence = 0, n_safe = 0;
	std::vector<float> prompts; // [n][dim]，已归一化
	std::vector<float> input;
};

SceneDetector::SceneDetector() : impl_(std::make_unique<Impl>()) {}
SceneDetector::~SceneDetector() = default;

bool SceneDetector::loaded() const
{
	return impl_->session && !impl_->prompts.empty();
}

bool SceneDetector::load(const std::string &vision_model, const std::string &prompts, std::string &error)
{
	impl_->session.reset();
	impl_->prompts.clear();

	std::ifstream f(prompts, std::ios::binary);
	char magic[8] = {};
	int32_t hdr[4] = {};
	if (!f.read(magic, 8) || std::memcmp(magic, "CLIPPRM1", 8) != 0 || !f.read((char *)hdr, sizeof(hdr))) {
		error = "文字向量文件格式不对：" + prompts;
		return false;
	}
	impl_->dim = hdr[0];
	impl_->n_gore = hdr[1];
	impl_->n_violence = hdr[2];
	impl_->n_safe = hdr[3];
	size_t n = (size_t)hdr[1] + hdr[2] + hdr[3];
	if (impl_->dim <= 0 || n == 0 || n > 1000) {
		error = "文字向量文件内容不对：" + prompts;
		return false;
	}
	impl_->prompts.resize(n * impl_->dim);
	if (!f.read((char *)impl_->prompts.data(), impl_->prompts.size() * sizeof(float))) {
		error = "文字向量文件不完整：" + prompts;
		impl_->prompts.clear();
		return false;
	}
	try {
		// int8 量化模型：CoreML / DirectML 不支持其中的量化算子，固定用 CPU（三块画面一批约 30 ms）
		impl_->session = make_session(ort_env(), vision_model, 1, OrtAccel::CpuOnly);
	} catch (const std::exception &e) {
		error = e.what();
		impl_->prompts.clear();
		return false;
	}
	return true;
}

SceneScores SceneDetector::detect(const uint8_t *rgba, int w, int h, int stride)
{
	SceneScores result;
	if (!loaded() || w <= 0 || h <= 0)
		return result;

	// 沿长边取左 / 中 / 右三块正方形，各缩放到 224x224（双线性），RGB，按 CLIP 的均值方差归一化
	const int m = std::min(w, h);
	const int L = std::max(w, h);
	const int offs[kCrops] = {0, (L - m) / 2, L - m};
	const size_t plane = (size_t)kSize * kSize;
	impl_->input.resize(kCrops * 3 * plane);
	for (int k = 0; k < kCrops; k++) {
		int ox = w >= h ? offs[k] : 0, oy = w >= h ? 0 : offs[k];
		float *dst = impl_->input.data() + (size_t)k * 3 * plane;
		for (int y = 0; y < kSize; y++) {
			float fy = std::clamp((y + 0.5f) * m / kSize - 0.5f, 0.0f, (float)(m - 1));
			int y0 = (int)fy, y1 = std::min(y0 + 1, m - 1);
			float ty = fy - y0;
			const uint8_t *r0 = rgba + (size_t)(oy + y0) * stride;
			const uint8_t *r1 = rgba + (size_t)(oy + y1) * stride;
			for (int x = 0; x < kSize; x++) {
				float fx = std::clamp((x + 0.5f) * m / kSize - 0.5f, 0.0f, (float)(m - 1));
				int x0 = (int)fx, x1 = std::min(x0 + 1, m - 1);
				float tx = fx - x0;
				for (int c = 0; c < 3; c++) {
					float a = r0[(ox + x0) * 4 + c] * (1 - tx) + r0[(ox + x1) * 4 + c] * tx;
					float b = r1[(ox + x0) * 4 + c] * (1 - tx) + r1[(ox + x1) * 4 + c] * tx;
					dst[c * plane + (size_t)y * kSize + x] =
						((a * (1 - ty) + b * ty) / 255.0f - kMean[c]) / kStd[c];
				}
			}
		}
	}

	std::vector<Ort::Value> out;
	try {
		auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
		std::array<int64_t, 4> shape = {kCrops, 3, kSize, kSize};
		Ort::Value tensor = Ort::Value::CreateTensor<float>(mem, impl_->input.data(), impl_->input.size(),
								    shape.data(), shape.size());
		const char *in_names[] = {"pixel_values"};
		const char *out_names[] = {"image_embeds"};
		out = impl_->session->Run(Ort::RunOptions{nullptr}, in_names, &tensor, 1, out_names, 1);
	} catch (const std::exception &) {
		return result;
	}
	auto shape = out[0].GetTensorTypeAndShapeInfo().GetShape();
	if (shape.size() != 2 || shape[0] != kCrops || shape[1] != impl_->dim)
		return result;
	const float *emb = out[0].GetTensorData<float>();

	const int D = impl_->dim;
	const int n = impl_->n_gore + impl_->n_violence + impl_->n_safe;
	std::vector<float> logits(n);
	for (int k = 0; k < kCrops; k++) {
		const float *e = emb + (size_t)k * D;
		double norm = 0;
		for (int d = 0; d < D; d++)
			norm += (double)e[d] * e[d];
		float inv = norm > 0 ? (float)(1.0 / std::sqrt(norm)) : 0.0f;
		float mx = -1e30f;
		for (int i = 0; i < n; i++) {
			const float *p = impl_->prompts.data() + (size_t)i * D;
			double dot = 0;
			for (int d = 0; d < D; d++)
				dot += (double)e[d] * p[d];
			logits[i] = kLogitScale * (float)dot * inv;
			mx = std::max(mx, logits[i]);
		}
		double sum = 0;
		for (int i = 0; i < n; i++) {
			logits[i] = std::exp(logits[i] - mx);
			sum += logits[i];
		}
		float gore = 0, violence = 0;
		for (int i = 0; i < impl_->n_gore; i++)
			gore += logits[i];
		for (int i = impl_->n_gore; i < impl_->n_gore + impl_->n_violence; i++)
			violence += logits[i];
		result.gore = std::max(result.gore, (float)(gore / sum));
		result.violence = std::max(result.violence, (float)(violence / sum));
	}
	return result;
}
