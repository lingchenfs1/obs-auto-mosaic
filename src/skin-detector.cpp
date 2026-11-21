#include "skin-detector.hpp"

#include "face-detector.hpp"

#include "ort-runtime.hpp"

#include <algorithm>
#include <array>
#include <cmath>

namespace {

Ort::Env &ort_env()
{
	static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "obs-face-mosaic-skin");
	return env;
}

constexpr int kSegSize = 192; // PP-HumanSeg 输入固定 192x192
constexpr int kGridW = 160;   // 分析网格宽度

// 4 邻接连通块标记，返回每块的格子下标
std::vector<std::vector<int>> components(const std::vector<uint8_t> &mask, int gw, int gh, std::vector<int> &label)
{
	std::vector<std::vector<int>> comps;
	label.assign(mask.size(), -1);
	std::vector<int> stack;
	for (int i = 0; i < gw * gh; i++) {
		if (!mask[i] || label[i] >= 0)
			continue;
		int id = (int)comps.size();
		comps.emplace_back();
		stack.push_back(i);
		label[i] = id;
		while (!stack.empty()) {
			int c = stack.back();
			stack.pop_back();
			comps[id].push_back(c);
			int cx = c % gw, cy = c / gw;
			const int nb[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
			for (auto &d : nb) {
				int nx = cx + d[0], ny = cy + d[1];
				if (nx < 0 || ny < 0 || nx >= gw || ny >= gh)
					continue;
				int n = ny * gw + nx;
				if (mask[n] && label[n] < 0) {
					label[n] = id;
					stack.push_back(n);
				}
			}
		}
	}
	return comps;
}

} // namespace

struct SkinDetector::Impl {
	std::unique_ptr<Ort::Session> seg;
	FaceDetector face;
	std::vector<float> seg_input;
};

SkinDetector::SkinDetector() : impl_(std::make_unique<Impl>()) {}
SkinDetector::~SkinDetector() = default;

bool SkinDetector::loaded() const
{
	return impl_->seg != nullptr && impl_->face.loaded();
}

bool SkinDetector::load(const std::string &seg_model, const std::string &face_model, std::string &error,
			int accel)
{
	try {
		Ort::SessionOptions opts = make_session_options(1, accel == 0 ? OrtAccel::Auto : OrtAccel::CpuOnly, &provider_);
		impl_->seg = std::make_unique<Ort::Session>(ort_env(), ort_path(seg_model).c_str(), opts);
	} catch (const std::exception &e) {
		error = e.what();
		impl_->seg.reset();
		return false;
	}
	if (!impl_->face.load(face_model, error, accel)) {
		impl_->seg.reset();
		return false;
	}
	return true;
}

std::vector<SkinBox> SkinDetector::detect(const uint8_t *rgba, int w, int h, int stride, const SkinParams &params,
					  std::vector<SkinPerson> *persons)
{
	std::vector<SkinBox> result;
	if (!loaded() || w <= 0 || h <= 0)
		return result;

	// ---- 1. 人体分割：缩到 192x192，RGB，(v/255 - 0.5) / 0.5 ----
	const int S = kSegSize;
	impl_->seg_input.resize(3 * S * S);
	float *in = impl_->seg_input.data();
	for (int y = 0; y < S; y++) {
		float fy = std::clamp((y + 0.5f) * h / S - 0.5f, 0.0f, (float)(h - 1));
		int y0 = (int)fy, y1 = std::min(y0 + 1, h - 1);
		float ty = fy - y0;
		for (int x = 0; x < S; x++) {
			float fx = std::clamp((x + 0.5f) * w / S - 0.5f, 0.0f, (float)(w - 1));
			int x0 = (int)fx, x1 = std::min(x0 + 1, w - 1);
			float tx = fx - x0;
			for (int c = 0; c < 3; c++) {
				float a = rgba[(size_t)y0 * stride + x0 * 4 + c] * (1 - tx) +
					  rgba[(size_t)y0 * stride + x1 * 4 + c] * tx;
				float b = rgba[(size_t)y1 * stride + x0 * 4 + c] * (1 - tx) +
					  rgba[(size_t)y1 * stride + x1 * 4 + c] * tx;
				in[c * S * S + y * S + x] = ((a * (1 - ty) + b * ty) / 255.0f - 0.5f) / 0.5f;
			}
		}
	}
	std::vector<Ort::Value> outputs;
	try {
		auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
		std::array<int64_t, 4> shape = {1, 3, S, S};
		Ort::Value tensor = Ort::Value::CreateTensor<float>(mem, in, impl_->seg_input.size(), shape.data(),
								    shape.size());
		const char *in_names[] = {"x"};
		const char *out_names[] = {"save_infer_model/scale_0.tmp_1"};
		outputs = impl_->seg->Run(Ort::RunOptions{nullptr}, in_names, &tensor, 1, out_names, 1);
	} catch (const std::exception &) {
		return result;
	}
	const float *seg = outputs[0].GetTensorData<float>(); // [1, 2, S, S]：背景、人

	// ---- 2. 分析网格：每格取像素平均颜色，人体概率按格子中心双线性取样 ----
	const int gw = kGridW;
	const int gh = std::max(1, (int)std::lround((double)kGridW * h / w));
	const int n = gw * gh;
	std::vector<float> Y(n), Cr(n), Cb(n);
	std::vector<uint8_t> person(n), skin(n), face(n, 0), skin_any(n);
	for (int gy = 0; gy < gh; gy++) {
		int py0 = gy * h / gh, py1 = std::max(py0 + 1, (gy + 1) * h / gh);
		for (int gx = 0; gx < gw; gx++) {
			int px0 = gx * w / gw, px1 = std::max(px0 + 1, (gx + 1) * w / gw);
			double r = 0, g = 0, b = 0;
			int cnt = 0;
			for (int y = py0; y < py1 && y < h; y++)
				for (int x = px0; x < px1 && x < w; x++, cnt++) {
					const uint8_t *p = rgba + (size_t)y * stride + x * 4;
					r += p[0];
					g += p[1];
					b += p[2];
				}
			cnt = std::max(cnt, 1);
			float R = (float)(r / cnt), G = (float)(g / cnt), B = (float)(b / cnt);
			int i = gy * gw + gx;
			Y[i] = 0.299f * R + 0.587f * G + 0.114f * B;
			Cr[i] = (R - Y[i]) * 0.713f + 128.0f;
			Cb[i] = (B - Y[i]) * 0.564f + 128.0f;
			skin[i] = Cr[i] >= 133 && Cr[i] <= 173 && Cb[i] >= 77 && Cb[i] <= 127 && Y[i] > 40;

			float sx = std::clamp((gx + 0.5f) * S / gw - 0.5f, 0.0f, (float)(S - 1));
			float sy = std::clamp((gy + 0.5f) * S / gh - 0.5f, 0.0f, (float)(S - 1));
			int x0 = (int)sx, x1 = std::min(x0 + 1, S - 1), y0 = (int)sy, y1 = std::min(y0 + 1, S - 1);
			float tx = sx - x0, ty = sy - y0;
			auto diff = [&](int yy, int xx) { return seg[S * S + yy * S + xx] - seg[yy * S + xx]; };
			float v = (diff(y0, x0) * (1 - tx) + diff(y0, x1) * tx) * (1 - ty) +
				  (diff(y1, x0) * (1 - tx) + diff(y1, x1) * tx) * ty;
			person[i] = v > 0;
		}
	}
	skin_any = skin;

	// ---- 3. 人脸：排除脸部；取脸中间的肤色做自适应模型；定出脸下方的躯干区域 ----
	struct Torso {
		int x0, y0, x1, y1;
		std::vector<uint8_t> adapt;
	};
	std::vector<Torso> torsos;
	const float sc = (float)gw / w;
	for (const FaceBox &fb : impl_->face.detect(rgba, w, h, stride, params.face_threshold)) {
		int fcx = (int)((fb.x + fb.w / 2) * sc), fcy = (int)((fb.y + fb.h / 2) * sc);
		if (fcx < 0 || fcy < 0 || fcx >= gw || fcy >= gh || !person[fcy * gw + fcx])
			continue; // 脸不在分割出的人体上（多半是背景误检）
		int x0 = std::max(0, (int)((fb.x - fb.w * 0.15f) * sc)), y0 = std::max(0, (int)((fb.y - fb.h * 0.25f) * sc));
		int x1 = std::min(gw, (int)((fb.x + fb.w * 1.15f) * sc) + 1);
		int y1 = std::min(gh, (int)((fb.y + fb.h * 1.15f) * sc) + 1);
		for (int y = y0; y < y1; y++)
			for (int x = x0; x < x1; x++)
				face[y * gw + x] = 1;

		int cx0 = std::max(0, (int)((fb.x + fb.w * 0.25f) * sc)), cy0 = std::max(0, (int)((fb.y + fb.h * 0.35f) * sc));
		int cx1 = std::min(gw, std::max(cx0 + 1, (int)((fb.x + fb.w * 0.75f) * sc)));
		int cy1 = std::min(gh, std::max(cy0 + 1, (int)((fb.y + fb.h * 0.8f) * sc)));
		double mr = 0, mb = 0, my = 0;
		int k = 0;
		for (int y = cy0; y < cy1; y++)
			for (int x = cx0; x < cx1; x++, k++) {
				mr += Cr[y * gw + x];
				mb += Cb[y * gw + x];
				my += Y[y * gw + x];
			}
		if (k < 4)
			continue;
		mr /= k;
		mb /= k;
		my /= k;
		double srr = 0, sbb = 0, srb = 0;
		for (int y = cy0; y < cy1; y++)
			for (int x = cx0; x < cx1; x++) {
				double dr = Cr[y * gw + x] - mr, db = Cb[y * gw + x] - mb;
				srr += dr * dr;
				sbb += db * db;
				srb += dr * db;
			}
		// 协方差（N-1）+ 4·I，求逆
		double a = srr / (k - 1) + 4.0, d = sbb / (k - 1) + 4.0, bcov = srb / (k - 1);
		double det = a * d - bcov * bcov;
		if (det <= 1e-9)
			continue;
		double ia = d / det, id = a / det, ib = -bcov / det;

		Torso t;
		t.adapt.assign(n, 0);
		double ylo = my * 0.35, yhi = std::min(255.0, my * 1.8);
		for (int i = 0; i < n; i++) {
			double dr = Cr[i] - mr, db = Cb[i] - mb;
			double md = dr * dr * ia + 2 * dr * db * ib + db * db * id;
			if (md < 9.0 && Y[i] > ylo && Y[i] < yhi) {
				t.adapt[i] = 1;
				skin_any[i] = 1;
			}
		}
		float fcxf = (fb.x + fb.w / 2) * sc;
		t.x0 = std::max(0, (int)(fcxf - fb.w * 1.5f * sc));
		t.x1 = std::min(gw, (int)(fcxf + fb.w * 1.5f * sc));
		t.y0 = std::max(0, (int)((fb.y + fb.h * 1.3f) * sc));
		t.y1 = std::min(gh, (int)((fb.y + fb.h * 3.8f) * sc));
		torsos.push_back(std::move(t));
	}

	// ---- 4. 每个人（人体连通块）算皮肤占比，判定是否裸露 ----
	std::vector<uint8_t> exposed(n);
	for (int i = 0; i < n; i++)
		exposed[i] = person[i] && skin_any[i] && !face[i];

	std::vector<int> label;
	auto comps = components(person, gw, gh, label);
	std::vector<uint8_t> in_comp(n, 0);
	for (size_t ci = 0; ci < comps.size(); ci++) {
		const auto &cells = comps[ci];
		if ((int)cells.size() < std::max(4, n / 500))
			continue;
		int body_cells = 0, skin_cells = 0;
		int bx0 = gw, by0 = gh, bx1 = -1, by1 = -1;
		for (int c : cells) {
			int x = c % gw, y = c / gw;
			bx0 = std::min(bx0, x);
			by0 = std::min(by0, y);
			bx1 = std::max(bx1, x);
			by1 = std::max(by1, y);
			if (face[c])
				continue;
			body_cells++;
			skin_cells += exposed[c];
		}
		float body = body_cells ? (float)skin_cells / body_cells : 0.0f;
		float torso = -1.0f;
		for (const Torso &t : torsos) {
			int tcells = 0, tskin = 0;
			for (int y = t.y0; y < t.y1; y++)
				for (int x = t.x0; x < t.x1; x++) {
					int i = y * gw + x;
					if (label[i] != (int)ci || face[i])
						continue;
					tcells++;
					tskin += t.adapt[i];
				}
			if (tcells >= 10)
				torso = std::max(torso, (float)tskin / tcells);
		}
		bool is_exposed = body >= params.body_threshold || torso >= params.torso_threshold;
		if (persons)
			persons->push_back({bx0, by0, bx1, by1, body, torso, is_exposed});
		if (!is_exposed)
			continue;

		// ---- 5. 裸露的人：身上的皮肤块（不含脸）各输出一个框 ----
		std::fill(in_comp.begin(), in_comp.end(), 0);
		for (int c : cells)
			in_comp[c] = exposed[c];
		std::vector<int> blabel;
		for (const auto &blob : components(in_comp, gw, gh, blabel)) {
			if (blob.size() < 3)
				continue;
			int x0 = gw, y0 = gh, x1 = -1, y1 = -1;
			for (int c : blob) {
				x0 = std::min(x0, c % gw);
				y0 = std::min(y0, c / gw);
				x1 = std::max(x1, c % gw);
				y1 = std::max(y1, c / gw);
			}
			float cw = (float)w / gw, ch = (float)h / gh;
			result.push_back({x0 * cw, y0 * ch, (x1 - x0 + 1) * cw, (y1 - y0 + 1) * ch, body, torso});
		}
	}
	return result;
}
