#include "text-detector.hpp"

#include "ort-options.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <list>
#include <regex>
#include <unordered_map>

namespace {

Ort::Env &ort_env()
{
	static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "obs-face-mosaic-text");
	return env;
}

constexpr int kRecHeight = 48;
constexpr int kRecMinWidth = 320;
constexpr int kRecMaxWidth = 48 * 30; // 超长行截断（约 30 倍高）
constexpr size_t kCacheMax = 1024;

// ---- UTF-8 ----
bool next_codepoint(const std::string &s, size_t &i, uint32_t &cp)
{
	if (i >= s.size())
		return false;
	unsigned char c = (unsigned char)s[i];
	int len = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 1;
	if (len == 1) {
		cp = c;
	} else {
		cp = c & (0xFF >> (len + 1));
		for (int k = 1; k < len; k++) {
			if (i + k >= s.size())
				break;
			cp = (cp << 6) | ((unsigned char)s[i + k] & 0x3F);
		}
	}
	i += len;
	return true;
}

void append_utf8(std::string &out, uint32_t cp)
{
	if (cp < 0x80) {
		out += (char)cp;
	} else if (cp < 0x800) {
		out += (char)(0xC0 | (cp >> 6));
		out += (char)(0x80 | (cp & 0x3F));
	} else if (cp < 0x10000) {
		out += (char)(0xE0 | (cp >> 12));
		out += (char)(0x80 | ((cp >> 6) & 0x3F));
		out += (char)(0x80 | (cp & 0x3F));
	} else {
		out += (char)(0xF0 | (cp >> 18));
		out += (char)(0x80 | ((cp >> 12) & 0x3F));
		out += (char)(0x80 | ((cp >> 6) & 0x3F));
		out += (char)(0x80 | (cp & 0x3F));
	}
}

uint64_t fnv1a(uint64_t h, uint32_t v)
{
	for (int k = 0; k < 4; k++) {
		h ^= (v >> (k * 8)) & 0xFF;
		h *= 1099511628211ULL;
	}
	return h;
}

} // namespace

std::string normalize_text(const std::string &utf8)
{
	std::string out;
	size_t i = 0;
	uint32_t cp;
	while (next_codepoint(utf8, i, cp)) {
		if (cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' || cp == 0x3000 || cp == 0xA0)
			continue;
		if (cp >= 0xFF01 && cp <= 0xFF5E)
			cp -= 0xFEE0; // 全角 ASCII → 半角
		if (cp >= 'A' && cp <= 'Z')
			cp += 32;
		append_utf8(out, cp);
	}
	return out;
}

std::vector<std::string> parse_keywords(const std::string &list)
{
	std::vector<std::string> out;
	std::string cur;
	auto flush = [&] {
		std::string n = normalize_text(cur);
		if (!n.empty())
			out.push_back(n);
		cur.clear();
	};
	size_t i = 0, prev = 0;
	uint32_t cp;
	while (next_codepoint(list, i, cp)) {
		bool sep = cp == '\n' || cp == '\r' || cp == ',' || cp == ';' || cp == 0xFF0C || cp == 0xFF1B ||
			   cp == 0x3001;
		if (sep)
			flush();
		else
			cur.append(list, prev, i - prev);
		prev = i;
	}
	flush();
	return out;
}

bool match_sensitive(const std::string &text, const TextRules &rules, std::string &reason)
{
	// 规整后匹配：去空格（「6222 0212 …」也能连成一串）、全角转半角、小写
	static const std::regex re_phone("(^|[^0-9])1[3-9][0-9]{9}([^0-9]|$)");
	static const std::regex re_id("(^|[^0-9])[1-9][0-9]{5}(19|20)[0-9]{2}(0[1-9]|1[0-2])(0[1-9]|[12][0-9]|3[01])"
				      "[0-9]{3}[0-9x]([^0-9]|$)");
	static const std::regex re_bank("(^|[^0-9])[0-9]{16,19}([^0-9]|$)");
	static const std::regex re_email("[a-z0-9._%+-]+@[a-z0-9-]+(\\.[a-z0-9-]+)*\\.[a-z]{2,}");
	static const std::regex re_url("(https?://|www\\.)[a-z0-9]|[a-z0-9-]+\\.(com|cn|net|org|top|xyz|cc|tv|me|io|"
				       "vip|club|site|shop|live|link)([^a-z]|$)");
	static const std::regex re_im("(微信号|微信|威信|薇信|v信|vx|wx|weixin|qq号|qq|扣扣)[:号]?[a-z0-9_-]{5,}");

	std::string n = normalize_text(text);
	if (n.empty())
		return false;
	struct Rule {
		bool on;
		const std::regex *re;
		const char *name;
	};
	const Rule list[] = {
		{rules.phone, &re_phone, "手机号"}, {rules.id_card, &re_id, "身份证号"}, {rules.bank_card, &re_bank, "银行卡号"},
		{rules.email, &re_email, "邮箱"},   {rules.url, &re_url, "网址"},       {rules.im, &re_im, "社交账号"},
	};
	for (const Rule &r : list) {
		if (r.on && std::regex_search(n, *r.re)) {
			reason = r.name;
			return true;
		}
	}
	for (const std::string &kw : rules.keywords) {
		if (!kw.empty() && n.find(kw) != std::string::npos) {
			reason = "关键词：" + kw;
			return true;
		}
	}
	return false;
}

struct TextDetector::Impl {
	std::unique_ptr<Ort::Session> det;
	std::unique_ptr<Ort::Session> rec;
	std::vector<std::string> chars; // 下标 0 为 CTC 空白
	std::vector<float> det_input, rec_input;

	struct CacheEntry {
		std::string text;
		float score;
		std::list<uint64_t>::iterator lru;
	};
	std::unordered_map<uint64_t, CacheEntry> cache;
	std::list<uint64_t> lru; // 最近用过的在前

	bool recognize(const uint8_t *rgba, int stride, int x0, int y0, int cw, int ch, std::string &text,
		       float &score);
};

TextDetector::TextDetector() : impl_(std::make_unique<Impl>()) {}
TextDetector::~TextDetector() = default;

bool TextDetector::loaded() const
{
	return impl_->det && impl_->rec && impl_->chars.size() > 2;
}

bool TextDetector::load(const std::string &det_model, const std::string &rec_model, std::string &error)
{
	try {
		Ort::SessionOptions opts = make_session_options(1);
		impl_->det = std::make_unique<Ort::Session>(ort_env(), det_model.c_str(), opts);
		impl_->rec = std::make_unique<Ort::Session>(ort_env(), rec_model.c_str(), opts);

		// 字典在识别模型的元数据里，每行一个字
		Ort::AllocatorWithDefaultOptions alloc;
		Ort::ModelMetadata md = impl_->rec->GetModelMetadata();
		Ort::AllocatedStringPtr dict = md.LookupCustomMetadataMapAllocated("character", alloc);
		if (!dict) {
			error = "识别模型里没有字典（元数据 character）";
			impl_->det.reset();
			impl_->rec.reset();
			return false;
		}
		impl_->chars.clear();
		impl_->chars.push_back(""); // CTC 空白
		std::string all = dict.get(), cur;
		for (char c : all) {
			if (c == '\n') {
				impl_->chars.push_back(cur);
				cur.clear();
			} else if (c != '\r') {
				cur += c;
			}
		}
		if (!cur.empty())
			impl_->chars.push_back(cur);
		impl_->chars.push_back(" "); // PaddleOCR 默认在字典末尾加空格
		impl_->cache.clear();
		impl_->lru.clear();
		return true;
	} catch (const std::exception &e) {
		error = e.what();
		impl_->det.reset();
		impl_->rec.reset();
		return false;
	}
}

bool TextDetector::Impl::recognize(const uint8_t *rgba, int stride, int x0, int y0, int cw, int ch,
				   std::string &text, float &score)
{
	int tw = std::clamp((int)std::ceil((double)kRecHeight * cw / ch), 1, kRecMaxWidth);
	int W = std::max(kRecMinWidth, tw);
	const size_t plane = (size_t)kRecHeight * W;
	rec_input.assign(plane * 3, 0.0f);
	float *pb = rec_input.data(), *pg = pb + plane, *pr = pg + plane;
	for (int y = 0; y < kRecHeight; y++) {
		float fy = std::clamp((y + 0.5f) * ch / kRecHeight - 0.5f, 0.0f, (float)(ch - 1));
		int ya = (int)fy, yb = std::min(ya + 1, ch - 1);
		float ty = fy - ya;
		const uint8_t *ra = rgba + (size_t)(y0 + ya) * stride;
		const uint8_t *rb = rgba + (size_t)(y0 + yb) * stride;
		for (int x = 0; x < tw; x++) {
			float fx = std::clamp((x + 0.5f) * cw / tw - 0.5f, 0.0f, (float)(cw - 1));
			int xa = (int)fx, xb = std::min(xa + 1, cw - 1);
			float tx = fx - xa;
			float v[3];
			for (int c = 0; c < 3; c++) {
				float a = ra[(x0 + xa) * 4 + c] * (1 - tx) + ra[(x0 + xb) * 4 + c] * tx;
				float b = rb[(x0 + xa) * 4 + c] * (1 - tx) + rb[(x0 + xb) * 4 + c] * tx;
				v[c] = ((a * (1 - ty) + b * ty) / 255.0f - 0.5f) / 0.5f;
			}
			size_t o = (size_t)y * W + x;
			pr[o] = v[0];
			pg[o] = v[1];
			pb[o] = v[2];
		}
	}

	std::vector<Ort::Value> out;
	try {
		auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
		std::array<int64_t, 4> shape = {1, 3, kRecHeight, W};
		Ort::Value tensor = Ort::Value::CreateTensor<float>(mem, rec_input.data(), rec_input.size(),
								    shape.data(), shape.size());
		const char *in_names[] = {"x"};
		const char *out_names[] = {"softmax_11.tmp_0"};
		out = rec->Run(Ort::RunOptions{nullptr}, in_names, &tensor, 1, out_names, 1);
	} catch (const std::exception &) {
		return false;
	}
	auto shape = out[0].GetTensorTypeAndShapeInfo().GetShape();
	if (shape.size() != 3 || shape[2] != (int64_t)chars.size())
		return false;
	const int64_t T = shape[1], C = shape[2];
	const float *p = out[0].GetTensorData<float>();

	text.clear();
	double sum = 0;
	int n = 0;
	int64_t prev = -1;
	for (int64_t t = 0; t < T; t++) {
		const float *row = p + t * C;
		int64_t best = std::max_element(row, row + C) - row;
		if (best != 0 && best != prev) {
			text += chars[best];
			sum += row[best];
			n++;
		}
		prev = best;
	}
	score = n ? (float)(sum / n) : 0.0f;
	return true;
}

std::vector<TextLine> TextDetector::detect(const uint8_t *rgba, int w, int h, int stride, int det_limit)
{
	std::vector<TextLine> lines;
	if (!loaded() || w <= 0 || h <= 0)
		return lines;

	// ---- 检测：缩到 det_limit 以内、32 的倍数 ----
	float sc = std::min(1.0f, (float)det_limit / std::max(w, h));
	int rw = std::max(32, (int)std::lround(w * sc / 32.0f) * 32);
	int rh = std::max(32, (int)std::lround(h * sc / 32.0f) * 32);
	const size_t plane = (size_t)rw * rh;
	auto &in = impl_->det_input;
	in.resize(plane * 3);
	// 通道顺序 B、G、R，均值方差按 PaddleOCR 的写法依次作用在这三个通道上
	const float mean[3] = {0.485f, 0.456f, 0.406f}, stdv[3] = {0.229f, 0.224f, 0.225f};
	const int src_c[3] = {2, 1, 0};
	for (int y = 0; y < rh; y++) {
		float fy = std::clamp((y + 0.5f) * h / rh - 0.5f, 0.0f, (float)(h - 1));
		int ya = (int)fy, yb = std::min(ya + 1, h - 1);
		float ty = fy - ya;
		const uint8_t *ra = rgba + (size_t)ya * stride, *rb = rgba + (size_t)yb * stride;
		for (int x = 0; x < rw; x++) {
			float fx = std::clamp((x + 0.5f) * w / rw - 0.5f, 0.0f, (float)(w - 1));
			int xa = (int)fx, xb = std::min(xa + 1, w - 1);
			float tx = fx - xa;
			for (int k = 0; k < 3; k++) {
				int c = src_c[k];
				float a = ra[xa * 4 + c] * (1 - tx) + ra[xb * 4 + c] * tx;
				float b = rb[xa * 4 + c] * (1 - tx) + rb[xb * 4 + c] * tx;
				in[k * plane + (size_t)y * rw + x] = ((a * (1 - ty) + b * ty) / 255.0f - mean[k]) / stdv[k];
			}
		}
	}
	std::vector<Ort::Value> out;
	try {
		auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
		std::array<int64_t, 4> shape = {1, 3, rh, rw};
		Ort::Value tensor = Ort::Value::CreateTensor<float>(mem, in.data(), in.size(), shape.data(), shape.size());
		const char *in_names[] = {"x"};
		const char *out_names[] = {"sigmoid_0.tmp_0"};
		out = impl_->det->Run(Ort::RunOptions{nullptr}, in_names, &tensor, 1, out_names, 1);
	} catch (const std::exception &) {
		return lines;
	}
	const float *prob = out[0].GetTensorData<float>();

	// ---- 二值化 + 8 邻接连通块 → 外接矩形 ----
	std::vector<int32_t> label(plane, -1);
	std::vector<int> stack;
	struct Box {
		int x0, y0, x1, y1;
	};
	std::vector<Box> boxes;
	for (size_t i0 = 0; i0 < plane; i0++) {
		if (label[i0] >= 0 || prob[i0] <= 0.3f)
			continue;
		int id = (int)boxes.size();
		Box b = {rw, rh, -1, -1};
		double psum = 0;
		int cnt = 0;
		stack.push_back((int)i0);
		label[i0] = id;
		while (!stack.empty()) {
			int c = stack.back();
			stack.pop_back();
			int cx = c % rw, cy = c / rw;
			b.x0 = std::min(b.x0, cx);
			b.y0 = std::min(b.y0, cy);
			b.x1 = std::max(b.x1, cx);
			b.y1 = std::max(b.y1, cy);
			psum += prob[c];
			cnt++;
			for (int dy = -1; dy <= 1; dy++)
				for (int dx = -1; dx <= 1; dx++) {
					int nx = cx + dx, ny = cy + dy;
					if (nx < 0 || ny < 0 || nx >= rw || ny >= rh)
						continue;
					int nidx = ny * rw + nx;
					if (label[nidx] < 0 && prob[nidx] > 0.3f) {
						label[nidx] = id;
						stack.push_back(nidx);
					}
				}
		}
		boxes.push_back(b);
		int bw = b.x1 - b.x0 + 1, bh = b.y1 - b.y0 + 1;
		if (cnt < 10 || std::min(bw, bh) < 3 || psum / cnt < 0.5)
			boxes.back().x1 = -1; // 作废（编号仍占着，保证 label 一致）
	}

	const float kx = (float)w / rw, ky = (float)h / rh;
	for (const Box &b : boxes) {
		if (b.x1 < 0)
			continue;
		float bw = (float)(b.x1 - b.x0 + 1), bh = (float)(b.y1 - b.y0 + 1);
		float d = bw * bh * 1.5f / (2.0f * (bw + bh)); // DB 的 unclip：按面积 / 周长外扩
		int x0 = std::clamp((int)std::floor((b.x0 - d) * kx), 0, w - 1);
		int y0 = std::clamp((int)std::floor((b.y0 - d) * ky), 0, h - 1);
		int x1 = std::clamp((int)std::ceil((b.x1 + 1 + d) * kx), x0 + 1, w);
		int y1 = std::clamp((int)std::ceil((b.y1 + 1 + d) * ky), y0 + 1, h);
		int cw = x1 - x0, ch = y1 - y0;
		// 太小认不出；太高的不是横排文字行（多半是大片色块、图案）
		if (ch < 6 || cw < 6 || ch > h * 0.15f)
			continue;

		// ---- 识别（带缓存：位置按 8 像素量化 + 32x8 灰度缩略图量化后取哈希） ----
		uint64_t key = 1469598103934665603ULL;
		key = fnv1a(key, (uint32_t)(x0 / 8));
		key = fnv1a(key, (uint32_t)(y0 / 8));
		key = fnv1a(key, (uint32_t)(cw / 8));
		key = fnv1a(key, (uint32_t)(ch / 8));
		for (int gy = 0; gy < 8; gy++)
			for (int gx = 0; gx < 32; gx++) {
				int px = x0 + (gx * 2 + 1) * cw / 64, py = y0 + (gy * 2 + 1) * ch / 16;
				const uint8_t *p = rgba + (size_t)py * stride + px * 4;
				key = fnv1a(key, (uint32_t)((p[0] * 3 + p[1] * 6 + p[2]) / 10) >> 4);
			}

		TextLine line = {(float)x0, (float)y0, (float)cw, (float)ch, "", 0.0f, false};
		auto it = impl_->cache.find(key);
		if (it != impl_->cache.end()) {
			line.text = it->second.text;
			line.score = it->second.score;
			line.cached = true;
			impl_->lru.splice(impl_->lru.begin(), impl_->lru, it->second.lru);
		} else {
			if (!impl_->recognize(rgba, stride, x0, y0, cw, ch, line.text, line.score))
				continue;
			impl_->lru.push_front(key);
			impl_->cache[key] = {line.text, line.score, impl_->lru.begin()};
			while (impl_->cache.size() > kCacheMax) {
				impl_->cache.erase(impl_->lru.back());
				impl_->lru.pop_back();
			}
		}
		lines.push_back(std::move(line));
	}
	std::sort(lines.begin(), lines.end(), [](const TextLine &a, const TextLine &b) {
		return a.y + a.h / 2 < b.y + b.h / 2 - 4 || (std::abs(a.y + a.h / 2 - b.y - b.h / 2) <= 4 && a.x < b.x);
	});
	return lines;
}
