// 自动马赛克滤镜：人脸 + 不雅内容（NudeNet）+ 裸露皮肤（人体分割 + 肤色）+ 敏感文字（PP-OCRv4）
//
// 检测通道：人脸（YuNet）、不雅内容（NudeNet 320n / 640m）、裸露皮肤（PP-HumanSeg + 肤色）、
// 敏感文字（PP-OCR 检测 + 识别 + 规则 / 关键词）各有一个推理线程、自己的检测历史，慢的通道不拖慢快的。
// 前三个通道共用一份缩到 640 以内的画面；文字通道单独采集一份高分辨率（长边 1920 以内）画面。
//
// 每个新帧（渲染线程，video_tick 标记新帧，同一帧被多处渲染时只处理一次）：
//   1. 上游画面渲染进环形缓冲 ring[frame_seq % N]，N = 延迟帧数 + 1
//   2. 按检测频率把最新一帧缩到 640 以内，gs_stage_texture 到暂存面；
//      暂存面在下一帧才 map，避免 GPU 读回卡住渲染线程；map 出的像素连同帧号交给空闲的通道
// 输出（每次渲染）：
//   取 delay 帧之前的画面 s；每个通道在自己的检测历史里找帧号 <= s 的最近结果 A、> s 的最近结果 B：
//   同一个框（同一个 track id）在 A、B 之间插值，再并上按前后速度外推的位置；只在 A 里的照遮；
//   只在 B 里的（刚出现的）也提前遮上；没有 B（检测没跟上）时按 A 之前的速度外推。
//   不雅内容可选「整屏打码」：任何暴露部位出现时整个画面打码。
// 推理线程：结果按 IoU / 中心距离与上次同类别的框匹配（沿用 id），消失后继续保留 hold_ms。
// 兜底：任一启用的通道超过 kStallMs 没有新结果（含模型加载中）时整屏打码。

#include <obs-module.h>
#include <graphics/vec2.h>
#include <graphics/vec4.h>
#include <util/platform.h>

#include "face-detector.hpp"
#include "nude-detector.hpp"
#include "skin-detector.hpp"
#include "text-detector.hpp"
#include "ort-runtime.hpp"

#include "stb_image_write.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr uint64_t kStallMs = 1000;
constexpr uint64_t kTextStallMs = 3000; // 文字通道一次 OCR 可能要几百毫秒，兜底等久一点
constexpr uint32_t kTextCaptureMax = 1920;
constexpr int kMaxDetectSize = FaceDetector::kInputSize;

const char *kEffectSource = R"(
uniform float4x4 ViewProj;
uniform texture2d image;
uniform float2 tex_size;
uniform float block;
uniform float4 color;

sampler_state textureSampler {
	Filter    = Linear;
	AddressU  = Clamp;
	AddressV  = Clamp;
};

struct VertData {
	float4 pos : POSITION;
	float2 uv  : TEXCOORD0;
};

VertData VSDefault(VertData v_in)
{
	VertData vert_out;
	vert_out.pos = mul(float4(v_in.pos.xyz, 1.0), ViewProj);
	vert_out.uv  = v_in.uv;
	return vert_out;
}

float4 PSPixelate(VertData v_in) : TARGET
{
	float2 px = v_in.uv * tex_size;
	float2 cell = floor(px / block) * block;
	float step = block / 4.0;
	float4 sum = float4(0.0, 0.0, 0.0, 0.0);
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			float2 p = cell + (float2(float(i), float(j)) + 0.5) * step;
			sum += image.Sample(textureSampler, p / tex_size);
		}
	}
	return sum / 16.0;
}

float4 PSSolid(VertData v_in) : TARGET
{
	return color;
}

technique Pixelate
{
	pass
	{
		vertex_shader = VSDefault(v_in);
		pixel_shader  = PSPixelate(v_in);
	}
}

technique Solid
{
	pass
	{
		vertex_shader = VSDefault(v_in);
		pixel_shader  = PSSolid(v_in);
	}
}
)";

constexpr size_t kMaxHistory = 256;
constexpr int kMaxDelayFrames = 60;

enum ChannelKind { CH_FACE = 0, CH_NUDE = 1, CH_SKIN = 2, CH_TEXT = 3, CH_COUNT = 4 };
const char *kChannelName[CH_COUNT] = {"人脸", "不雅内容", "裸露皮肤", "敏感文字"};
constexpr int kClsFace = -1, kClsSkin = -2, kClsText = -3; // Track.cls；>= 0 为 NudeClass
const int kTextLimits[4] = {640, 800, 960, 1280};

// 不雅类别分组（设置里按组开关）
enum NudeGroup {
	G_EXPOSED = 0, // 胸、生殖器、臀部、肛门暴露
	G_COVERED = 1, // 内衣 / 泳装遮盖的胸、生殖器、臀部、肛门
	G_SKIN = 2,    // 男性赤裸上身、腹部、腋下
	G_FEET = 3,    // 脚
	G_COUNT = 4,
};
const char *kGroupKey[G_COUNT] = {"nude_g_exposed", "nude_g_covered", "nude_g_skin", "nude_g_feet"};
const char *kGroupText[G_COUNT] = {"NudeGroup.Exposed", "NudeGroup.Covered", "NudeGroup.Skin", "NudeGroup.Feet"};
const bool kGroupDefault[G_COUNT] = {true, false, true, false}; // 「遮盖」组普通衣服也会触发，默认关；泳装由裸露皮肤通道覆盖

int nude_group(int cls)
{
	switch (cls) {
	case FEMALE_BREAST_EXPOSED:
	case FEMALE_GENITALIA_EXPOSED:
	case MALE_GENITALIA_EXPOSED:
	case BUTTOCKS_EXPOSED:
	case ANUS_EXPOSED:
		return G_EXPOSED;
	case FEMALE_BREAST_COVERED:
	case FEMALE_GENITALIA_COVERED:
	case BUTTOCKS_COVERED:
	case ANUS_COVERED:
		return G_COVERED;
	case MALE_BREAST_EXPOSED:
	case BELLY_EXPOSED:
	case ARMPITS_EXPOSED:
		return G_SKIN;
	case FEET_EXPOSED:
		return G_FEET;
	default:
		return -1; // 脸、被衣服盖住的腹部 / 腋下 / 脚：不遮
	}
}

const char *kNudeModelFile[2] = {"320n.onnx", "640m.onnx"};

// 归一化坐标（0~1）的遮挡框
struct Track {
	float x, y, w, h;
	uint64_t last_seen_ns;
	uint32_t id;
	int cls; // kClsFace、kClsSkin、kClsText，或 NudeClass
};

// 某一帧的检测结果（含保留中的旧框）
struct Snapshot {
	uint64_t seq;
	std::vector<Track> tracks;
};

float iou(const Track &a, const Track &b)
{
	float x1 = std::max(a.x, b.x), y1 = std::max(a.y, b.y);
	float x2 = std::min(a.x + a.w, b.x + b.w), y2 = std::min(a.y + a.h, b.y + b.h);
	float inter = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
	float uni = a.w * a.h + b.w * b.h - inter;
	return uni > 0.0f ? inter / uni : 0.0f;
}

struct Channel {
	ChannelKind kind = CH_FACE;
	std::atomic<bool> enabled{false};
	std::atomic<int> want_model{-1}; // -1 表示不加载；人脸 0；不雅 0 = 320n，1 = 640m
	int loaded_model = -1;           // 只在推理线程读写
	std::atomic<bool> model_ok{false};
	FaceDetector face;
	NudeDetector nude;
	SkinDetector skin;
	TextDetector text;

	std::thread worker;
	std::mutex job_mutex;
	std::condition_variable job_cv;
	bool stop = false;
	bool has_job = false;
	std::vector<uint8_t> job_pixels;
	int job_w = 0, job_h = 0;
	uint64_t job_seq = 0;
	std::atomic<bool> busy{false};
	uint64_t last_job_ns = 0; // 渲染线程：最近一次交任务的时间

	std::mutex track_mutex;
	std::vector<Track> tracks;
	std::deque<Snapshot> history;
	uint32_t next_id = 1;
	std::atomic<uint64_t> last_result_ns{0};

	// 统计（推理线程）
	uint64_t stat_start_ns = 0;
	int stat_runs = 0;
	double stat_ms = 0;
	size_t stat_boxes = 0;
};

// 一路 GPU → CPU 的画面采集：缩放到 max_size 以内，双缓冲暂存面，下一帧才 map
struct Capture {
	uint32_t max_size = 640;
	gs_texrender_t *tr = nullptr;
	gs_stagesurf_t *stage[2] = {nullptr, nullptr};
	uint32_t stage_w[2] = {0, 0}, stage_h[2] = {0, 0};
	uint64_t stage_seq[2] = {0, 0};
	uint32_t stage_targets[2] = {0, 0}; // 这份暂存要交给哪些通道（位掩码）
	bool stage_pending[2] = {false, false};
	int stage_idx = 0;
	uint64_t last_capture_ns = 0;
};

struct FaceMosaic {
	obs_source_t *context = nullptr;

	// 设置
	int mode = 0; // 0 马赛克，1 纯色块
	int block_size = 20;
	float margin = 0.25f;
	std::atomic<float> face_threshold{0.6f};
	std::atomic<float> nude_threshold{0.2f};
	std::atomic<uint32_t> nude_groups{0};
	int nude_action = 0; // 0 局部打码，1 出现暴露部位时整屏打码
	std::atomic<float> skin_body{0.4f};
	std::atomic<float> skin_torso{0.6f};
	std::mutex text_mutex; // 保护 text_rules（推理线程读、update 写）
	TextRules text_rules;
	std::atomic<int> text_limit{800};
	uint64_t text_interval_ns = 1000000000ULL / 3;
	std::atomic<uint64_t> hold_ns{500ULL * 1000000ULL};
	uint64_t detect_interval_ns = 1000000000ULL / 15;
	int delay_ms = 150;
	bool fail_safe = true;

	std::string face_model_path;
	std::string nude_model_path[2];
	std::string seg_model_path;
	std::string ocr_det_path, ocr_rec_path;

	Channel ch[CH_COUNT];

	// 图形资源（只在图形线程用）
	gs_effect_t *effect = nullptr;
	std::vector<gs_texrender_t *> ring;
	std::vector<uint64_t> ring_seq; // 0 表示空
	uint32_t ring_w = 0, ring_h = 0;
	uint64_t frame_seq = 0; // 最新一帧的帧号，从 1 开始
	bool new_frame = true;
	Capture cap_small; // 人脸、不雅内容、裸露皮肤共用（长边 640 以内）
	Capture cap_text;  // 敏感文字（长边 1920 以内）

	// 调试：环境变量 FACE_MOSAIC_DUMP=<路径前缀> 时，拿到检测结果约 3 秒后开始，
	// 每隔 FACE_MOSAIC_DUMP_EVERY 帧（默认 7）导出一组 <前缀>-<n>.png（输出）和
	// <前缀>-<n>-raw.png（同一帧未打码的画面），共 FACE_MOSAIC_DUMP_COUNT 组（默认 1）。
	std::string dump_prefix;
	int dump_total = 1;
	int dump_every = 7;
	int dump_done = 0;
	gs_texrender_t *dump_tr = nullptr;
	gs_stagesurf_t *dump_stage_out = nullptr;
	gs_stagesurf_t *dump_stage_raw = nullptr;
	bool dump_pending = false;
	uint64_t dump_after_ns = 0;
	uint64_t dump_next_seq = 0;

	// 渲染统计
	uint64_t rstat_start_ns = 0;
	int rstat_frames = 0;
	int rstat_aligned = 0; // 输出帧前后都有检测结果（真正对齐）的帧数
	int rstat_fullscreen = 0;
};

// 新检测结果和同类别的旧框匹配：IoU 足够大，或者中心距离不超过框尺寸（快速移动时 IoU 可能为 0）
int match_track(const Track &t, const std::vector<Track> &old, const std::vector<bool> &used, int w, int h)
{
	int best_i = -1;
	float best_cost = 1e9f;
	for (size_t i = 0; i < old.size(); i++) {
		if (used[i] || old[i].cls != t.cls)
			continue;
		const Track &o = old[i];
		float dx = ((t.x + t.w / 2) - (o.x + o.w / 2)) * w;
		float dy = ((t.y + t.h / 2) - (o.y + o.h / 2)) * h;
		float size = std::max({t.w * w, t.h * h, o.w * w, o.h * h});
		float dist = std::sqrt(dx * dx + dy * dy) / std::max(size, 1.0f);
		float v = iou(t, o);
		if (v < 0.1f && dist > 1.0f)
			continue;
		float cost = dist - v;
		if (cost < best_cost) {
			best_cost = cost;
			best_i = (int)i;
		}
	}
	return best_i;
}

// want 编码：模型编号 * 4 + 推理设备（0 自动 GPU 优先，1 仅 CPU）；-1 表示卸载
int want_code(int model, int accel)
{
	return model < 0 ? -1 : model * 4 + accel;
}

void load_model(FaceMosaic *f, Channel *c, int want)
{
	const int model = want < 0 ? -1 : want / 4;
	const int accel = want < 0 ? 1 : want % 4;
	bool ok = false;
	std::string err;
	uint64_t t0 = os_gettime_ns();
	if (!ort_runtime_ready()) {
		err = "ONNX Runtime 没有初始化成功（见插件加载时的日志）";
	} else if (c->kind == CH_FACE) {
		if (model >= 0)
			ok = c->face.load(f->face_model_path, err, accel);
	} else if (c->kind == CH_NUDE) {
		if (model >= 0 && model < 2)
			ok = c->nude.load(f->nude_model_path[model], 640, err, accel);
		else
			c->nude.load("", 640, err); // 加载空路径会失败，相当于卸载
	} else if (c->kind == CH_SKIN) {
		if (model >= 0)
			ok = c->skin.load(f->seg_model_path, f->face_model_path, err, accel);
		else
			c->skin.load("", "", err);
	} else {
		if (model >= 0)
			ok = c->text.load(f->ocr_det_path, f->ocr_rec_path, err, accel);
		else
			c->text.load("", "", err);
	}
	c->loaded_model = want;
	c->model_ok = ok;
	if (model < 0)
		return;
	const std::string &provider = c->kind == CH_FACE   ? c->face.provider()
				      : c->kind == CH_NUDE ? c->nude.provider()
				      : c->kind == CH_SKIN ? c->skin.provider()
							   : c->text.provider();
	if (ok)
		blog(LOG_INFO, "[face-mosaic] %s模型已加载（%s，%s），用时 %.0f ms", kChannelName[c->kind],
		     c->kind == CH_FACE   ? "YuNet"
		     : c->kind == CH_NUDE ? kNudeModelFile[model]
		     : c->kind == CH_SKIN ? "PP-HumanSeg + YuNet"
					  : "PP-OCRv4",
		     provider.c_str(), (os_gettime_ns() - t0) / 1e6);
	else
		blog(LOG_ERROR, "[face-mosaic] %s模型加载失败：%s", kChannelName[c->kind], err.c_str());
}

// 跑一次检测，返回归一化坐标的框（已按设置过滤）
std::vector<Track> run_detection(FaceMosaic *f, Channel *c, const std::vector<uint8_t> &px, int w, int h,
				 uint64_t now)
{
	std::vector<Track> out;
	if (c->kind == CH_FACE) {
		for (const FaceBox &b : c->face.detect(px.data(), w, h, w * 4, f->face_threshold))
			out.push_back({b.x / w, b.y / h, b.w / w, b.h / h, now, 0, kClsFace});
	} else if (c->kind == CH_SKIN) {
		SkinParams params;
		params.body_threshold = f->skin_body;
		params.torso_threshold = f->skin_torso;
		for (const SkinBox &b : c->skin.detect(px.data(), w, h, w * 4, params))
			out.push_back({b.x / w, b.y / h, b.w / w, b.h / h, now, 0, kClsSkin});
	} else if (c->kind == CH_TEXT) {
		TextRules rules;
		{
			std::lock_guard<std::mutex> lock(f->text_mutex);
			rules = f->text_rules;
		}
		// 注意：识别出的文字可能就是隐私信息，绝不写进日志；日志只记行数和命中规则
		for (const TextLine &l : c->text.detect(px.data(), w, h, w * 4, f->text_limit)) {
			std::string reason;
			if (match_sensitive(l.text, rules, reason))
				out.push_back({l.x / w, l.y / h, l.w / w, l.h / h, now, 0, kClsText});
		}
	} else {
		uint32_t groups = f->nude_groups;
		for (const NudeBox &b : c->nude.detect(px.data(), w, h, w * 4, f->nude_threshold)) {
			int g = nude_group(b.cls);
			if (g < 0 || !(groups & (1u << g)))
				continue;
			out.push_back({b.x / w, b.y / h, b.w / w, b.h / h, now, 0, b.cls});
		}
	}
	return out;
}

void worker_loop(FaceMosaic *f, Channel *c)
{
	std::vector<uint8_t> pixels;
	for (;;) {
		int w, h;
		uint64_t seq;
		{
			std::unique_lock<std::mutex> lock(c->job_mutex);
			c->job_cv.wait(lock, [c] {
				return c->stop || c->has_job || c->want_model.load() != c->loaded_model;
			});
			if (c->stop)
				return;
			int want = c->want_model;
			if (want != c->loaded_model) {
				lock.unlock();
				load_model(f, c, want);
				continue;
			}
			pixels.swap(c->job_pixels);
			w = c->job_w;
			h = c->job_h;
			seq = c->job_seq;
			c->has_job = false;
		}

		uint64_t t0 = os_gettime_ns();
		std::vector<Track> dets = run_detection(f, c, pixels, w, h, t0);
		uint64_t now = os_gettime_ns();
		uint64_t hold_ns = f->hold_ns;
		if (c->kind == CH_TEXT)
			hold_ns = std::max<uint64_t>(hold_ns, 1500ULL * 1000000ULL);

		{
			std::lock_guard<std::mutex> lock(c->track_mutex);
			std::vector<Track> next;
			std::vector<bool> used(c->tracks.size(), false);
			for (Track t : dets) {
				t.last_seen_ns = now;
				int mi = match_track(t, c->tracks, used, w, h);
				if (mi >= 0) {
					used[mi] = true;
					t.id = c->tracks[mi].id;
				} else {
					t.id = c->next_id++;
				}
				next.push_back(t);
			}
			// 没匹配上的旧框在保留时间内继续遮
			for (size_t i = 0; i < c->tracks.size(); i++) {
				if (!used[i] && now - c->tracks[i].last_seen_ns < hold_ns)
					next.push_back(c->tracks[i]);
			}
			c->tracks.swap(next);
			c->history.push_back({seq, c->tracks});
			while (c->history.size() > kMaxHistory)
				c->history.pop_front();
		}
		c->last_result_ns = now;
		c->busy = false;

		c->stat_runs++;
		c->stat_ms += (now - t0) / 1e6;
		c->stat_boxes += dets.size();
		if (c->stat_start_ns == 0)
			c->stat_start_ns = now;
		if (now - c->stat_start_ns > 10ULL * 1000000000ULL) {
			blog(LOG_INFO, "[face-mosaic] %s：最近 10 秒检测 %d 次，平均 %.1f ms，平均每次 %.1f 个框",
			     kChannelName[c->kind], c->stat_runs, c->stat_ms / c->stat_runs,
			     (double)c->stat_boxes / c->stat_runs);
			c->stat_start_ns = now;
			c->stat_runs = 0;
			c->stat_ms = 0;
			c->stat_boxes = 0;
		}
	}
}

double canvas_fps()
{
	struct obs_video_info ovi;
	if (obs_get_video_info(&ovi) && ovi.fps_den > 0)
		return (double)ovi.fps_num / ovi.fps_den;
	return 30.0;
}

int delay_frames(const FaceMosaic *f)
{
	if (f->delay_ms <= 0)
		return 0;
	int n = (int)std::ceil(f->delay_ms * canvas_fps() / 1000.0 - 1e-6);
	return std::clamp(n, 0, kMaxDelayFrames);
}

const char *fm_get_name(void *)
{
	return obs_module_text("FaceMosaic");
}

void set_want_model(Channel *c, int model)
{
	{
		std::lock_guard<std::mutex> lock(c->job_mutex);
		c->want_model = model;
	}
	c->job_cv.notify_all();
}

void fm_update(void *data, obs_data_t *settings)
{
	auto *f = static_cast<FaceMosaic *>(data);
	f->mode = (int)obs_data_get_int(settings, "mode");
	f->block_size = std::max(2, (int)obs_data_get_int(settings, "block_size"));
	f->margin = (float)obs_data_get_int(settings, "margin") / 100.0f;
	f->hold_ns = (uint64_t)obs_data_get_int(settings, "hold_ms") * 1000000ULL;
	int fps = std::max(1, (int)obs_data_get_int(settings, "detect_fps"));
	f->detect_interval_ns = 1000000000ULL / fps;
	f->delay_ms = std::max(0, (int)obs_data_get_int(settings, "delay_ms"));
	f->fail_safe = obs_data_get_bool(settings, "fail_safe");
	const int accel = std::clamp((int)obs_data_get_int(settings, "accel"), 0, 1);

	f->face_threshold = (float)obs_data_get_double(settings, "threshold");
	bool face_on = obs_data_get_bool(settings, "face_enabled");
	f->ch[CH_FACE].enabled = face_on;
	set_want_model(&f->ch[CH_FACE], want_code(face_on ? 0 : -1, accel));

	f->nude_threshold = (float)obs_data_get_double(settings, "nude_threshold");
	uint32_t groups = 0;
	for (int g = 0; g < G_COUNT; g++)
		if (obs_data_get_bool(settings, kGroupKey[g]))
			groups |= 1u << g;
	f->nude_groups = groups;
	f->nude_action = (int)obs_data_get_int(settings, "nude_action");
	bool nude_on = obs_data_get_bool(settings, "nude_enabled");
	int nude_model = std::clamp((int)obs_data_get_int(settings, "nude_model"), 0, 1);
	f->ch[CH_NUDE].enabled = nude_on;
	set_want_model(&f->ch[CH_NUDE], want_code(nude_on ? nude_model : -1, accel));

	f->skin_body = (float)obs_data_get_int(settings, "skin_body") / 100.0f;
	f->skin_torso = (float)obs_data_get_int(settings, "skin_torso") / 100.0f;
	bool skin_on = obs_data_get_bool(settings, "skin_enabled");
	f->ch[CH_SKIN].enabled = skin_on;
	set_want_model(&f->ch[CH_SKIN], want_code(skin_on ? 0 : -1, accel));

	TextRules rules;
	rules.phone = obs_data_get_bool(settings, "text_phone");
	rules.id_card = obs_data_get_bool(settings, "text_id_card");
	rules.bank_card = obs_data_get_bool(settings, "text_bank_card");
	rules.email = obs_data_get_bool(settings, "text_email");
	rules.url = obs_data_get_bool(settings, "text_url");
	rules.im = obs_data_get_bool(settings, "text_im");
	rules.keywords = parse_keywords(obs_data_get_string(settings, "text_keywords"));
	{
		std::lock_guard<std::mutex> lock(f->text_mutex);
		f->text_rules = std::move(rules);
	}
	f->text_limit = kTextLimits[std::clamp((int)obs_data_get_int(settings, "text_limit"), 0, 3)];
	int text_fps = std::max(1, (int)obs_data_get_int(settings, "text_fps"));
	f->text_interval_ns = 1000000000ULL / text_fps;
	bool text_on = obs_data_get_bool(settings, "text_enabled");
	f->ch[CH_TEXT].enabled = text_on;
	set_want_model(&f->ch[CH_TEXT], want_code(text_on ? 0 : -1, accel));
}

void fm_defaults(obs_data_t *settings)
{
	obs_data_set_default_int(settings, "mode", 0);
	obs_data_set_default_int(settings, "block_size", 20);
	obs_data_set_default_int(settings, "margin", 25);
	obs_data_set_default_int(settings, "hold_ms", 500);
	obs_data_set_default_int(settings, "detect_fps", 15);
	obs_data_set_default_int(settings, "delay_ms", 150);
	obs_data_set_default_bool(settings, "fail_safe", true);
	obs_data_set_default_int(settings, "accel", 0);

	obs_data_set_default_bool(settings, "face_enabled", true);
	obs_data_set_default_double(settings, "threshold", 0.6);

	obs_data_set_default_bool(settings, "nude_enabled", true);
	obs_data_set_default_int(settings, "nude_model", 0);
	obs_data_set_default_double(settings, "nude_threshold", 0.2);
	obs_data_set_default_int(settings, "nude_action", 0);
	for (int g = 0; g < G_COUNT; g++)
		obs_data_set_default_bool(settings, kGroupKey[g], kGroupDefault[g]);

	obs_data_set_default_bool(settings, "skin_enabled", true);
	obs_data_set_default_int(settings, "skin_body", 40);
	obs_data_set_default_int(settings, "skin_torso", 60);

	obs_data_set_default_bool(settings, "text_enabled", true);
	for (const char *k : {"text_phone", "text_id_card", "text_bank_card", "text_email", "text_url", "text_im"})
		obs_data_set_default_bool(settings, k, true);
	obs_data_set_default_string(settings, "text_keywords", "");
	obs_data_set_default_int(settings, "text_limit", 1); // 800
	obs_data_set_default_int(settings, "text_fps", 3);
}

obs_properties_t *fm_properties(void *data)
{
	obs_properties_t *props = obs_properties_create();

	// 不雅内容
	obs_properties_t *nude = obs_properties_create();
	obs_property_t *model = obs_properties_add_list(nude, "nude_model", obs_module_text("NudeModel"),
							OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
	obs_property_list_add_int(model, obs_module_text("NudeModel.Fast"), 0);
	obs_property_list_add_int(model, obs_module_text("NudeModel.Accurate"), 1);
	obs_properties_add_float_slider(nude, "nude_threshold", obs_module_text("NudeThreshold"), 0.1, 0.9, 0.05);
	for (int g = 0; g < G_COUNT; g++)
		obs_properties_add_bool(nude, kGroupKey[g], obs_module_text(kGroupText[g]));
	obs_property_t *action = obs_properties_add_list(nude, "nude_action", obs_module_text("NudeAction"),
							 OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
	obs_property_list_add_int(action, obs_module_text("NudeAction.Region"), 0);
	obs_property_list_add_int(action, obs_module_text("NudeAction.Full"), 1);
	obs_properties_add_group(props, "nude_enabled", obs_module_text("NudeEnabled"), OBS_GROUP_CHECKABLE, nude);

	// 裸露皮肤
	obs_properties_t *skin = obs_properties_create();
	obs_properties_add_int_slider(skin, "skin_body", obs_module_text("SkinBody"), 10, 90, 5);
	obs_properties_add_int_slider(skin, "skin_torso", obs_module_text("SkinTorso"), 10, 100, 5);
	obs_properties_add_text(skin, "skin_info", obs_module_text("SkinInfo"), OBS_TEXT_INFO);
	obs_properties_add_group(props, "skin_enabled", obs_module_text("SkinEnabled"), OBS_GROUP_CHECKABLE, skin);

	// 敏感文字
	obs_properties_t *text = obs_properties_create();
	obs_properties_add_bool(text, "text_phone", obs_module_text("TextPhone"));
	obs_properties_add_bool(text, "text_id_card", obs_module_text("TextIdCard"));
	obs_properties_add_bool(text, "text_bank_card", obs_module_text("TextBankCard"));
	obs_properties_add_bool(text, "text_email", obs_module_text("TextEmail"));
	obs_properties_add_bool(text, "text_url", obs_module_text("TextUrl"));
	obs_properties_add_bool(text, "text_im", obs_module_text("TextIm"));
	obs_properties_add_text(text, "text_keywords", obs_module_text("TextKeywords"), OBS_TEXT_MULTILINE);
	obs_property_t *limit = obs_properties_add_list(text, "text_limit", obs_module_text("TextLimit"),
							OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
	obs_property_list_add_int(limit, "640", 0);
	obs_property_list_add_int(limit, "800", 1);
	obs_property_list_add_int(limit, "960", 2);
	obs_property_list_add_int(limit, "1280", 3);
	obs_properties_add_int_slider(text, "text_fps", obs_module_text("TextFps"), 1, 10, 1);
	obs_properties_add_text(text, "text_info", obs_module_text("TextInfo"), OBS_TEXT_INFO);
	obs_properties_add_group(props, "text_enabled", obs_module_text("TextEnabled"), OBS_GROUP_CHECKABLE, text);

	// 人脸
	obs_properties_t *face = obs_properties_create();
	obs_properties_add_float_slider(face, "threshold", obs_module_text("Threshold"), 0.3, 0.95, 0.05);
	obs_properties_add_group(props, "face_enabled", obs_module_text("FaceEnabled"), OBS_GROUP_CHECKABLE, face);

	// 通用
	obs_property_t *mode = obs_properties_add_list(props, "mode", obs_module_text("Mode"), OBS_COMBO_TYPE_LIST,
						       OBS_COMBO_FORMAT_INT);
	obs_property_list_add_int(mode, obs_module_text("Mode.Pixelate"), 0);
	obs_property_list_add_int(mode, obs_module_text("Mode.Solid"), 1);
	obs_properties_add_int_slider(props, "block_size", obs_module_text("BlockSize"), 4, 80, 1);
	obs_properties_add_int_slider(props, "margin", obs_module_text("Margin"), 0, 100, 5);
	obs_properties_add_int_slider(props, "hold_ms", obs_module_text("HoldMs"), 0, 3000, 100);
	obs_properties_add_int_slider(props, "detect_fps", obs_module_text("DetectFps"), 1, 30, 1);
	obs_properties_add_int_slider(props, "delay_ms", obs_module_text("DelayMs"), 0, 500, 10);
	obs_properties_add_text(props, "delay_info", obs_module_text("DelayInfo"), OBS_TEXT_INFO);
	obs_properties_add_bool(props, "fail_safe", obs_module_text("FailSafe"));
	obs_property_t *accel = obs_properties_add_list(props, "accel", obs_module_text("Accel"), OBS_COMBO_TYPE_LIST,
							OBS_COMBO_FORMAT_INT);
	std::string gpu = obs_module_text("Accel.Auto");
	if (*ort_gpu_provider())
		gpu += std::string("（") + ort_gpu_provider() + "）";
	obs_property_list_add_int(accel, gpu.c_str(), 0);
	obs_property_list_add_int(accel, obs_module_text("Accel.Cpu"), 1);

	UNUSED_PARAMETER(data);
	return props;
}

void *fm_create(obs_data_t *settings, obs_source_t *source)
{
	auto *f = new FaceMosaic();
	f->context = source;
	f->ch[CH_FACE].kind = CH_FACE;
	f->ch[CH_NUDE].kind = CH_NUDE;
	f->ch[CH_SKIN].kind = CH_SKIN;
	f->ch[CH_TEXT].kind = CH_TEXT;
	f->cap_small.max_size = kMaxDetectSize;
	f->cap_text.max_size = kTextCaptureMax;

	auto module_file = [](const char *name) {
		char *p = obs_module_file(name);
		std::string s = p ? p : "";
		bfree(p);
		if (s.empty())
			blog(LOG_ERROR, "[face-mosaic] 找不到模型文件：%s", name);
		return s;
	};
	f->face_model_path = module_file("face_detection_yunet_2023mar.onnx");
	for (int i = 0; i < 2; i++)
		f->nude_model_path[i] = module_file(kNudeModelFile[i]);
	f->seg_model_path = module_file("pphumanseg_2023mar.onnx");
	f->ocr_det_path = module_file("ch_PP-OCRv4_det_infer.onnx");
	f->ocr_rec_path = module_file("ch_PP-OCRv4_rec_infer.onnx");

	obs_enter_graphics();
	char *effect_err = nullptr;
	f->effect = gs_effect_create(kEffectSource, "face-mosaic.effect", &effect_err);
	if (!f->effect)
		blog(LOG_ERROR, "[face-mosaic] 着色器编译失败：%s", effect_err ? effect_err : "(无信息)");
	bfree(effect_err);
	f->cap_small.tr = gs_texrender_create(GS_RGBA, GS_ZS_NONE);
	f->cap_text.tr = gs_texrender_create(GS_RGBA, GS_ZS_NONE);
	obs_leave_graphics();

	if (const char *dump = getenv("FACE_MOSAIC_DUMP"))
		f->dump_prefix = dump;
	if (const char *n = getenv("FACE_MOSAIC_DUMP_COUNT"))
		f->dump_total = std::max(1, atoi(n));
	if (const char *n = getenv("FACE_MOSAIC_DUMP_EVERY"))
		f->dump_every = std::max(1, atoi(n));

	f->rstat_start_ns = os_gettime_ns();
	for (Channel &c : f->ch)
		c.worker = std::thread(worker_loop, f, &c);
	fm_update(f, settings);
	return f;
}

void destroy_ring(FaceMosaic *f)
{
	for (gs_texrender_t *tr : f->ring)
		gs_texrender_destroy(tr);
	f->ring.clear();
	f->ring_seq.clear();
}

void fm_destroy(void *data)
{
	auto *f = static_cast<FaceMosaic *>(data);
	for (Channel &c : f->ch) {
		{
			std::lock_guard<std::mutex> lock(c.job_mutex);
			c.stop = true;
		}
		c.job_cv.notify_all();
	}
	for (Channel &c : f->ch)
		if (c.worker.joinable())
			c.worker.join();

	obs_enter_graphics();
	gs_effect_destroy(f->effect);
	destroy_ring(f);
	for (Capture *cap : {&f->cap_small, &f->cap_text}) {
		gs_texrender_destroy(cap->tr);
		gs_stagesurface_destroy(cap->stage[0]);
		gs_stagesurface_destroy(cap->stage[1]);
	}
	gs_texrender_destroy(f->dump_tr);
	gs_stagesurface_destroy(f->dump_stage_out);
	gs_stagesurface_destroy(f->dump_stage_raw);
	obs_leave_graphics();
	delete f;
}

void fm_tick(void *data, float)
{
	static_cast<FaceMosaic *>(data)->new_frame = true;
}

// 把滤镜的上游画面渲染进 texrender
bool render_target(FaceMosaic *f, gs_texrender_t *tr, uint32_t w, uint32_t h)
{
	obs_source_t *target = obs_filter_get_target(f->context);
	obs_source_t *parent = obs_filter_get_parent(f->context);
	if (!target || !parent)
		return false;

	gs_texrender_reset(tr);
	if (!gs_texrender_begin(tr, w, h))
		return false;

	struct vec4 clear;
	vec4_zero(&clear);
	gs_clear(GS_CLEAR_COLOR, &clear, 0.0f, 0);
	gs_ortho(0.0f, (float)w, 0.0f, (float)h, -100.0f, 100.0f);
	gs_blend_state_push();
	gs_blend_function(GS_BLEND_ONE, GS_BLEND_ZERO);

	uint32_t flags = obs_source_get_output_flags(target);
	bool custom_draw = (flags & OBS_SOURCE_CUSTOM_DRAW) != 0;
	bool async = (flags & OBS_SOURCE_ASYNC) != 0;
	if (target == parent && !custom_draw && !async)
		obs_source_default_render(target);
	else
		obs_source_video_render(target);

	gs_blend_state_pop();
	gs_texrender_end(tr);
	return true;
}

// 读回上一帧暂存的画面，交给对应的通道
void collect_stage(FaceMosaic *f, Capture &cap)
{
	int idx = cap.stage_idx ^ 1;
	if (!cap.stage_pending[idx])
		return;
	cap.stage_pending[idx] = false;

	uint8_t *data = nullptr;
	uint32_t linesize = 0;
	if (!gs_stagesurface_map(cap.stage[idx], &data, &linesize)) {
		for (int k = 0; k < CH_COUNT; k++)
			if (cap.stage_targets[idx] & (1u << k))
				f->ch[k].busy = false;
		return;
	}

	int w = (int)cap.stage_w[idx], h = (int)cap.stage_h[idx];
	for (int k = 0; k < CH_COUNT; k++) {
		if (!(cap.stage_targets[idx] & (1u << k)))
			continue;
		Channel &c = f->ch[k];
		{
			std::lock_guard<std::mutex> lock(c.job_mutex);
			c.job_pixels.resize((size_t)w * h * 4);
			for (int y = 0; y < h; y++)
				memcpy(c.job_pixels.data() + (size_t)y * w * 4, data + (size_t)y * linesize,
				       (size_t)w * 4);
			c.job_w = w;
			c.job_h = h;
			c.job_seq = cap.stage_seq[idx];
			c.has_job = true;
		}
		c.job_cv.notify_one();
	}
	gs_stagesurface_unmap(cap.stage[idx]);
}

// 把画面缩小后放进暂存面（下一帧再 map）
void capture_for_detection(Capture &cap, gs_texture_t *tex, uint32_t w, uint32_t h, uint64_t seq,
			   uint32_t targets)
{
	float scale = std::min(1.0f, std::min((float)cap.max_size / w, (float)cap.max_size / h));
	uint32_t dw = std::max(1u, (uint32_t)(w * scale + 0.5f));
	uint32_t dh = std::max(1u, (uint32_t)(h * scale + 0.5f));
	dw = std::min<uint32_t>(dw, cap.max_size);
	dh = std::min<uint32_t>(dh, cap.max_size);

	gs_texrender_reset(cap.tr);
	if (!gs_texrender_begin(cap.tr, dw, dh))
		return;
	gs_ortho(0.0f, (float)w, 0.0f, (float)h, -100.0f, 100.0f);
	gs_blend_state_push();
	gs_blend_function(GS_BLEND_ONE, GS_BLEND_ZERO);
	gs_effect_t *def = obs_get_base_effect(OBS_EFFECT_DEFAULT);
	gs_effect_set_texture(gs_effect_get_param_by_name(def, "image"), tex);
	while (gs_effect_loop(def, "Draw"))
		gs_draw_sprite(tex, 0, w, h);
	gs_blend_state_pop();
	gs_texrender_end(cap.tr);

	int idx = cap.stage_idx;
	if (!cap.stage[idx] || cap.stage_w[idx] != dw || cap.stage_h[idx] != dh) {
		gs_stagesurface_destroy(cap.stage[idx]);
		cap.stage[idx] = gs_stagesurface_create(dw, dh, GS_RGBA);
		cap.stage_w[idx] = dw;
		cap.stage_h[idx] = dh;
	}
	gs_stage_texture(cap.stage[idx], gs_texrender_get_texture(cap.tr));
	cap.stage_seq[idx] = seq;
	cap.stage_targets[idx] = targets;
	cap.stage_pending[idx] = true;
	cap.stage_idx ^= 1;
}

// 取输出帧 s 要遮的框：A = 帧号 <= s 的最近结果，B = 帧号 > s 的最近结果
std::vector<Track> boxes_for_frame(Channel *c, uint64_t s, bool &aligned)
{
	std::lock_guard<std::mutex> lock(c->track_mutex);
	const Snapshot *a = nullptr, *b = nullptr;
	for (const Snapshot &snap : c->history) {
		if (snap.seq <= s) {
			if (!a || snap.seq >= a->seq)
				a = &snap;
		} else if (!b || snap.seq < b->seq) {
			b = &snap;
		}
	}
	aligned = a && b;
	if (!a && !b)
		return {};
	if (!a)
		return b->tracks;

	// A 之前、B 之后各一次结果，用来估计速度
	const Snapshot *a_prev = nullptr, *b_next = nullptr;
	for (const Snapshot &snap : c->history) {
		if (snap.seq < a->seq && (!a_prev || snap.seq > a_prev->seq))
			a_prev = &snap;
		if (b && snap.seq > b->seq && (!b_next || snap.seq < b_next->seq))
			b_next = &snap;
	}
	auto find_id = [](const Snapshot *snap, uint32_t id) -> const Track * {
		if (!snap)
			return nullptr;
		for (const Track &t : snap->tracks)
			if (t.id == id)
				return &t;
		return nullptr;
	};
	// 从 (p0@s0, p1@s1) 线性外推到帧 s
	auto extrapolate = [](const Track &p0, uint64_t s0, const Track &p1, uint64_t s1, uint64_t s) {
		Track r = p1;
		float k = ((float)s - (float)s1) / (float)((int64_t)s1 - (int64_t)s0);
		r.x = p1.x + (p1.x - p0.x) * k;
		r.y = p1.y + (p1.y - p0.y) * k;
		return r;
	};
	auto unite = [](Track &u, const Track &o) {
		float x2 = std::max(u.x + u.w, o.x + o.w), y2 = std::max(u.y + u.h, o.y + o.h);
		u.x = std::min(u.x, o.x);
		u.y = std::min(u.y, o.y);
		u.w = x2 - u.x;
		u.h = y2 - u.y;
	};

	std::vector<Track> out;
	if (!b) {
		// 检测没跟上（延迟不够或负载高）：按 A 之前的速度外推到 s，并上 A 本身
		for (const Track &ta : a->tracks) {
			Track m = ta;
			if (const Track *p = find_id(a_prev, ta.id))
				unite(m, extrapolate(*p, a_prev->seq, ta, a->seq, s));
			out.push_back(m);
		}
		return out;
	}

	float t = (float)(s - a->seq) / (float)(b->seq - a->seq);
	std::vector<bool> b_used(b->tracks.size(), false);
	for (const Track &ta : a->tracks) {
		bool matched = false;
		for (size_t j = 0; j < b->tracks.size(); j++) {
			const Track &tb = b->tracks[j];
			if (tb.id != ta.id)
				continue;
			b_used[j] = true;
			matched = true;
			// 插值
			Track m = ta;
			m.x = ta.x + (tb.x - ta.x) * t;
			m.y = ta.y + (tb.y - ta.y) * t;
			m.w = std::max(ta.w + (tb.w - ta.w) * t, std::min(ta.w, tb.w));
			m.h = std::max(ta.h + (tb.h - ta.h) * t, std::min(ta.h, tb.h));
			// 变速、掉头时插值会偏：再并上按 A 之前的速度往后推、按 B 之后的速度往前推的位置
			if (const Track *p = find_id(a_prev, ta.id))
				unite(m, extrapolate(*p, a_prev->seq, ta, a->seq, s));
			if (const Track *n = find_id(b_next, tb.id))
				unite(m, extrapolate(*n, b_next->seq, tb, b->seq, s));
			out.push_back(m);
			break;
		}
		if (!matched)
			out.push_back(ta); // B 里已经没有（保留时间过了），这一帧仍照 A 遮
	}
	for (size_t j = 0; j < b->tracks.size(); j++) {
		if (!b_used[j])
			out.push_back(b->tracks[j]); // 新出现的：用之后的结果提前遮
	}
	return out;
}

void draw_region(FaceMosaic *f, gs_texture_t *tex, uint32_t w, uint32_t h, int x0, int y0, int x1, int y1,
		 float block)
{
	x0 = std::clamp(x0, 0, (int)w);
	x1 = std::clamp(x1, 0, (int)w);
	y0 = std::clamp(y0, 0, (int)h);
	y1 = std::clamp(y1, 0, (int)h);
	if (x1 <= x0 || y1 <= y0)
		return;

	struct vec2 size;
	vec2_set(&size, (float)w, (float)h);
	gs_effect_set_texture(gs_effect_get_param_by_name(f->effect, "image"), tex);
	gs_effect_set_vec2(gs_effect_get_param_by_name(f->effect, "tex_size"), &size);
	gs_effect_set_float(gs_effect_get_param_by_name(f->effect, "block"), block);
	struct vec4 color;
	vec4_set(&color, 0.0f, 0.0f, 0.0f, 1.0f);
	gs_effect_set_vec4(gs_effect_get_param_by_name(f->effect, "color"), &color);

	// 黑色遮挡：直接覆盖（不与底下画面混合），保证 100% 不透明纯黑
	const bool solid = f->mode == 1;
	if (solid) {
		gs_blend_state_push();
		gs_blend_function(GS_BLEND_ONE, GS_BLEND_ZERO);
	}
	gs_matrix_push();
	gs_matrix_translate3f((float)x0, (float)y0, 0.0f);
	while (gs_effect_loop(f->effect, solid ? "Solid" : "Pixelate"))
		gs_draw_sprite_subregion(tex, 0, x0, y0, x1 - x0, y1 - y0);
	gs_matrix_pop();
	if (solid)
		gs_blend_state_pop();
}

void draw_plain(gs_texture_t *tex, uint32_t w, uint32_t h)
{
	gs_effect_t *def = obs_get_base_effect(OBS_EFFECT_DEFAULT);
	gs_effect_set_texture(gs_effect_get_param_by_name(def, "image"), tex);
	while (gs_effect_loop(def, "Draw"))
		gs_draw_sprite(tex, 0, w, h);
}

// 画最终输出：原画面 + 遮挡。full_screen 为真时整屏打码（兜底或「整屏打码」模式）
void draw_output(FaceMosaic *f, gs_texture_t *tex, uint32_t w, uint32_t h, const std::vector<Track> &boxes,
		 bool full_screen)
{
	draw_plain(tex, w, h);
	if (full_screen) {
		draw_region(f, tex, w, h, 0, 0, (int)w, (int)h, (float)std::max(w, h) / 24.0f);
		return;
	}
	for (const Track &t : boxes) {
		float bx = t.x * w, by = t.y * h, bw = t.w * w, bh = t.h * h;
		float mx = bw * f->margin, my = bh * f->margin;
		// 格子至少为框短边的 1/6，大框不会因为格子太细而看得出轮廓
		float block = std::max((float)f->block_size, std::min(bw, bh) / 6.0f);
		draw_region(f, tex, w, h, (int)(bx - mx), (int)(by - my), (int)(bx + bw + mx + 0.5f),
			    (int)(by + bh + my + 0.5f), block);
	}
}

// 调试导出的 PNG 由一个后台线程排队写（低压缩），避免和检测抢 CPU
void dump_write_async(std::string path, std::vector<uint8_t> pixels, uint32_t w, uint32_t h)
{
	struct Job {
		std::string path;
		std::vector<uint8_t> pixels;
		uint32_t w, h;
	};
	static std::mutex mtx;
	static std::condition_variable cv;
	static std::deque<Job> queue;
	static bool started = false;

	std::lock_guard<std::mutex> lock(mtx);
	queue.push_back({std::move(path), std::move(pixels), w, h});
	cv.notify_one();
	if (started)
		return;
	started = true;
	std::thread([] {
		stbi_write_png_compression_level = 1;
		for (;;) {
			Job job;
			{
				std::unique_lock<std::mutex> lk(mtx);
				cv.wait(lk, [] { return !queue.empty(); });
				job = std::move(queue.front());
				queue.pop_front();
			}
			bool ok = stbi_write_png(job.path.c_str(), (int)job.w, (int)job.h, 4, job.pixels.data(),
						 (int)job.w * 4);
			blog(ok ? LOG_INFO : LOG_WARNING, "[face-mosaic] 调试帧%s：%s", ok ? "已导出" : "导出失败",
			     job.path.c_str());
		}
	}).detach();
}

// 调试导出：输出画面和同一帧的原画面各读回一份
void dump_frames(FaceMosaic *f, gs_texture_t *tex, uint32_t w, uint32_t h, const std::vector<Track> &boxes,
		 bool full_screen, bool stalled, uint64_t out_seq, uint64_t now)
{
	if (f->dump_prefix.empty() || f->dump_done >= f->dump_total)
		return;

	if (f->dump_pending) {
		f->dump_pending = false;
		gs_stagesurf_t *surfs[2] = {f->dump_stage_out, f->dump_stage_raw};
		const char *suffix[2] = {"", "-raw"};
		for (int k = 0; k < 2; k++) {
			uint8_t *data = nullptr;
			uint32_t linesize = 0;
			if (!gs_stagesurface_map(surfs[k], &data, &linesize))
				continue;
			uint32_t sw = gs_stagesurface_get_width(surfs[k]);
			uint32_t sh = gs_stagesurface_get_height(surfs[k]);
			std::vector<uint8_t> copy((size_t)sw * sh * 4);
			for (uint32_t y = 0; y < sh; y++)
				memcpy(copy.data() + (size_t)y * sw * 4, data + (size_t)y * linesize, (size_t)sw * 4);
			gs_stagesurface_unmap(surfs[k]);
			std::string path = f->dump_prefix + "-" + std::to_string(f->dump_done) + suffix[k] + ".png";
			dump_write_async(std::move(path), std::move(copy), sw, sh);
		}
		f->dump_done++;
		f->dump_next_seq = f->frame_seq + (uint64_t)f->dump_every;
		return;
	}

	if (stalled)
		return;
	if (f->dump_after_ns == 0)
		f->dump_after_ns = now + 3ULL * 1000000000ULL;
	if (now < f->dump_after_ns || f->frame_seq < f->dump_next_seq)
		return;

	if (!f->dump_tr)
		f->dump_tr = gs_texrender_create(GS_RGBA, GS_ZS_NONE);
	gs_texrender_reset(f->dump_tr);
	if (!gs_texrender_begin(f->dump_tr, w, h))
		return;
	struct vec4 clear;
	vec4_zero(&clear);
	gs_clear(GS_CLEAR_COLOR, &clear, 0.0f, 0);
	gs_ortho(0.0f, (float)w, 0.0f, (float)h, -100.0f, 100.0f);
	gs_blend_state_push();
	gs_blend_function(GS_BLEND_ONE, GS_BLEND_ZERO);
	draw_output(f, tex, w, h, boxes, full_screen);
	gs_blend_state_pop();
	gs_texrender_end(f->dump_tr);

	for (gs_stagesurf_t **s : {&f->dump_stage_out, &f->dump_stage_raw}) {
		if (!*s || gs_stagesurface_get_width(*s) != w || gs_stagesurface_get_height(*s) != h) {
			gs_stagesurface_destroy(*s);
			*s = gs_stagesurface_create(w, h, GS_RGBA);
		}
	}
	gs_stage_texture(f->dump_stage_out, gs_texrender_get_texture(f->dump_tr));
	gs_stage_texture(f->dump_stage_raw, tex);
	f->dump_pending = true;

	std::string desc;
	for (const Track &t : boxes) {
		desc += t.cls == kClsFace   ? "FACE"
			: t.cls == kClsSkin ? "SKIN"
			: t.cls == kClsText ? "TEXT"
					    : nude_class_name(t.cls);
		desc += " ";
	}
	blog(LOG_INFO, "[face-mosaic] 调试：暂存第 %d 组，输出帧 %llu（最新帧 %llu），%s框 %zu 个：%s", f->dump_done,
	     (unsigned long long)out_seq, (unsigned long long)f->frame_seq, full_screen ? "整屏打码，" : "",
	     boxes.size(), desc.c_str());
}

void fm_video_render(void *data, gs_effect_t *)
{
	auto *f = static_cast<FaceMosaic *>(data);
	obs_source_t *target = obs_filter_get_target(f->context);
	uint32_t w = target ? obs_source_get_base_width(target) : 0;
	uint32_t h = target ? obs_source_get_base_height(target) : 0;
	if (!w || !h || !f->effect) {
		obs_source_skip_video_filter(f->context);
		return;
	}

	// 延迟帧数或分辨率变了：重建环形缓冲，旧帧作废
	size_t ring_size = (size_t)delay_frames(f) + 1;
	if (f->ring.size() != ring_size || f->ring_w != w || f->ring_h != h) {
		destroy_ring(f);
		for (size_t i = 0; i < ring_size; i++)
			f->ring.push_back(gs_texrender_create(GS_RGBA, GS_ZS_NONE));
		f->ring_seq.assign(ring_size, 0);
		f->ring_w = w;
		f->ring_h = h;
		f->new_frame = true;
	}

	uint64_t now = os_gettime_ns();
	bool fresh = f->new_frame; // 本次渲染是不是这一帧第一次渲染（统计只算一次）
	if (f->new_frame) {
		f->new_frame = false;
		uint64_t seq = f->frame_seq + 1;
		size_t idx = seq % ring_size;
		if (!render_target(f, f->ring[idx], w, h)) {
			obs_source_skip_video_filter(f->context);
			return;
		}
		f->frame_seq = seq;
		f->ring_seq[idx] = seq;

		collect_stage(f, f->cap_small);
		collect_stage(f, f->cap_text);
		gs_texture_t *latest = gs_texrender_get_texture(f->ring[idx]);
		if (now - f->cap_small.last_capture_ns >= f->detect_interval_ns) {
			uint32_t targets = 0;
			for (int k = 0; k < CH_COUNT; k++) {
				Channel &c = f->ch[k];
				if (k != CH_TEXT && c.enabled && c.model_ok && !c.busy) {
					c.busy = true;
					c.last_job_ns = now;
					targets |= 1u << k;
				}
			}
			if (targets) {
				f->cap_small.last_capture_ns = now;
				capture_for_detection(f->cap_small, latest, w, h, seq, targets);
			}
		}
		Channel &tc = f->ch[CH_TEXT];
		if (tc.enabled && tc.model_ok && !tc.busy && now - f->cap_text.last_capture_ns >= f->text_interval_ns) {
			tc.busy = true;
			tc.last_job_ns = now;
			f->cap_text.last_capture_ns = now;
			capture_for_detection(f->cap_text, latest, w, h, seq, 1u << CH_TEXT);
		}
		f->rstat_frames++;
	}

	// 输出帧：尽量取 delay 帧之前的，缓冲还没填满时取能拿到的最旧一帧
	uint64_t out_seq = 0;
	gs_texture_t *tex = nullptr;
	for (int k = (int)ring_size - 1; k >= 0; k--) {
		if (f->frame_seq < (uint64_t)k)
			continue;
		uint64_t s = f->frame_seq - (uint64_t)k;
		size_t idx = s % ring_size;
		if (f->ring_seq[idx] == s) {
			out_seq = s;
			tex = gs_texrender_get_texture(f->ring[idx]);
			break;
		}
	}
	if (!tex) {
		obs_source_skip_video_filter(f->context);
		return;
	}

	// 兜底判断 + 收集各通道的框
	bool stalled = false, aligned_all = true, exposed = false;
	std::vector<Track> boxes;
	for (Channel &c : f->ch) {
		if (!c.enabled)
			continue;
		uint64_t last = c.last_result_ns;
		uint64_t stall_ns = (c.kind == CH_TEXT ? kTextStallMs : kStallMs) * 1000000ULL;
		bool st = !c.model_ok || last == 0 || now - last > stall_ns;
		// 推理线程可能还在处理被丢弃的旧任务，超时后允许重新提交
		if (st && c.busy && now - c.last_job_ns > stall_ns)
			c.busy = false;
		stalled = stalled || st;

		bool aligned = false;
		std::vector<Track> part = boxes_for_frame(&c, out_seq, aligned);
		if (c.kind != CH_TEXT) // 文字通道每秒只检测几次，不计入对齐统计
			aligned_all = aligned_all && aligned;
		for (const Track &t : part)
			if (t.cls >= 0 && nude_group(t.cls) == G_EXPOSED)
				exposed = true;
		boxes.insert(boxes.end(), part.begin(), part.end());
	}

	bool full_screen = (stalled && f->fail_safe) || (exposed && f->nude_action == 1);
	draw_output(f, tex, w, h, boxes, full_screen);
	dump_frames(f, tex, w, h, boxes, full_screen, stalled, out_seq, now);

	if (fresh && aligned_all)
		f->rstat_aligned++;
	if (fresh && full_screen)
		f->rstat_fullscreen++;
	if (now - f->rstat_start_ns > 10ULL * 1000000000ULL) {
		if (f->rstat_frames > 0)
			blog(LOG_INFO,
			     "[face-mosaic] 最近 10 秒：输出 %d 帧，延迟 %d 帧（%d ms），前后都有检测结果的帧 %.0f%%，整屏打码 %d 帧",
			     f->rstat_frames, (int)ring_size - 1, f->delay_ms,
			     100.0 * f->rstat_aligned / f->rstat_frames, f->rstat_fullscreen);
		f->rstat_start_ns = now;
		f->rstat_frames = 0;
		f->rstat_aligned = 0;
		f->rstat_fullscreen = 0;
	}
}

} // namespace

struct obs_source_info face_mosaic_filter_info = [] {
	struct obs_source_info info = {};
	info.id = "face_mosaic_filter";
	info.type = OBS_SOURCE_TYPE_FILTER;
	info.output_flags = OBS_SOURCE_VIDEO;
	info.get_name = fm_get_name;
	info.create = fm_create;
	info.destroy = fm_destroy;
	info.update = fm_update;
	info.get_defaults = fm_defaults;
	info.get_properties = fm_properties;
	info.video_tick = fm_tick;
	info.video_render = fm_video_render;
	return info;
}();
