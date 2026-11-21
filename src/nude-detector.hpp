#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// NudeNet v3 检测结果，坐标为输入图像的像素坐标
struct NudeBox {
	float x, y, w, h;
	float score;
	int cls; // NudeClass
};

// 与 NudeNet 官方 __labels 顺序一致
enum NudeClass {
	FEMALE_GENITALIA_COVERED = 0,
	FACE_FEMALE = 1,
	BUTTOCKS_EXPOSED = 2,
	FEMALE_BREAST_EXPOSED = 3,
	FEMALE_GENITALIA_EXPOSED = 4,
	MALE_BREAST_EXPOSED = 5,
	ANUS_EXPOSED = 6,
	FEET_EXPOSED = 7,
	BELLY_COVERED = 8,
	FEET_COVERED = 9,
	ARMPITS_COVERED = 10,
	ARMPITS_EXPOSED = 11,
	FACE_MALE = 12,
	BELLY_EXPOSED = 13,
	MALE_GENITALIA_EXPOSED = 14,
	ANUS_COVERED = 15,
	FEMALE_BREAST_COVERED = 16,
	BUTTOCKS_COVERED = 17,
	NUDE_CLASS_COUNT = 18,
};

const char *nude_class_name(int cls);

// NudeNet v3（320n.onnx / 640m.onnx，YOLOv8 结构，输入尺寸可变，需为 32 的倍数）。
// 前处理与官方一致：RGB、/255、右侧和下方补黑；不同的是不补成正方形，
// 而是长边缩放到 input_size、短边补到 32 的倍数（模型支持矩形输入，16:9 画面算量少约一半）。
class NudeDetector {
public:
	NudeDetector();
	~NudeDetector();

	// accel：0 = 有 GPU 方案就用，1 = 只用 CPU
	bool load(const std::string &model_path, int input_size, std::string &error, int accel = 1);
	bool loaded() const;
	// 实际使用的推理方案（"CPU"、"CoreML"、"CUDA"、"DirectML"）
	const std::string &provider() const { return provider_; }

	// rgba：RGBA8 像素，任意尺寸（内部缩放）。返回分数 >= min_score 的结果，按类别分别做 NMS。
	std::vector<NudeBox> detect(const uint8_t *rgba, int w, int h, int stride, float min_score,
				    float nms_threshold = 0.45f);

private:
	struct Impl;
	std::unique_ptr<Impl> impl_;
	std::string provider_ = "CPU";
};
