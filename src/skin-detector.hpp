#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// 裸露皮肤区域，坐标为输入图像的像素坐标
struct SkinBox {
	float x, y, w, h;
	float body_ratio;  // 所属的人全身皮肤占比
	float torso_ratio; // 所属的人躯干皮肤占比（没有脸时为 -1）
};

struct SkinParams {
	float body_threshold = 0.4f;  // 全身皮肤占比 >= 它判定为裸露
	float torso_threshold = 0.6f;  // 或者躯干皮肤占比 >= 它
	float face_threshold = 0.7f;   // 用来取肤色、排除脸部的人脸检测阈值
};

// 每个人（分割连通块）的统计，调试用
struct SkinPerson {
	int x0, y0, x1, y1; // 网格坐标
	float body_ratio, torso_ratio;
	bool exposed;
};

// 「裸露皮肤」检测：PP-HumanSeg 人体分割 + 肤色。
// 肤色：通用 YCrCb 规则，加上按每张脸取样的自适应肤色（适应灯光和肤色深浅）。
// 一个人全身皮肤占比够高、或脸下方躯干的皮肤占比够高时，把他身上的皮肤块（不含脸）都输出。
class SkinDetector {
public:
	SkinDetector();
	~SkinDetector();

	bool load(const std::string &seg_model, const std::string &face_model, std::string &error);
	bool loaded() const;

	// rgba：RGBA8 像素，w、h <= 640
	std::vector<SkinBox> detect(const uint8_t *rgba, int w, int h, int stride, const SkinParams &params,
				    std::vector<SkinPerson> *persons = nullptr);

private:
	struct Impl;
	std::unique_ptr<Impl> impl_;
};
