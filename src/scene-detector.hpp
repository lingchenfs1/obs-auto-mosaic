#pragma once

#include <cstdint>
#include <memory>
#include <string>

// 整个画面「血腥 / 暴力」的概率（0~1）
struct SceneScores {
	float gore = 0.0f;
	float violence = 0.0f;
};

// CLIP（ViT-B/32，int8）零样本判别：画面和预先算好的文字描述向量（血腥、暴力、正常三组）比较，
// 对全部描述做 softmax，各组概率相加。沿长边取左 / 中 / 右三块正方形分别判断，各取最高分，
// 避免 16:9 画面只看中间漏掉两侧。只判断整个画面，不给位置。
class SceneDetector {
public:
	SceneDetector();
	~SceneDetector();

	// vision_model：clip_vision_int8.onnx；prompts：clip_prompts.bin（tools/make-clip-prompts.py 生成）
	bool load(const std::string &vision_model, const std::string &prompts, std::string &error);
	bool loaded() const;

	// rgba：RGBA8 像素，任意尺寸
	SceneScores detect(const uint8_t *rgba, int w, int h, int stride);

private:
	struct Impl;
	std::unique_ptr<Impl> impl_;
};
