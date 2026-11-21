#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// 人脸框，坐标为输入图像的像素坐标
struct FaceBox {
	float x, y, w, h;
	float score;
};

// YuNet（opencv_zoo face_detection_yunet_2023mar.onnx）人脸检测器。
// 模型输入固定 1x3x640x640，BGR，0~255，不做归一化。
// 输入图像宽高都不能超过 640，右侧和下方补零。
class FaceDetector {
public:
	static constexpr int kInputSize = 640;

	FaceDetector();
	~FaceDetector();

	// accel：0 = 有 GPU 方案就用，1 = 只用 CPU
	bool load(const std::string &model_path, std::string &error, int accel = 1);
	bool loaded() const;
	// 实际使用的推理方案（"CPU"、"CoreML"、"CUDA"、"DirectML"）
	const std::string &provider() const { return provider_; }

	// rgba：RGBA8 像素，stride 为每行字节数。w、h <= 640。
	std::vector<FaceBox> detect(const uint8_t *rgba, int w, int h, int stride, float score_threshold,
				    float nms_threshold = 0.3f);

private:
	struct Impl;
	std::unique_ptr<Impl> impl_;
	std::string provider_ = "CPU";
};
