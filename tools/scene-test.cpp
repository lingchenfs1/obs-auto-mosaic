// 离线测试「血腥 / 暴力」检测：图片先缩到 640 以内（与插件一致），输出两个概率和耗时。
// 用法：scene-test <clip_vision_int8.onnx> <clip_prompts.bin> <图片>...
#include "ort-runtime.hpp"
#include "scene-detector.hpp"

#include "stb_image.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <vector>

static std::vector<uint8_t> resize_bilinear(const uint8_t *src, int sw, int sh, int dw, int dh)
{
	std::vector<uint8_t> dst((size_t)dw * dh * 4);
	for (int y = 0; y < dh; y++) {
		float fy = std::clamp((y + 0.5f) * sh / dh - 0.5f, 0.0f, (float)(sh - 1));
		int y0 = (int)fy, y1 = std::min(y0 + 1, sh - 1);
		float ty = fy - y0;
		for (int x = 0; x < dw; x++) {
			float fx = std::clamp((x + 0.5f) * sw / dw - 0.5f, 0.0f, (float)(sw - 1));
			int x0 = (int)fx, x1 = std::min(x0 + 1, sw - 1);
			float tx = fx - x0;
			for (int c = 0; c < 4; c++) {
				float a = src[(y0 * sw + x0) * 4 + c] * (1 - tx) + src[(y0 * sw + x1) * 4 + c] * tx;
				float b = src[(y1 * sw + x0) * 4 + c] * (1 - tx) + src[(y1 * sw + x1) * 4 + c] * tx;
				dst[((size_t)y * dw + x) * 4 + c] = (uint8_t)(a * (1 - ty) + b * ty + 0.5f);
			}
		}
	}
	return dst;
}

int main(int argc, char **argv)
{
	std::string info;
	ort_runtime_init("", info); // 测试工具只在 macOS 构建，直接用链接进来的运行库
	if (argc < 4) {
		fprintf(stderr, "用法：%s <clip_vision_int8.onnx> <clip_prompts.bin> <图片>...\n", argv[0]);
		return 2;
	}
	SceneDetector det;
	std::string err;
	if (!det.load(argv[1], argv[2], err)) {
		fprintf(stderr, "加载失败：%s\n", err.c_str());
		return 1;
	}
	for (int i = 3; i < argc; i++) {
		int w, h, n;
		uint8_t *img = stbi_load(argv[i], &w, &h, &n, 4);
		if (!img) {
			fprintf(stderr, "读图失败：%s\n", argv[i]);
			continue;
		}
		float scale = std::min(1.0f, std::min(640.0f / w, 640.0f / h));
		int dw = std::max(1, (int)(w * scale + 0.5f)), dh = std::max(1, (int)(h * scale + 0.5f));
		auto small = resize_bilinear(img, w, h, dw, dh);
		stbi_image_free(img);

		SceneScores s = det.detect(small.data(), dw, dh, dw * 4); // 预热
		const int runs = 5;
		auto t0 = std::chrono::steady_clock::now();
		for (int r = 0; r < runs; r++)
			s = det.detect(small.data(), dw, dh, dw * 4);
		double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / runs;
		const char *name = argv[i];
		for (const char *p = argv[i]; *p; p++)
			if (*p == '/')
				name = p + 1;
		printf("%-24s 血腥 %.2f  暴力 %.2f  %5.1f ms%s\n", name, s.gore, s.violence, ms,
		       std::max(s.gore, s.violence) >= 0.5f ? "  ← 整屏打码" : "");
	}
	return 0;
}
