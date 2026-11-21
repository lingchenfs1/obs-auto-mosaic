// 离线测试：读一张图片，缩放到 640 以内跑 YuNet，在人脸处打马赛克后写出 PNG。
// 用法：detect-test <模型.onnx> <输入图片> <输出.png> [阈值]
#include "face-detector.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <vector>

static std::vector<uint8_t> resize_bilinear(const uint8_t *src, int sw, int sh, int dw, int dh)
{
	std::vector<uint8_t> dst((size_t)dw * dh * 4);
	for (int y = 0; y < dh; y++) {
		float fy = (y + 0.5f) * sh / dh - 0.5f;
		int y0 = std::clamp((int)fy, 0, sh - 1), y1 = std::min(y0 + 1, sh - 1);
		float ty = std::clamp(fy - y0, 0.0f, 1.0f);
		for (int x = 0; x < dw; x++) {
			float fx = (x + 0.5f) * sw / dw - 0.5f;
			int x0 = std::clamp((int)fx, 0, sw - 1), x1 = std::min(x0 + 1, sw - 1);
			float tx = std::clamp(fx - x0, 0.0f, 1.0f);
			for (int c = 0; c < 4; c++) {
				float a = src[(y0 * sw + x0) * 4 + c] * (1 - tx) + src[(y0 * sw + x1) * 4 + c] * tx;
				float b = src[(y1 * sw + x0) * 4 + c] * (1 - tx) + src[(y1 * sw + x1) * 4 + c] * tx;
				dst[((size_t)y * dw + x) * 4 + c] = (uint8_t)(a * (1 - ty) + b * ty + 0.5f);
			}
		}
	}
	return dst;
}

static void pixelate(uint8_t *img, int w, int h, int x0, int y0, int x1, int y1, int block)
{
	x0 = std::clamp(x0, 0, w);
	x1 = std::clamp(x1, 0, w);
	y0 = std::clamp(y0, 0, h);
	y1 = std::clamp(y1, 0, h);
	for (int by = y0 - y0 % block; by < y1; by += block) {
		for (int bx = x0 - x0 % block; bx < x1; bx += block) {
			int ex = std::min(bx + block, w), ey = std::min(by + block, h);
			long sum[4] = {0, 0, 0, 0};
			int n = 0;
			for (int y = by; y < ey; y++)
				for (int x = bx; x < ex; x++, n++)
					for (int c = 0; c < 4; c++)
						sum[c] += img[(y * w + x) * 4 + c];
			for (int y = std::max(by, y0); y < std::min(ey, y1); y++)
				for (int x = std::max(bx, x0); x < std::min(ex, x1); x++)
					for (int c = 0; c < 4; c++)
						img[(y * w + x) * 4 + c] = (uint8_t)(sum[c] / n);
		}
	}
}

int main(int argc, char **argv)
{
	if (argc < 4) {
		fprintf(stderr, "用法：%s <模型.onnx> <输入图片> <输出.png> [阈值]\n", argv[0]);
		return 2;
	}
	float threshold = argc > 4 ? (float)atof(argv[4]) : 0.6f;

	FaceDetector det;
	std::string err;
	if (!det.load(argv[1], err)) {
		fprintf(stderr, "模型加载失败：%s\n", err.c_str());
		return 1;
	}

	int w, h, n;
	uint8_t *img = stbi_load(argv[2], &w, &h, &n, 4);
	if (!img) {
		fprintf(stderr, "读图失败：%s\n", argv[2]);
		return 1;
	}

	float scale = std::min(1.0f, std::min(640.0f / w, 640.0f / h));
	int dw = std::max(1, (int)(w * scale + 0.5f)), dh = std::max(1, (int)(h * scale + 0.5f));
	auto small = resize_bilinear(img, w, h, dw, dh);

	std::vector<FaceBox> faces;
	auto t0 = std::chrono::steady_clock::now();
	const int runs = 20;
	for (int i = 0; i < runs; i++)
		faces = det.detect(small.data(), dw, dh, dw * 4, threshold);
	double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / runs;

	printf("图片 %dx%d，检测输入 %dx%d，平均每次 %.2f ms，检出 %zu 张脸\n", w, h, dw, dh, ms, faces.size());
	for (auto &f : faces) {
		float x = f.x / scale, y = f.y / scale, fw = f.w / scale, fh = f.h / scale;
		printf("  score=%.3f  x=%.0f y=%.0f w=%.0f h=%.0f\n", f.score, x, y, fw, fh);
		float mx = fw * 0.25f, my = fh * 0.25f;
		pixelate(img, w, h, (int)(x - mx), (int)(y - my), (int)(x + fw + mx), (int)(y + fh + my),
			 std::max(8, (int)(fw / 8)));
	}

	if (!stbi_write_png(argv[3], w, h, 4, img, w * 4)) {
		fprintf(stderr, "写图失败：%s\n", argv[3]);
		return 1;
	}
	stbi_image_free(img);
	return 0;
}
