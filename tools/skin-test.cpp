// 离线测试「裸露皮肤」检测：图片先缩到 640 以内（与插件一致），检测后把框标红写出 PNG。
// 用法：skin-test <分割模型> <人脸模型> <输入图片> <输出.png> [全身门槛] [躯干门槛]
#include "skin-detector.hpp"

#include "stb_image.h"
#include "stb_image_write.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
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
	if (argc < 5) {
		fprintf(stderr, "用法：%s <分割模型> <人脸模型> <输入图片> <输出.png> [全身门槛] [躯干门槛]\n", argv[0]);
		return 2;
	}
	SkinParams params;
	if (argc > 5)
		params.body_threshold = (float)atof(argv[5]);
	if (argc > 6)
		params.torso_threshold = (float)atof(argv[6]);

	SkinDetector det;
	std::string err;
	if (!det.load(argv[1], argv[2], err)) {
		fprintf(stderr, "模型加载失败：%s\n", err.c_str());
		return 1;
	}
	int w, h, n;
	uint8_t *img = stbi_load(argv[3], &w, &h, &n, 4);
	if (!img) {
		fprintf(stderr, "读图失败：%s\n", argv[3]);
		return 1;
	}
	float scale = std::min(1.0f, std::min(640.0f / w, 640.0f / h));
	int dw = std::max(1, (int)(w * scale + 0.5f)), dh = std::max(1, (int)(h * scale + 0.5f));
	auto small = resize_bilinear(img, w, h, dw, dh);

	std::vector<SkinBox> boxes;
	std::vector<SkinPerson> persons;
	const int runs = 10;
	auto t0 = std::chrono::steady_clock::now();
	for (int i = 0; i < runs; i++) {
		persons.clear();
		boxes = det.detect(small.data(), dw, dh, dw * 4, params, &persons);
	}
	double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / runs;

	printf("%s %dx%d，平均每次 %.1f ms，%zu 个人，%zu 个皮肤框\n", argv[3], w, h, ms, persons.size(), boxes.size());
	for (const SkinPerson &p : persons) {
		printf("  人 [%d,%d-%d,%d] 全身 %.0f%%", p.x0, p.y0, p.x1, p.y1, p.body_ratio * 100);
		if (p.torso_ratio >= 0)
			printf("，躯干 %.0f%%", p.torso_ratio * 100);
		else
			printf("，无脸");
		printf(" → %s\n", p.exposed ? "裸露" : "不算裸露");
	}
	for (const SkinBox &b : boxes) {
		int x0 = (int)(b.x / scale), y0 = (int)(b.y / scale);
		int x1 = std::min(w, (int)((b.x + b.w) / scale)), y1 = std::min(h, (int)((b.y + b.h) / scale));
		for (int y = std::max(0, y0); y < y1; y++)
			for (int x = std::max(0, x0); x < x1; x++) {
				uint8_t *p = img + ((size_t)y * w + x) * 4;
				p[0] = (uint8_t)((p[0] + 255) / 2);
				p[1] = (uint8_t)(p[1] / 2);
				p[2] = (uint8_t)(p[2] / 2);
			}
	}
	stbi_write_png(argv[4], w, h, 4, img, w * 4);
	stbi_image_free(img);
	return 0;
}
