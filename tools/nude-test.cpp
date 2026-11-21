// 离线测试 NudeNet：读图片，检测后把框打码写出 PNG，并打印每个框。
// 用法：nude-test <模型.onnx> <输入尺寸 320|640> <输入图片> <输出.png> [最低分数] [gpu]
#include "nude-detector.hpp"
#include "ort-runtime.hpp"

#include "stb_image.h"
#include "stb_image_write.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>

int main(int argc, char **argv)
{
	std::string ort_info;
	ort_runtime_init("", ort_info); // 测试工具只在 macOS 构建，直接用链接进来的运行库
	if (argc < 5) {
		fprintf(stderr, "用法：%s <模型.onnx> <输入尺寸> <输入图片> <输出.png> [最低分数] [gpu]\n", argv[0]);
		return 2;
	}
	int size = atoi(argv[2]);
	float min_score = argc > 5 ? (float)atof(argv[5]) : 0.25f;

	NudeDetector det;
	std::string err;
	const bool gpu = argc > 6 && std::string(argv[6]) == "gpu";
	if (!det.load(argv[1], size, err, gpu ? 0 : 1)) {
		fprintf(stderr, "模型加载失败：%s\n", err.c_str());
		return 1;
	}
	int w, h, n;
	uint8_t *img = stbi_load(argv[3], &w, &h, &n, 4);
	if (!img) {
		fprintf(stderr, "读图失败：%s\n", argv[3]);
		return 1;
	}

	std::vector<NudeBox> boxes = det.detect(img, w, h, w * 4, min_score); // 预热（GPU 首次推理要编译）
	const int runs = 10;
	auto t0 = std::chrono::steady_clock::now();
	for (int i = 0; i < runs; i++)
		boxes = det.detect(img, w, h, w * 4, min_score);
	double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / runs;
	printf("%s %dx%d，%s，平均每次 %.1f ms，%zu 个框\n", argv[3], w, h, det.provider().c_str(), ms, boxes.size());

	for (const NudeBox &b : boxes) {
		printf("  %-26s %.2f  x=%.0f y=%.0f w=%.0f h=%.0f\n", nude_class_name(b.cls), b.score, b.x, b.y, b.w,
		       b.h);
		// 半透明红色填充，方便目视核对位置
		for (int y = std::max(0, (int)b.y); y < std::min(h, (int)(b.y + b.h)); y++)
			for (int x = std::max(0, (int)b.x); x < std::min(w, (int)(b.x + b.w)); x++) {
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
