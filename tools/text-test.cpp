// 离线测试敏感文字识别：OCR 每一行，按规则和关键词判断，命中的行标红、其余标绿，写出 PNG。
// 用法：text-test <检测模型> <识别模型> <输入图片> <输出.png> [关键词列表，逗号分隔] [检测分辨率上限]
#include "text-detector.hpp"
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
		fprintf(stderr, "用法：%s <检测模型> <识别模型> <输入图片> <输出.png> [关键词] [检测分辨率上限]\n", argv[0]);
		return 2;
	}
	TextRules rules;
	if (argc > 5)
		rules.keywords = parse_keywords(argv[5]);
	int limit = argc > 6 ? atoi(argv[6]) : 960;

	TextDetector det;
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

	std::vector<TextLine> lines;
	double ms[2];
	for (int run = 0; run < 2; run++) {
		auto t0 = std::chrono::steady_clock::now();
		lines = det.detect(img, w, h, w * 4, limit);
		ms[run] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
	}
	printf("%s %dx%d，检测上限 %d，首次 %.1f ms，第二次（缓存）%.1f ms，%zu 行\n", argv[3], w, h, limit, ms[0], ms[1],
	       lines.size());

	int hits = 0;
	for (const TextLine &l : lines) {
		std::string reason;
		bool hit = match_sensitive(l.text, rules, reason);
		hits += hit;
		printf("  %s [%4.0f,%4.0f %3.0fx%2.0f] %.2f %s%s%s\n", hit ? "打码" : "    ", l.x, l.y, l.w, l.h, l.score,
		       l.text.c_str(), hit ? "  ← " : "", hit ? reason.c_str() : "");
		for (int y = std::max(0, (int)l.y); y < std::min(h, (int)(l.y + l.h)); y++)
			for (int x = std::max(0, (int)l.x); x < std::min(w, (int)(l.x + l.w)); x++) {
				uint8_t *p = img + ((size_t)y * w + x) * 4;
				if (hit) {
					p[0] = (uint8_t)((p[0] + 255) / 2);
					p[1] = (uint8_t)(p[1] / 2);
					p[2] = (uint8_t)(p[2] / 2);
				} else {
					p[0] = (uint8_t)(p[0] * 3 / 4);
					p[1] = (uint8_t)((p[1] * 3 + 255) / 4);
					p[2] = (uint8_t)(p[2] * 3 / 4);
				}
			}
	}
	printf("命中 %d 行\n", hits);
	stbi_write_png(argv[4], w, h, 4, img, w * 4);
	stbi_image_free(img);
	return 0;
}
