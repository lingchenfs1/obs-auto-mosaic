#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// 一行文字，坐标为输入图像的像素坐标
struct TextLine {
	float x, y, w, h;
	std::string text; // UTF-8
	float score;      // 识别置信度（各字符最大概率的平均）
	bool cached;      // 是否命中识别缓存
};

// 敏感文字规则
struct TextRules {
	bool phone = true;    // 手机号
	bool id_card = true;  // 身份证号
	bool bank_card = true; // 银行卡号（16~19 位连续数字）
	bool email = true;
	bool url = true;
	bool im = true; // 微信 / QQ 等账号（关键字后面跟账号）
	std::vector<std::string> keywords; // 自定义关键词（匹配前同样规整）
};

// 文字规整：去掉空白，全角 ASCII 转半角，英文转小写
std::string normalize_text(const std::string &utf8);

// 解析关键词列表：按换行、逗号（含全角）、分号分隔，规整后去空
std::vector<std::string> parse_keywords(const std::string &list);

// 判断一行文字是否敏感；命中时 reason 写入命中的规则名
bool match_sensitive(const std::string &text, const TextRules &rules, std::string &reason);

// PP-OCRv4（RapidOCR 导出的 ONNX）文字检测 + 识别。
// 检测：长边缩到 det_limit 以内（32 的倍数），BGR，ImageNet 均值方差；概率图 > 0.3 二值化，
//       连通块取外接矩形，按 DB 的 unclip 比例 1.5 外扩。只处理横排文字。
// 识别：从原图裁行，缩到 48 高（宽至少 320，右侧补 0），BGR，(v/255-0.5)/0.5；CTC 贪心解码，
//       字典取自模型元数据 "character"。同一位置、内容没变的行直接用缓存。
class TextDetector {
public:
	TextDetector();
	~TextDetector();

	bool load(const std::string &det_model, const std::string &rec_model, std::string &error);
	bool loaded() const;

	// rgba：RGBA8 像素，任意尺寸（建议原分辨率或 1920 以内，小字才认得出）
	std::vector<TextLine> detect(const uint8_t *rgba, int w, int h, int stride, int det_limit = 960);

private:
	struct Impl;
	std::unique_ptr<Impl> impl_;
};
