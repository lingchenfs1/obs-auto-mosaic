"""生成敏感文字测试画面（1920x1080 PNG）：聊天窗口 + 普通文字，混合几种字号。
里面的「隐私信息」都是示例值（身份证号是 GB 11643 文档里的示例，邮箱和网址用 example.com）。

用法：python3 tools/make-text-test.py <输出.png> [字体路径]
返回码 0；同时打印每行文字、是否应被打码。
"""
import sys

from PIL import Image, ImageDraw, ImageFont

# macOS 自带的中文字体，按顺序找第一个存在的
FONTS = ["/System/Library/Fonts/Hiragino Sans GB.ttc", "/System/Library/Fonts/STHeiti Medium.ttc",
         "/System/Library/Fonts/Supplemental/Songti.ttc", "/Library/Fonts/Arial Unicode.ttf"]

# (文字, 字号, 是否应打码)
LINES = [
    ("今天的直播就到这里，谢谢大家！", 34, False),
    ("有事联系我：13812345678", 30, True),
    ("加微信 demo_wx_2026 拉你进群", 28, True),
    ("身份证 11010519491231002X 别发出来", 26, True),
    ("银行卡号 6222 0212 3456 7890 123", 26, True),
    ("邮箱 test.user@example.com", 24, True),
    ("网址 www.example.com/live", 24, True),
    ("这句话里有测试敏感词要遮住", 28, True),
    ("下一局我们打排位，记得点关注", 30, False),
    ("QQ 123456789 有问题私聊", 22, True),
    ("游戏时长 02:35:17  分数 98765", 22, False),
    ("普通的小字说明文字 font size 18", 18, False),
]


def main():
    out = sys.argv[1]
    import os
    font_path = sys.argv[2] if len(sys.argv) > 2 else next(f for f in FONTS if os.path.exists(f))
    img = Image.new("RGB", (1920, 1080), (32, 36, 44))
    d = ImageDraw.Draw(img)
    # 左边：假装是游戏画面的色块
    for i in range(12):
        d.rectangle([60 + i * 70, 120 + (i % 3) * 40, 110 + i * 70, 700 - (i % 4) * 60],
                    fill=(60 + i * 12, 90 + (i * 7) % 80, 140 - i * 5))
    # 右边：聊天窗口
    d.rounded_rectangle([1060, 60, 1880, 1020], radius=18, fill=(245, 245, 247))
    y = 100
    for text, size, _ in LINES:
        font = ImageFont.truetype(font_path, size)
        d.text((1090, y), text, font=font, fill=(30, 30, 30))
        y += int(size * 2.1)
    # 左下：浅色背景上的标题
    d.text((80, 900), "直播间公告：文明发言", font=ImageFont.truetype(font_path, 40), fill=(250, 220, 120))
    img.save(out)
    for text, size, mask in LINES:
        print(("应打码  " if mask else "不打码  ") + text)


if __name__ == "__main__":
    main()
