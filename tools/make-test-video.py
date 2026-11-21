"""生成延迟对齐测试视频：一张脸快速横移，一张脸每半秒突然出现/消失。

用法：python3 tools/make-test-video.py test/group.jpg test/moving-faces.mp4
需要 Pillow 和 ffmpeg。
"""
import subprocess
import sys

from PIL import Image, ImageDraw

W, H, FPS, SECONDS = 1920, 1080, 30, 8
SPEED = 1500  # 像素/秒


def face_tile(img, box, scale):
    x, y, w, h = box
    pad = 0.3
    crop = img.crop((int(x - w * pad), int(y - h * pad), int(x + w * (1 + pad)), int(y + h * (1 + pad))))
    return crop.resize((int(crop.width * scale), int(crop.height * scale)), Image.BICUBIC)


def main():
    src, out = sys.argv[1], sys.argv[2]
    img = Image.open(src).convert("RGB")
    # group.jpg 里前排两张脸（坐标来自 detect-test 输出）
    moving = face_tile(img, (1132, 874, 140, 184), 1.6)
    popping = face_tile(img, (1428, 764, 137, 178), 1.6)

    bg = Image.new("RGB", (W, H))
    d = ImageDraw.Draw(bg)
    for y in range(H):
        c = int(60 + 80 * y / H)
        d.line([(0, y), (W, y)], fill=(c, c + 10, c + 25))

    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
         "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", out],
        stdin=subprocess.PIPE)
    span = W - moving.width
    for i in range(FPS * SECONDS):
        t = i / FPS
        frame = bg.copy()
        # 来回匀速移动
        p = (t * SPEED) % (2 * span)
        x = p if p < span else 2 * span - p
        frame.paste(moving, (int(x), 80))
        # 每 0.5 秒切换一次显示
        if int(t * 2) % 2 == 1:
            frame.paste(popping, (W // 2 - popping.width // 2, H - popping.height - 60))
        ff.stdin.write(frame.tobytes())
    ff.stdin.close()
    sys.exit(ff.wait())


if __name__ == "__main__":
    main()
