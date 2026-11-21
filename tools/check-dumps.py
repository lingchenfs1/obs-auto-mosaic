"""核对调试导出帧：在未打码的原帧（*-raw.png）上跑人脸检测当作真值，
统计每张脸的框内有多少像素在输出帧里没被改动（= 露出来的比例）。

用法：python3 -I tools/check-dumps.py <detect-test 路径> <模型> <导出前缀> <组数>
"""
import re
import subprocess
import sys

from PIL import Image, ImageChops


def main():
    detect, model, prefix, count = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    worst = 0.0
    total_faces = 0
    for n in range(count):
        raw_path, out_path = f"{prefix}-{n}-raw.png", f"{prefix}-{n}.png"
        res = subprocess.run([detect, model, raw_path, f"{prefix}-{n}-det.png", "0.7"],
                             capture_output=True, text=True, check=True)
        boxes = [tuple(map(float, m)) for m in
                 re.findall(r"x=(-?\d+) y=(-?\d+) w=(\d+) h=(\d+)", res.stdout)]
        raw = Image.open(raw_path).convert("RGB")
        out = Image.open(out_path).convert("RGB")
        diff = ImageChops.difference(raw, out).convert("L")
        parts = []
        for x, y, w, h in boxes:
            # 只看人脸框中间 80%，避开检测框边缘误差
            box = (int(x + w * 0.1), int(y + h * 0.1), int(x + w * 0.9), int(y + h * 0.9))
            region = diff.crop(box)
            hist = region.histogram()
            same = sum(hist[:4])  # 差值 < 4 视为没改动
            exposed = same / max(1, region.width * region.height)
            worst = max(worst, exposed)
            total_faces += 1
            parts.append(f"({int(x)},{int(y)}) 露出 {exposed * 100:.1f}%")
        print(f"第 {n} 组：{len(boxes)} 张脸  " + "  ".join(parts))
    print(f"合计 {total_faces} 张脸，最大露出比例 {worst * 100:.1f}%")


if __name__ == "__main__":
    main()
