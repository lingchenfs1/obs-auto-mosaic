#!/usr/bin/env bash
# 下载测试图片（不放进仓库，图里是真人），并生成延迟对齐测试视频。
# 来源和许可见 test/nsfw-benign/SOURCES.md、test/clothed/SOURCES.md。
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}/test"
UA="obs-auto-mosaic-test/0.1"

fetch() { # <目标路径> <URL>
	if [ ! -s "$1" ]; then
		echo "下载 $1"
		mkdir -p "$(dirname "$1")"
		curl -fsSL -A "${UA}" -o "$1" "$2"
	fi
}

# OpenCV 示例图（人脸）
fetch group.jpg https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/example_outputs/largest_selfie.jpg
fetch faces.jpg https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/messi5.jpg

# Wikimedia Commons：非露骨的健美、沙滩排球照
W=https://upload.wikimedia.org/wikipedia/commons
T=https://thumb.wikimedia.org/wikipedia/commons/thumb
fetch nsfw-benign/bodybuilder.jpg "${W}/4/4f/Bodybuilder_John_Quinlan_On_Stage_1998.jpg"
fetch nsfw-benign/lineup.jpg "${T}/c/c1/Mohammed_Daood_MoDaood_bodybuilding_competition_lineup.jpg/1280px-Mohammed_Daood_MoDaood_bodybuilding_competition_lineup.jpg"
fetch nsfw-benign/volleyball1.jpg "${T}/7/7e/Beach_volleyball-Huntington_Beach-California_3.jpg/1280px-Beach_volleyball-Huntington_Beach-California_3.jpg"
fetch nsfw-benign/volleyball2.jpg "${T}/9/98/Women%27s_Beach_Volleyball%2C_London_2012.jpg/1280px-Women%27s_Beach_Volleyball%2C_London_2012.jpg"

# Wikimedia Commons：穿衣服的人像（裸露检测的对照）
fetch clothed/tshirt.jpg "${T}/c/cf/Portrait_Photo_Of_Young_Man_Wearing_A_Teal_Shirt.tif/lossy-page1-1280px-Portrait_Photo_Of_Young_Man_Wearing_A_Teal_Shirt.tif.jpg"
fetch clothed/desk1.jpg "${T}/f/fb/Frustrated_woman_holds_telephone_receiver_while_working_at_laptop.jpg/1280px-Frustrated_woman_holds_telephone_receiver_while_working_at_laptop.jpg"
fetch clothed/desk2.jpg "${T}/3/3a/Woman_stretches_at_home_office_desk_while_working_on_laptop.jpg/1280px-Woman_stretches_at_home_office_desk_while_working_on_laptop.jpg"
fetch clothed/robbie.jpg "${T}/c/cc/Robbie_Pierce.jpg/1280px-Robbie_Pierce.jpg"

# 延迟对齐测试视频（需要 Pillow 和 ffmpeg）
if [ ! -s moving-faces.mp4 ]; then
	echo "生成 moving-faces.mp4"
	python3 "${ROOT}/tools/make-test-video.py" group.jpg moving-faces.mp4
fi
echo "测试素材已就绪"
