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

# Wikimedia Commons：血腥暴力检测测试图（仿真伤口化妆、冲突现场）和容易误判的反例
fetch violence/gore-mask.jpg "${T}/d/d4/Homemade_liquid_latex_SFX_mask_for_a_haunted_house.jpg/1280px-Homemade_liquid_latex_SFX_mask_for_a_haunted_house.jpg"
fetch violence/gore-wrist.jpg "${T}/d/d7/Healthcare_Simulation_Moulage_Standardized_Patient_Actor_Deep_Gash_Wound_Cut_on_Wrist.jpg/1280px-Healthcare_Simulation_Moulage_Standardized_Patient_Actor_Deep_Gash_Wound_Cut_on_Wrist.jpg"
fetch violence/gore-burn.jpg "${T}/7/7a/First_aid_training_moulage_burn_simulation_in_Bosnia_and_Herzegovina_%282026%29-bs.jpg/1280px-First_aid_training_moulage_burn_simulation_in_Bosnia_and_Herzegovina_%282026%29-bs.jpg"
fetch violence/violence-clash.jpg "${W}/8/8b/Police_Clash_-_geograph.org.uk_-_3093340.jpg"
fetch violence/violence-riot.jpg "${T}/0/0b/Riot_police_%28Berkut%29_and_protesters%2C_clash_at_Bankova_str%2C_Kiev%2C_Ukraine._December_1%2C_2013.jpg/1280px-Riot_police_%28Berkut%29_and_protesters%2C_clash_at_Bankova_str%2C_Kiev%2C_Ukraine._December_1%2C_2013.jpg"
fetch violence/safe-moulage-prep.jpg "${T}/1/1e/General_receives_head_wound_from_pastry_chef%2C_medical_training_ensues_%289092533%29.jpg/1280px-General_receives_head_wound_from_pastry_chef%2C_medical_training_ensues_%289092533%29.jpg"
fetch violence/safe-kickboxing.jpg "${W}/5/5a/Amateur_Kick_boxing_%2872386820%29.jpg"
fetch violence/safe-fps-game.png "${W}/2/22/Anarch_screenshot.png"
fetch violence/safe-spaghetti.jpg "${T}/0/0b/Liat_Portal_for_Foodie_Disorder_-_Israeli_schnitzel_with_spaghetti_and_tomato_sauce.jpg/1280px-Liat_Portal_for_Foodie_Disorder_-_Israeli_schnitzel_with_spaghetti_and_tomato_sauce.jpg"

# 敏感文字测试画面（需要 Pillow；里面的隐私信息都是示例值）
if [ ! -s text-sample.png ]; then
	echo "生成 text-sample.png"
	python3 "${ROOT}/tools/make-text-test.py" text-sample.png
fi

# 延迟对齐测试视频（需要 Pillow 和 ffmpeg）
if [ ! -s moving-faces.mp4 ]; then
	echo "生成 moving-faces.mp4"
	python3 "${ROOT}/tools/make-test-video.py" group.jpg moving-faces.mp4
fi
echo "测试素材已就绪"
