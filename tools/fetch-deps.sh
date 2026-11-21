#!/usr/bin/env bash
# 下载构建所需的依赖和模型（都不放进仓库）：
#   third_party/obs-studio   libobs 头文件（版本跟已安装的 OBS 一致）
#   third_party/onnxruntime  ONNX Runtime 预编译包（NuGet 官方包里的 macOS arm64 动态库和头文件）
#   data/                    YuNet、PP-HumanSeg（opencv_zoo）、NudeNet v3（320n / 640m）
# 用法：tools/fetch-deps.sh [OBS 版本，默认 33.0.1] [onnxruntime 版本，默认 1.32.0]
set -euo pipefail

OBS_VERSION="${1:-33.0.1}"
ORT_VERSION="${2:-1.32.0}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

# ---- libobs 头文件 ----
if [ ! -d third_party/obs-studio/libobs ]; then
	echo "下载 obs-studio ${OBS_VERSION} 的 libobs 头文件"
	git clone --depth 1 --branch "${OBS_VERSION}" --filter=blob:none --sparse \
		https://github.com/obsproject/obs-studio.git third_party/obs-studio
	git -C third_party/obs-studio sparse-checkout set libobs
fi

# ---- ONNX Runtime ----
if [ ! -f third_party/onnxruntime/lib/libonnxruntime.dylib ]; then
	echo "下载 onnxruntime ${ORT_VERSION}（NuGet）"
	tmp="$(mktemp -d)"
	curl -fsSL -o "${tmp}/ort.nupkg" \
		"https://api.nuget.org/v3-flatcontainer/microsoft.ml.onnxruntime/${ORT_VERSION}/microsoft.ml.onnxruntime.${ORT_VERSION}.nupkg"
	unzip -q "${tmp}/ort.nupkg" 'build/native/include/*' 'runtimes/osx-arm64/native/*' -d "${tmp}/x"
	mkdir -p third_party/onnxruntime/lib
	rm -rf third_party/onnxruntime/include
	mv "${tmp}/x/build/native/include" third_party/onnxruntime/include
	mv "${tmp}/x/runtimes/osx-arm64/native/libonnxruntime.dylib" third_party/onnxruntime/lib/
	rm -rf "${tmp}"
fi

# ---- 模型 ----
fetch() { # <目标路径> <URL>
	if [ ! -s "$1" ]; then
		echo "下载 $1"
		mkdir -p "$(dirname "$1")"
		curl -fsSL -o "$1" "$2"
	fi
}
fetch data/face_detection_yunet_2023mar.onnx \
	https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
fetch data/seg/pphumanseg_2023mar.onnx \
	https://github.com/opencv/opencv_zoo/raw/main/models/human_segmentation_pphumanseg/human_segmentation_pphumanseg_2023mar.onnx

# NudeNet 的 release 文件直链会跳转到 GitHub 登录页，需要用已登录的 gh 下载
mkdir -p data/nudenet
for m in 320n.onnx 640m.onnx; do
	if [ ! -s "data/nudenet/${m}" ]; then
		if command -v gh >/dev/null; then
			echo "下载 NudeNet ${m}"
			gh release download v3.4-weights -R notAI-tech/NudeNet -p "${m}" -D data/nudenet
		else
			echo "缺少 data/nudenet/${m}：请安装并登录 gh（gh auth login）后重跑，" >&2
			echo "或手动从 https://github.com/notAI-tech/NudeNet/releases/tag/v3.4-weights 下载" >&2
			exit 1
		fi
	fi
done

# 下载到的模型如果是网页（被重定向到登录页等），文件开头会是 HTML
for f in data/face_detection_yunet_2023mar.onnx data/seg/pphumanseg_2023mar.onnx data/nudenet/*.onnx; do
	if grep -qi "<html\|<!doctype" < <(head -c 64 "${f}"); then
		echo "${f} 下载到的是网页而不是模型，请删掉后重试" >&2
		exit 1
	fi
done
echo "依赖已就绪"
