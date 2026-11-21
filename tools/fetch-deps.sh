#!/usr/bin/env bash
# 下载构建所需的依赖和模型（都不放进仓库）：
#   third_party/obs-studio   libobs 头文件（版本跟目标 OBS 一致）
#   data/                    YuNet、PP-HumanSeg（opencv_zoo）、PP-OCRv4（RapidOCR 导出的 ONNX）、NudeNet v3（320n / 640m）、
#                            CLIP ViT-B/32 图像模型（Xenova 导出的 int8 ONNX）
# macOS：
#   third_party/onnxruntime  ONNX Runtime（NuGet 官方包里的 macOS arm64 动态库和头文件）
# Windows（Git Bash，GitHub Actions 用）：
#   third_party/obs-win/obs.dll      OBS 发布包里的 obs.dll（之后用 tools/make-import-lib.py 生成 obs.lib）
#   third_party/onnxruntime/include  ONNX Runtime 头文件（取 DirectML 包的 1.24，新版运行库兼容旧版接口）
#   third_party/ort-win/dml          onnxruntime.dll（DirectML 版）+ DirectML.dll
#   third_party/ort-win/cuda         onnxruntime*.dll（CUDA 版，可选，体积大）+ 随包的 CUDA 13 / cuDNN 9 运行库
#                                    （取自 NVIDIA 在 PyPI 发布的官方 wheel，许可证放在 licenses/ 下）
# 用法：tools/fetch-deps.sh
# 版本可用环境变量覆盖：OBS_VERSION、ORT_VERSION（macOS）、ORT_DML_VERSION、DIRECTML_VERSION、ORT_CUDA_VERSION、
# 设 SKIP_CUDA=1 不下载 CUDA 版运行库。
set -euo pipefail

OBS_VERSION="${OBS_VERSION:-33.0.1}"
ORT_VERSION="${ORT_VERSION:-1.32.0}"
ORT_DML_VERSION="${ORT_DML_VERSION:-1.24.4}"
DIRECTML_VERSION="${DIRECTML_VERSION:-1.15.4}"
ORT_CUDA_VERSION="${ORT_CUDA_VERSION:-1.32.0}"
# 随 CUDA 扩展包分发的 NVIDIA 运行库（PyPI 包名==版本）。onnxruntime 1.32 CUDA 版需要 CUDA 13 + cuDNN 9
NVIDIA_WHEELS="${NVIDIA_WHEELS:-nvidia-cuda-runtime==13.5.37 nvidia-cublas==13.10.0.20 nvidia-cudnn-cu13==9.28.1.4 nvidia-cufft==12.5.0.14 nvidia-cuda-nvrtc==13.5.37}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

PY="$(command -v python3 || command -v python)"
case "$(uname -s)" in
MINGW* | MSYS* | CYGWIN*) OS=windows ;;
Darwin) OS=macos ;;
*) OS=other ;;
esac

# 从 zip / nupkg 里解出若干成员：unzip_members <压缩包> <目标目录> <成员前缀>...
# 只取以给定前缀开头的文件，按原路径放到目标目录下（用 Python，Git Bash 不一定有 unzip）
unzip_members() {
	"${PY}" - "$@" <<'PYEOF'
import os, sys, zipfile
src, dst, prefixes = sys.argv[1], sys.argv[2], sys.argv[3:]
with zipfile.ZipFile(src) as z:
    for n in z.namelist():
        if not n.endswith("/") and any(n.startswith(p) for p in prefixes):
            out = os.path.join(dst, *n.split("/"))
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with z.open(n) as f, open(out, "wb") as o:
                o.write(f.read())
PYEOF
}

# 从 PyPI 下载某个包某个版本的 win_amd64 wheel：pypi_wheel <包名> <版本> <保存路径>
pypi_wheel() {
	local url
	url="$(curl -fsSL "https://pypi.org/pypi/$1/$2/json" | "${PY}" -c "
import json, sys
urls = [f['url'] for f in json.load(sys.stdin)['urls'] if f['filename'].endswith('win_amd64.whl')]
print(urls[0] if urls else '')")"
	if [ -z "${url}" ]; then
		echo "PyPI 上找不到 $1 $2 的 Windows 包" >&2
		return 1
	fi
	curl -fsSL -o "$3" "${url}"
}

# 从 wheel 里取出所有 .dll（不管在哪个子目录，平铺到目标目录）和许可证文件（放到 目标目录/licenses/<包名>-<文件名>）
extract_wheel_dlls() { # <wheel> <目标目录> <包名>
	"${PY}" - "$@" <<'PYEOF'
import os, sys, zipfile
whl, dst, pkg = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(os.path.join(dst, "licenses"), exist_ok=True)
n_dll = 0
with zipfile.ZipFile(whl) as z:
    for n in z.namelist():
        base = n.rsplit("/", 1)[-1]
        if base.lower().endswith(".dll"):
            out = os.path.join(dst, base)
            n_dll += 1
        elif ".dist-info/" in n and base.lower().startswith(("license", "notice")):
            out = os.path.join(dst, "licenses", f"{pkg}-{base}")
        else:
            continue
        with z.open(n) as f, open(out, "wb") as o:
            o.write(f.read())
print(f"  {pkg}: {n_dll} 个 DLL")
PYEOF
}

nuget() { # <包名（小写）> <版本> <保存路径>
	curl -fsSL -o "$3" "https://api.nuget.org/v3-flatcontainer/$1/$2/$1.$2.nupkg"
}

# ---- libobs 头文件 ----
if [ ! -d third_party/obs-studio/libobs ]; then
	echo "下载 obs-studio ${OBS_VERSION} 的 libobs 头文件"
	git clone --depth 1 --branch "${OBS_VERSION}" --filter=blob:none --sparse \
		https://github.com/obsproject/obs-studio.git third_party/obs-studio
	git -C third_party/obs-studio sparse-checkout set libobs
fi

if [ "${OS}" = macos ]; then
	# ---- ONNX Runtime（macOS）----
	if [ ! -f third_party/onnxruntime/lib/libonnxruntime.dylib ]; then
		echo "下载 onnxruntime ${ORT_VERSION}（NuGet）"
		tmp="$(mktemp -d)"
		nuget microsoft.ml.onnxruntime "${ORT_VERSION}" "${tmp}/ort.nupkg"
		unzip_members "${tmp}/ort.nupkg" "${tmp}/x" build/native/include/ runtimes/osx-arm64/native/
		mkdir -p third_party/onnxruntime/lib
		rm -rf third_party/onnxruntime/include
		mv "${tmp}/x/build/native/include" third_party/onnxruntime/include
		mv "${tmp}/x/runtimes/osx-arm64/native/libonnxruntime.dylib" third_party/onnxruntime/lib/
		rm -rf "${tmp}"
	fi
elif [ "${OS}" = windows ]; then
	tmp="$(mktemp -d)"
	# ---- OBS 发布包里的 obs.dll ----
	if [ ! -f third_party/obs-win/obs.dll ]; then
		echo "下载 OBS ${OBS_VERSION} Windows 发布包（取 obs.dll）"
		curl -fsSL -o "${tmp}/obs.zip" \
			"https://github.com/obsproject/obs-studio/releases/download/${OBS_VERSION}/OBS-Studio-${OBS_VERSION}-Windows-x64.zip"
		unzip_members "${tmp}/obs.zip" "${tmp}/obs" bin/64bit/obs.dll
		mkdir -p third_party/obs-win
		mv "${tmp}/obs/bin/64bit/obs.dll" third_party/obs-win/
	fi
	# ---- ONNX Runtime DirectML 版 + 头文件 ----
	if [ ! -f third_party/ort-win/dml/onnxruntime.dll ]; then
		echo "下载 onnxruntime ${ORT_DML_VERSION}（DirectML 版）和 DirectML ${DIRECTML_VERSION}"
		nuget microsoft.ml.onnxruntime.directml "${ORT_DML_VERSION}" "${tmp}/dml.nupkg"
		unzip_members "${tmp}/dml.nupkg" "${tmp}/dml" build/native/include/ runtimes/win-x64/native/
		nuget microsoft.ai.directml "${DIRECTML_VERSION}" "${tmp}/directml.nupkg"
		unzip_members "${tmp}/directml.nupkg" "${tmp}/directml" bin/x64-win/DirectML.dll
		rm -rf third_party/onnxruntime/include third_party/ort-win/dml
		mkdir -p third_party/onnxruntime third_party/ort-win/dml
		mv "${tmp}/dml/build/native/include" third_party/onnxruntime/include
		cp "${tmp}/dml/runtimes/win-x64/native/onnxruntime.dll" third_party/ort-win/dml/
		cp "${tmp}/directml/bin/x64-win/DirectML.dll" third_party/ort-win/dml/
	fi
	# ---- ONNX Runtime CUDA 版（可选）----
	if [ "${SKIP_CUDA:-0}" != 1 ] && [ ! -f third_party/ort-win/cuda/onnxruntime.dll ]; then
		echo "下载 onnxruntime ${ORT_CUDA_VERSION}（CUDA 版）"
		nuget microsoft.ml.onnxruntime.gpu.windows "${ORT_CUDA_VERSION}" "${tmp}/cuda.nupkg"
		unzip_members "${tmp}/cuda.nupkg" "${tmp}/cuda" runtimes/win-x64/native/
		mkdir -p third_party/ort-win/cuda
		# 只要 onnxruntime 本体、共享层和 CUDA 方案；TensorRT 方案用不上，体积大，不要
		for f in onnxruntime.dll onnxruntime_providers_shared.dll onnxruntime_providers_cuda.dll; do
			cp "${tmp}/cuda/runtimes/win-x64/native/${f}" third_party/ort-win/cuda/
		done
		# 随包的 CUDA / cuDNN 运行库：用户不用再装 CUDA Toolkit 和 cuDNN（显卡驱动仍需自己装）
		echo "下载 NVIDIA 运行库：${NVIDIA_WHEELS}"
		for spec in ${NVIDIA_WHEELS}; do
			pkg="${spec%%==*}"
			ver="${spec##*==}"
			pypi_wheel "${pkg}" "${ver}" "${tmp}/${pkg}.whl"
			extract_wheel_dlls "${tmp}/${pkg}.whl" third_party/ort-win/cuda "${pkg}"
			rm -f "${tmp}/${pkg}.whl"
		done
	fi
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
for m in ch_PP-OCRv4_det_infer.onnx ch_PP-OCRv4_rec_infer.onnx; do
	fetch "data/ocr/${m}" "https://huggingface.co/SWHL/RapidOCR/resolve/main/PP-OCRv4/${m}"
done
# 血腥暴力检测：CLIP ViT-B/32 图像模型（int8，MIT）；文字向量 data/scene/clip_prompts.bin 已在仓库里
fetch data/scene/clip_vision_int8.onnx \
	https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main/onnx/vision_model_int8.onnx

# NudeNet 的 release 文件直链会跳转到 GitHub 登录页，需要用已登录的 gh 下载（GitHub Actions 里用 GH_TOKEN）
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
for f in data/face_detection_yunet_2023mar.onnx data/seg/pphumanseg_2023mar.onnx data/ocr/*.onnx data/nudenet/*.onnx data/scene/*.onnx; do
	if grep -qi "<html\|<!doctype" < <(head -c 64 "${f}"); then
		echo "${f} 下载到的是网页而不是模型，请删掉后重试" >&2
		exit 1
	fi
done
echo "依赖已就绪"
