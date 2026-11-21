#!/usr/bin/env bash
# 在本机 OBS 里跑一次测试：写测试场景集合 → 启动 OBS → 导出调试帧、测 CPU → 退出 → 恢复设置文件。
# 用法：tools/run-obs-test.sh <输出目录> <delay_ms> [--filter-off] [滤镜设置 键=值 ...]
# 环境变量 MEDIA 指定视频或图片（默认 test/moving-faces.mp4）
set -uo pipefail

OUT_DIR="$1"
DELAY="$2"
shift 2
EXTRA=("$@")
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OBS_CFG="$HOME/Library/Application Support/obs-studio"
DUMP_COUNT="${DUMP_COUNT:-12}"
DUMP_EVERY="${DUMP_EVERY:-5}"

mkdir -p "${OUT_DIR}"
if pgrep -x OBS >/dev/null; then
	echo "OBS 正在运行，先退出再测" >&2
	exit 1
fi

cp "${OBS_CFG}/user.ini" "${OUT_DIR}/user.ini.bak"
cp "${OBS_CFG}/global.ini" "${OUT_DIR}/global.ini.bak"
restore() {
	cp "${OUT_DIR}/user.ini.bak" "${OBS_CFG}/user.ini"
	cp "${OUT_DIR}/global.ini.bak" "${OBS_CFG}/global.ini"
}
trap restore EXIT

python3 "${ROOT}/tools/make-test-collection.py" "${OBS_CFG}/basic/scenes/obs-ai-test.json" \
	"${MEDIA:-${ROOT}/test/moving-faces.mp4}" "${DELAY}" "${EXTRA[@]}"

open -n -a /Applications/OBS.app \
	--env "FACE_MOSAIC_DUMP=${OUT_DIR}/dump" \
	--env "FACE_MOSAIC_DUMP_COUNT=${DUMP_COUNT}" \
	--env "FACE_MOSAIC_DUMP_EVERY=${DUMP_EVERY}" \
	--env "FACE_MOSAIC_ORT_THREADS=${FACE_MOSAIC_ORT_THREADS:-}" \
	--env "FACE_MOSAIC_ORT_SPIN=${FACE_MOSAIC_ORT_SPIN:-}" \
	--env "FACE_MOSAIC_COREML_UNITS=${FACE_MOSAIC_COREML_UNITS:-}" \
	--args --collection obs-ai-test --disable-shutdown-check --disable-updater

pid=""
for _ in $(seq 1 20); do
	pid="$(pgrep -x OBS || true)"
	[ -n "${pid}" ] && break
	sleep 1
done
if [ -z "${pid}" ]; then
	echo "OBS 没有启动" >&2
	exit 1
fi

cpu_seconds() {
	# ps 的 time 格式为 分:秒.百分秒
	ps -o time= -p "$1" | awk -F'[:.]' '{print $1*60 + $2 + $3/100}'
}

sleep 10
c1="$(cpu_seconds "${pid}")"
sleep 10
c2="$(cpu_seconds "${pid}")"
awk -v a="${c1}" -v b="${c2}" 'BEGIN{printf "OBS 进程 CPU：%.1f%%（10 秒平均，100%% = 1 个核）\n", (b-a)*10}'

# 环境变量 EXTRA_WAIT：测完 CPU 后再多跑几秒（统计更多 10 秒时段）
sleep "${EXTRA_WAIT:-0}"
osascript -e 'quit app "OBS"' >/dev/null 2>&1 || true
for _ in $(seq 1 15); do
	pgrep -x OBS >/dev/null || break
	sleep 1
done
if pgrep -x OBS >/dev/null; then
	kill -TERM "${pid}" || true
	sleep 3
fi

log="$(ls -t "${OBS_CFG}/logs/"*.txt | head -1)"
grep "face-mosaic" "${log}" | grep -v "调试帧已导出" | tail -"${LOG_LINES:-8}"
