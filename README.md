# obs-auto-mosaic：OBS 自动马赛克滤镜

给 OBS 里任意视频源加一个「自动马赛克」滤镜：检测到不雅部位、大面积裸露皮肤、敏感文字或人脸，就自动打码（马赛克，或 100% 不透明的黑色色块）。支持 macOS（Apple Silicon）和 Windows 10/11 x64，OBS 33.x；macOS 版在 OBS 33.0.1 上实测，Windows 版由 GitHub Actions 构建、**尚未在 Windows 上实测**。可通过 OBS 虚拟摄像头配合抖音直播伴侣使用。

四个检测通道，各自在独立线程里推理，可分别开关：

| 通道 | 模型 | 作用 |
|---|---|---|
| 不雅内容 | [NudeNet](https://github.com/notAI-tech/NudeNet) v3（`320n.onnx` 快速 / `640m.onnx` 精确） | 识别暴露的胸部、生殖器、臀部等 18 类部位 |
| 裸露皮肤 | [PP-HumanSeg](https://github.com/opencv/opencv_zoo/tree/main/models/human_segmentation_pphumanseg) 人体分割 + 肤色 | 补 NudeNet 的漏检：健美、泳装、光膀子、背影 |
| 敏感文字 | [PP-OCRv4](https://github.com/PaddlePaddle/PaddleOCR)（[RapidOCR](https://github.com/RapidAI/RapidOCR) 导出的 ONNX） | 手机号、身份证号、银行卡号、邮箱、网址、微信 / QQ 号，以及自定义关键词 |
| 人脸 | [YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) | 给路人等人脸打码 |

推理用 [ONNX Runtime](https://onnxruntime.ai/)，可用 GPU 加速（macOS：CoreML；Windows：CUDA 或 DirectML），打码用 GPU 着色器。

## 主要特性

- **延迟对齐**：滤镜把画面延迟一小段时间（默认 150 ms）再输出，让每一帧都能用上「之后」的检测结果。人或部位刚出现、快速移动时也不会漏出（测试数据见下文）。**开启后画面会比声音晚同样的时间**，需要在 OBS「高级音频属性」里给麦克风等音频源设置相同的同步偏移。
- **兜底**：任一启用的通道超过 1 秒没有新结果（模型加载中、推理卡住、源刚显示），整个画面打码。
- **可选整屏打码**：出现暴露的胸部、生殖器、臀部时，可以选择直接给整个画面打码。
- **遮挡方式**：「马赛克」或「黑色色块（100% 不透明）」。黑色色块直接覆盖、不与原画面混合，兜底时的整屏遮挡也会变成全黑；在 OBS 里逐像素验证过，被遮挡的像素全部是 (0,0,0,255)。

## GPU 加速

滤镜设置「推理设备」：「自动（有 GPU 就用 GPU）」或「仅 CPU」，默认自动。实际用的方案写在 OBS 日志的 `[face-mosaic] … 模型已加载（…，CoreML / CUDA / DirectML / CPU）` 里。

| 平台 | 方案 | 说明 |
|---|---|---|
| macOS | CoreML（CPU + GPU） | 首次加载每个模型要编译 0.5~2 秒，期间兜底整屏打码 |
| Windows | CUDA（可选扩展包） | 英伟达显卡专用，需要装好对应版本的 CUDA 和 cuDNN；加不上会自动换 DirectML |
| Windows | DirectML（默认） | 任何支持 DirectX 12 的显卡（英伟达 / AMD / Intel），不用额外安装 |

Windows 上插件按完整路径加载自带的 `onnxruntime.dll`（`bin/64bit/ort-cuda/` 或 `ort-dml/`），不会误加载 Windows 11 系统目录里自带的旧版。DirectML 版 ONNX Runtime 停在 1.24.4，CUDA 版用 1.32.0，Windows 版统一按 1.24 的接口编译（新版运行库兼容旧接口）。

macOS 上的实测（M4，1080p 60fps 测试视频，四个通道全开，每个模型 1 个线程）：

| 推理设备 | OBS 进程 CPU | 人脸 | 裸露皮肤 | 不雅内容 | 敏感文字 |
|---|---|---|---|---|---|
| 仅 CPU | 176% | 12.9 ms | 26.4 ms | 31.9 ms | 81.8 ms |
| CoreML，CPU + GPU（默认） | 123% | 6.5 ms | 14.6 ms | 24.8 ms | 82.2 ms |
| CoreML，交给系统分配（ALL） | 117% | 23.6 ms | 28.7 ms | 28.2 ms | 96.6 ms |
| CoreML，CPU + 神经引擎 | 142% | 12.4 ms | 18.7 ms | 26.5 ms | 75.1 ms |

（同一台机器、相近负载下各测一次，数值有波动；CoreML 下的延迟对齐复测同样无漏出。）

## 配合抖音直播伴侣（OBS 虚拟摄像头）

直播伴侣不支持 OBS 插件，做法是在 OBS 里打码，再用 OBS 自带的虚拟摄像头把画面交给直播伴侣：

1. OBS 里搭好场景，给来源或场景加上「自动马赛克」滤镜；「设置 → 视频」的分辨率设成直播用的分辨率。
2. 点 OBS 右下角「启动虚拟摄像机」。
3. 直播伴侣 → 添加素材 →「摄像头」→ 选「OBS Virtual Camera」，铺满画面。
4. 虚拟摄像头只传画面不传声音：滤镜的画面延迟（默认 150 ms）会让画面比直播伴侣采集的声音晚，需要在直播伴侣里给音频设相同的延迟（如果有这个选项），或调小画面延迟，或用虚拟声卡把声音也经 OBS 延迟后再给直播伴侣。开播前先录一段确认口型。
5. 不要在直播伴侣里再直接添加游戏、窗口、原始摄像头等素材，那些画面不经过 OBS，不会被打码。

（以上步骤未在 Windows 上实测。）

## 不雅内容的类别分组

| 设置项 | 包含的 NudeNet 类别 | 默认 |
|---|---|---|
| 暴露的胸部、生殖器、臀部、肛门 | FEMALE_BREAST / FEMALE_GENITALIA / MALE_GENITALIA / BUTTOCKS / ANUS _EXPOSED | 开 |
| 内衣 / 泳装遮盖的胸部、生殖器、臀部 | FEMALE_BREAST / FEMALE_GENITALIA / BUTTOCKS / ANUS _COVERED | 关（普通衣服也会触发；泳装由「裸露皮肤」通道覆盖） |
| 男性赤裸上身、裸露的腹部、腋下 | MALE_BREAST / BELLY / ARMPITS _EXPOSED | 开 |
| 裸露的脚 | FEET_EXPOSED | 关 |

不雅检测灵敏度默认 0.2（宁可多遮）。

## 裸露皮肤通道的原理

1. PP-HumanSeg 把画面里的人分割出来（输入 192×192），按 160 宽的网格分析；
2. 肤色 = 通用 YCrCb 规则（133≤Cr≤173、77≤Cb≤127）∪ 按每张脸取样的自适应肤色（Cr/Cb 马氏距离 < 3，适应灯光和肤色深浅）；脸本身不计入；
3. 每个人（分割连通块）算「全身皮肤占比」和「脸下方躯干的皮肤占比」，全身 ≥ 40% 或躯干 ≥ 60% 判定为裸露（两个门槛都可调）；
4. 判定为裸露的人，身上每块皮肤（不含脸）都打码。

在测试图上的标定结果（`tools/skin-test`）：

| 图 | 全身 | 躯干 | 判定 |
|---|---|---|---|
| 健美（单人 / 合影） | 77% / 79% | 59% / 85% | 裸露 |
| 沙滩排球 4 人（含背影、泳装） | 50%~92% | 无脸或 21% | 全部裸露 |
| 穿衣服的人像 4 张 | 12%~29% | 0%~12% | 不算 |
| 红蓝条纹球衣 | 28% | 0% | 不算 |
| 密集人群 | 30% | 37% | 不算 |

## 敏感文字通道

- 单独采集一份高分辨率画面（长边 1920 以内），默认每秒 3 次；文字检测分辨率可选 640 / 800 / 960 / 1280（默认 800）。
- 检测出的文字行（只处理横排）逐行识别，同一位置、内容没变的行用缓存，不重复识别。
- 识别结果规整（去空格、全角转半角、转小写）后按规则匹配，命中的**整行**打码：

| 规则 | 匹配方式 |
|---|---|
| 手机号 | `1[3-9]` 开头的 11 位数字（前后不是数字） |
| 身份证号 | 18 位，含出生日期格式校验，末位可为 X |
| 银行卡号 | 16~19 位连续数字（去空格后） |
| 邮箱、网址 | 常见格式；网址含 `http(s)://`、`www.` 和常见后缀 |
| 微信 / QQ 号 | 「微信 / vx / wx / QQ / 扣扣」等后面紧跟 5 位以上的账号 |
| 自定义关键词 | 设置里每行一个（或逗号分隔），包含即命中 |

- **识别出的文字不会写进 OBS 日志**（日志常被上传分享），日志里只有行数和耗时。
- 测试画面（`tools/make-text-test.py` 生成，1920×1080，18~34 号字）：8 行敏感信息全部命中，4 行普通文字（含「游戏时长 02:35:17 分数 98765」）没有误伤。M4 上每次约 80 ms（有缓存时；首次约 270 ms）。

## 已知局限

- 只用非露骨的公开照片（健美、沙滩排球、穿衣服的人像）测试过；**真正露骨的类别没有验证**，请用自己的素材测试后再用于直播。
- NudeNet 主要按色情内容训练，对健美、背影识别差，这部分靠「裸露皮肤」通道补；远景里很小的人（约 20 像素高）两个通道都检不出。
- 肤色和皮肤相近的衣服（米色、肉色）会抬高皮肤占比；露背装、短裤短裙等介于两者之间的情况没测过，可能需要按实际画面调门槛。
- 文字通道每秒只检测几次，新出现的文字可能要几百毫秒才被遮住；竖排、倾斜、艺术字、很小的字（低于约 12 像素）可能认不出；号码被拆成多行或中间夹了其他字符时规则匹配不到。
- 人脸打码默认开启：如果是自己出镜直播，记得关掉，否则自己的脸也会被打码。
- Windows 版没有在 Windows 上实测过；macOS 只支持 Apple Silicon。

## 安装 Windows 版

从 [Releases](https://github.com/lingchenfs1/obs-auto-mosaic/releases) 或 GitHub Actions 的构建产物下载：

- `obs-auto-mosaic-windows-x64.zip`：主包（含 DirectML 运行库和全部模型）；
- `obs-auto-mosaic-windows-cuda-addon.zip`：可选的英伟达 CUDA 扩展包。

解压后按里面的 `README-Windows.txt` 安装：把 `obs-face-mosaic` 文件夹复制到 `C:\ProgramData\obs-studio\plugins\`（或以管理员身份运行 `install.ps1`）。

## 构建 Windows 版

由 `.github/workflows/build.yml` 在 GitHub Actions 的 `windows-2022` 上完成：`tools/fetch-deps.sh`（Git Bash）下载 OBS 发布包里的 `obs.dll`、两套 ONNX Runtime 和 DirectML.dll，`tools/make-import-lib.py` 用 `dumpbin` / `lib` 从 `obs.dll` 生成导入库，再用 Visual Studio 2022 构建并打包。推 `v*` 标签时自动建 Release。

## 构建（macOS，Apple Silicon）

需要：Xcode 命令行工具、CMake、Ninja、Homebrew、已登录的 [gh](https://cli.github.com/)（下载 NudeNet 模型用），以及安装在 `/Applications/OBS.app` 的 OBS。

```bash
brew install cmake ninja simde
```

```bash
tools/fetch-deps.sh
```

```bash
cmake -S . -B build -G Ninja
```

```bash
cmake --build build
```

```bash
cmake --build build --target install-plugin
```

- `fetch-deps.sh` 下载 libobs 头文件（默认 33.0.1，要和安装的 OBS 版本一致）、ONNX Runtime（NuGet 官方包）和全部模型，都不进 git。
- `install-plugin` 把 `obs-face-mosaic.plugin` 复制到 `~/Library/Application Support/obs-studio/plugins/`。重启 OBS 后，在源的「滤镜 → 效果滤镜」里添加「自动马赛克」。
- 链接方式：头文件来自源码，库直接链接 `OBS.app` 里的 `libobs.framework`；onnxruntime 动态库打进 bundle 的 `Contents/Frameworks`。OBS 带 `disable-library-validation` 权限，插件只需 ad-hoc 签名。

## 性能

在 Apple M4 上（每个模型 1 个推理线程，每秒各检测约 13 次，1080p 60fps 测试视频）：

| 通道 | 每次检测 |
|---|---|
| 人脸 | 约 13 ms |
| 裸露皮肤 | 约 27 ms |
| 不雅内容（320n） | 约 33 ms |
| 敏感文字（每秒 3 次） | 约 80 ms |

前三个通道全开时 OBS 进程约占 1.3 个核，加上文字通道约 1.35 个核（含视频解码等 OBS 自身开销）。ONNX Runtime 默认的「2 线程 + 自旋等待」会让 OBS 涨到 2.3~2.7 个核，所以改成了每个模型 1 个线程。调低「每秒检测次数」可以进一步降低占用。

精确模型 `640m` 每次 100~200 ms，需要把画面延迟调到 300 ms 以上。

## 测试

```bash
tools/fetch-test-assets.sh
```

下载测试图片（真人照片，不放进仓库；来源和许可见 `test/*/SOURCES.md`）并生成延迟对齐测试视频。离线工具：

```bash
./build/nude-test data/nudenet/320n.onnx 640 test/nsfw-benign/volleyball1.jpg out.png 0.2
```

```bash
./build/skin-test data/seg/pphumanseg_2023mar.onnx data/face_detection_yunet_2023mar.onnx test/nsfw-benign/lineup.jpg out.png
```

```bash
./build/detect-test data/face_detection_yunet_2023mar.onnx test/group.jpg out.png
```

```bash
./build/text-test data/ocr/ch_PP-OCRv4_det_infer.onnx data/ocr/ch_PP-OCRv4_rec_infer.onnx test/text-sample.png out.png "测试敏感词"
```

在 OBS 里自动实测（会临时切换场景集合，并在结束后恢复 OBS 的 `user.ini`、`global.ini`）：

```bash
DUMP_COUNT=60 DUMP_EVERY=1 tools/run-obs-test.sh /tmp/obs-run 150
```

```bash
python3 -I tools/check-dumps.py "$PWD/build/detect-test" "$PWD/data/face_detection_yunet_2023mar.onnx" /tmp/obs-run/dump 60
```

`run-obs-test.sh` 默认播放 `test/moving-faces.mp4`；环境变量 `MEDIA` 可换成别的视频或图片，后面的 `键=值` 参数会写进滤镜设置（如 `face_enabled=false nude_model=1`）。

调试导出：启动 OBS 时设置环境变量 `FACE_MOSAIC_DUMP=<路径前缀>`，滤镜会导出 `<前缀>-<n>.png`（输出画面）和 `<前缀>-<n>-raw.png`（同一帧未打码的画面）；`FACE_MOSAIC_DUMP_COUNT`、`FACE_MOSAIC_DUMP_EVERY` 控制导出几组、间隔几帧。`check-dumps.py` 在原帧上检测人脸作为真值，统计每张脸框内没被改动的像素比例；被完整打码的脸也会有约 17% 的像素差值很小，所以 17%~20% 算作已遮住。

延迟对齐的实测（1920×1080、60fps，一张脸以每秒 1500 像素来回移动，另一张每 0.5 秒出现或消失一次）：

| 延迟 | 结果 |
|---|---|
| 0 ms | 刚出现的脸完全露出，移动的脸露出 26%~54%（码落在后面） |
| 150 ms | 连续 60 帧、90 张脸全部 ≤ 19.6%（底噪水平），包括刚出现那一帧和贴边掉头 |

## 待办

- 延迟对齐时自动同步音频（现在需要手动设同步偏移）
- Windows 实测（DirectML、CUDA、抖音直播伴侣虚拟摄像头）；摄像头实测
- 远处小人、小脸：分块检测
- 其他敏感内容：窗口区域遮挡、二维码；文字只遮命中的那几个字而不是整行

## 许可证

本项目代码以 [AGPL-3.0](LICENSE) 发布（NudeNet 为 AGPL-3.0，OBS 为 GPL-2.0-or-later）。第三方组件和模型各自的许可：

| 组件 | 许可 |
|---|---|
| NudeNet 模型 | AGPL-3.0 |
| YuNet、PP-HumanSeg（opencv_zoo） | MIT、Apache-2.0 |
| PP-OCRv4（PaddleOCR / RapidOCR） | Apache-2.0 |
| ONNX Runtime、DirectML | MIT |
| stb_image / stb_image_write（`third_party/stb`） | MIT 或 Public Domain |
| libobs 头文件 | GPL-2.0-or-later |

模型和 OBS、ONNX Runtime 都由 `tools/fetch-deps.sh` 从原始出处下载，不包含在本仓库中。
