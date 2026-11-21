# 测试素材

测试图片都是真人照片，不放进仓库，运行 `tools/fetch-test-assets.sh` 下载：

- `group.jpg`、`faces.jpg`：opencv_zoo / OpenCV 的示例图（人脸）
- `nsfw-benign/`：非露骨的健美、沙滩排球照（来源和许可见该目录 `SOURCES.md`）
- `clothed/`：穿衣服的人像，作为裸露检测的对照（来源和许可见该目录 `SOURCES.md`）
- `moving-faces.mp4`：由 `tools/make-test-video.py` 用 `group.jpg` 生成，用于延迟对齐测试
- `text-sample.png`：由 `tools/make-text-test.py` 生成的敏感文字测试画面（手机号、身份证号等都是示例值）

不含任何露骨素材；暴露类别需要自行准备素材验证。
