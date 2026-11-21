obs-auto-mosaic 英伟达 CUDA 扩展包（可选）
==========================================

只适用于英伟达显卡。不装这个扩展包也能用显卡加速（DirectML），CUDA 一般更快一些。

需要先安装（版本必须对得上，否则插件会自动退回 DirectML）：
  - 英伟达显卡驱动（较新版本）
  - CUDA Toolkit：版本见下方「依赖」
  - cuDNN：版本见下方「依赖」
  安装后，CUDA 和 cuDNN 的 bin 目录需要在系统 PATH 里（CUDA 安装程序默认会加；cuDNN 按其说明添加）。

依赖（构建时由 onnxruntime_providers_cuda.dll 的导入表确定，见 GitHub Actions 构建日志）：
  __CUDA_DEPS__

安装：退出 OBS，把本压缩包里的 obs-face-mosaic 文件夹复制到 C:\ProgramData\obs-studio\plugins\，
      和主包合并（会多出 bin\64bit\ort-cuda 文件夹）。
确认：OBS 日志里出现「ONNX Runtime：使用 CUDA 运行库（ort-cuda）」。
      如果是「CUDA 不可用（…），改用 DirectML」，括号里是原因，多半是 CUDA / cuDNN 版本或 PATH 不对。
