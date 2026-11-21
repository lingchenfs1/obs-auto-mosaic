obs-auto-mosaic 英伟达 CUDA 扩展包（可选）
==========================================

只适用于英伟达显卡。不装这个扩展包也能用显卡加速（DirectML），CUDA 一般更快一些。

需要先安装（版本必须对得上，否则插件会自动退回 DirectML）：
  - 支持 CUDA 13 的英伟达显卡驱动（较新版本）
  - CUDA Toolkit 13.x
  - cuDNN 9.x（选「for CUDA 13」的版本）
  安装后，CUDA 和 cuDNN 的 bin 目录需要在系统 PATH 里（CUDA 安装程序默认会加；cuDNN 按其说明添加）。

依赖（本扩展包里的 onnxruntime 1.32.0 CUDA 版）：
  - 直接依赖（构建时由 onnxruntime_providers_cuda.dll 的导入表读出）：__CUDA_DEPS__
  - 运行时按名字加载：cudart64_13.dll、cudnn64_9.dll、cufft、nvrtc 等

安装：退出 OBS，把本压缩包里的 obs-face-mosaic 文件夹复制到 C:\ProgramData\obs-studio\plugins\，
      和主包合并（会多出 bin\64bit\ort-cuda 文件夹）。
确认：OBS 日志里出现「ONNX Runtime：使用 CUDA 运行库（ort-cuda）」。
      如果是「CUDA 不可用（…），改用 DirectML」，括号里是原因，多半是 CUDA / cuDNN 版本或 PATH 不对。
