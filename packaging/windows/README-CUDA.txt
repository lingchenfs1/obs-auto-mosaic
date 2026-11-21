obs-auto-mosaic 英伟达 CUDA 扩展包（可选）
==========================================

只适用于英伟达显卡。不装这个扩展包也能用显卡加速（DirectML），CUDA 一般更快一些。

本扩展包已经自带 CUDA 13 和 cuDNN 9 的运行库，不用再装 CUDA Toolkit 和 cuDNN。只需要：
  - RTX 20 系列或更新的英伟达显卡（CUDA 13 不再支持 GTX 10 系列及更早的显卡）
  - 支持 CUDA 13 的显卡驱动（580 系列或更新；在 NVIDIA 官网或 GeForce Experience / NVIDIA App 更新）
条件不满足时插件会自动改用 DirectML，不影响使用。

自带的运行库（来自 NVIDIA 在 PyPI 发布的官方包）：
  CUDA 运行时 13.5、cuBLAS 13.10、cuDNN 9.28（CUDA 13 版）、cuFFT 12.5、NVRTC 13.5
  onnxruntime_providers_cuda.dll 直接依赖：__CUDA_DEPS__
插件启动时按完整路径加载这些运行库，不需要配置 PATH，也不会和电脑上另装的 CUDA 冲突。

许可：这些 NVIDIA 运行库按 NVIDIA 的许可协议随本扩展包分发，许可证全文在
  obs-face-mosaic\bin\64bit\ort-cuda\licenses\ 下。它们不属于本项目的 AGPL-3.0 代码。

安装：退出 OBS，把本压缩包里的 obs-face-mosaic 文件夹复制到 C:\ProgramData\obs-studio\plugins\，
      和主包合并（会多出 bin\64bit\ort-cuda 文件夹）。
确认：OBS 日志里出现「ONNX Runtime：使用 CUDA 运行库（ort-cuda，预加载随包的 NVIDIA 运行库 N 个，实测推理通过）」。
      如果是「CUDA 不可用（…），改用 DirectML」，括号里是原因，多半是显卡太老或驱动太旧。
