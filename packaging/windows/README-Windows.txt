obs-auto-mosaic Windows 版（OBS 插件）
=====================================

自动给 OBS 画面里的不雅内容、裸露皮肤、敏感文字、人脸打码（马赛克或 100% 黑色色块）。
项目主页：https://github.com/lingchenfs1/obs-auto-mosaic

一、系统要求
  - Windows 10 / 11 64 位
  - OBS Studio 33.x（64 位）
  - 支持 DirectX 12 的显卡（英伟达、AMD、Intel 都可以）。没有可用的显卡时自动用 CPU。

二、安装
  推荐：一键安装程序 obs-auto-mosaic-windows-x64-installer.exe
  1. 先完全退出 OBS（包括系统托盘里的图标）。
  2. 双击运行安装程序。程序没有数字签名，Windows 可能提示「Windows 已保护你的电脑」：
     点「更多信息」→「仍要运行」。
  3. 按向导点「下一步」。「英伟达 CUDA 加速」组件在检测到英伟达显卡驱动时默认勾选，
     不需要可以取消（不装也能用 DirectML 显卡加速）。
  4. 卸载：「设置」→「应用」里找到「obs-auto-mosaic（OBS 自动马赛克插件）」卸载。

  也可以用压缩包手动安装（obs-auto-mosaic-windows-x64.zip）：
     a) 右键「开始」→「终端（管理员）」或「Windows PowerShell（管理员）」，进入解压目录，运行：
          powershell -ExecutionPolicy Bypass -File .\install.ps1
     b) 或把 obs-face-mosaic 文件夹整个复制到 C:\ProgramData\obs-studio\plugins\
        （复制后的路径应是 C:\ProgramData\obs-studio\plugins\obs-face-mosaic\bin\64bit\obs-face-mosaic.dll）

  装好后打开 OBS，在要处理的来源（或场景）上右键 →「滤镜」→「效果滤镜」→「+」→「自动马赛克」。

三、英伟达显卡加速（可选，更快）
  默认用 DirectML（显卡通用，不用额外安装）。想用 CUDA（RTX 20 系列或更新的英伟达显卡）：
  1. 把显卡驱动更新到 580 系列或更新。
  2. 用一键安装程序时勾选「英伟达 CUDA 加速」组件即可（已自带 CUDA 13 和 cuDNN 9，不用另装）；
     手动安装时下载 obs-auto-mosaic-windows-cuda-addon.zip（约 1.2GB），
     把里面的 obs-face-mosaic 文件夹复制到同一位置（和原来的合并）。
  插件启动时先用 CUDA 实际跑一次模型，跑通才用 CUDA，否则自动换回 DirectML。

  确认实际用的是哪种：OBS 菜单「帮助」→「日志文件」→「查看当前日志」，搜索 [face-mosaic]：
    「ONNX Runtime：使用 CUDA 运行库」或「使用 DirectML 运行库」
    各通道「模型已加载（…，CUDA / DirectML / CPU）」
  滤镜设置里的「推理设备」选「仅 CPU」可以关掉显卡加速。

四、配合抖音直播伴侣使用（OBS 虚拟摄像头）
  直播伴侣不能装 OBS 插件，做法是：在 OBS 里打码，再用 OBS 的虚拟摄像头把画面交给直播伴侣。
  1. 在 OBS 里搭好要直播的场景（摄像头、游戏、窗口采集等），给来源或场景加上「自动马赛克」滤镜。
  2. 「设置」→「视频」：基础分辨率、输出分辨率设成直播用的分辨率（如 1920x1080），帧率 30 或 60。
  3. 点 OBS 主界面右下角「启动虚拟摄像机」。
  4. 打开抖音直播伴侣 → 添加素材 →「摄像头」→ 设备选「OBS Virtual Camera」，分辨率选和 OBS 一致的，
     调整到铺满画面。
  5. 声音：虚拟摄像头只传画面，不传声音。麦克风、电脑声音继续由直播伴侣采集。
     滤镜的「画面延迟」（默认 150 毫秒）加上虚拟摄像头本身的延迟，会让画面比声音晚一点：
       - 如果直播伴侣的音频设置里有「延迟」「同步偏移」之类的选项，设成相同的毫秒数；
       - 如果没有，可以把滤镜的「画面延迟」调小（如 100 毫秒），代价是快速出现的内容更容易漏出一两帧；
       - 或者用虚拟声卡（如 VB-Audio Virtual Cable）把声音也交给 OBS 统一延迟后再给直播伴侣。
     开播前先录一段，确认口型对得上。
  注意：直播伴侣里不要再直接添加「游戏」「窗口」「摄像头原始设备」等会绕过 OBS 的素材，
        否则这些画面不会被打码。

五、卸载
  用安装程序装的：「设置」→「应用」里卸载「obs-auto-mosaic（OBS 自动马赛克插件）」。
  手动装的：退出 OBS，删除 C:\ProgramData\obs-studio\plugins\obs-face-mosaic 文件夹。

六、已知情况
  - Windows 版由 GitHub Actions 自动构建，作者没有在 Windows 上实测过；遇到问题请附上 OBS 日志里
    [face-mosaic] 开头的行。
  - 真正露骨的类别没有用露骨素材验证过，开播前请用自己的素材测试。
