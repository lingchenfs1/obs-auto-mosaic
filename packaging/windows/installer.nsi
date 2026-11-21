; obs-auto-mosaic Windows 安装程序（NSIS 3，由 GitHub Actions 构建）
; makensis 参数（都用绝对路径）：
;   /DVERSION=<版本>  /DPKG=<build\package>  /DPKG_CUDA=<build\package-cuda>
;   /DREADME=<README-Windows.txt>  /DOUTFILE=<输出 exe>
Unicode true
ManifestDPIAware true
RequestExecutionLevel admin
; 不用 solid 压缩：CUDA 运行库很大，solid 要把整块数据放进内存
SetCompressor lzma

!include "MUI2.nsh"
!include "x64.nsh"
!include "LogicLib.nsh"
!include "Sections.nsh"
!include "FileFunc.nsh"

!ifndef VERSION
  !define VERSION "dev"
!endif
!define UNINST_KEY "Software\Microsoft\Windows\CurrentVersion\Uninstall\obs-auto-mosaic"

Name "obs-auto-mosaic（OBS 自动马赛克插件）${VERSION}"
OutFile "${OUTFILE}"
BrandingText "obs-auto-mosaic ${VERSION}"
ShowInstDetails show
ShowUninstDetails show

!define MUI_ABORTWARNING
!define MUI_COMPONENTSPAGE_SMALLDESC
!define MUI_WELCOMEPAGE_TEXT "本向导将把 OBS 自动马赛克插件安装到 OBS 的插件目录：$\r$\n$\r$\n  C:\ProgramData\obs-studio\plugins\obs-face-mosaic$\r$\n$\r$\n安装前请先完全退出 OBS。$\r$\n$\r$\n需要 OBS Studio 33 或更新版本（64 位）。"
!insertmacro MUI_PAGE_WELCOME
!define MUI_PAGE_CUSTOMFUNCTION_LEAVE CheckObsNotRunning
!insertmacro MUI_PAGE_COMPONENTS
!insertmacro MUI_PAGE_INSTFILES
!define MUI_FINISHPAGE_TEXT "安装完成。$\r$\n$\r$\n打开 OBS，在来源或场景上右键 →「滤镜」→「效果滤镜」→「+」→「自动马赛克」。$\r$\n$\r$\n配合抖音直播伴侣：在 OBS 里点「启动虚拟摄像机」，再在直播伴侣里添加「摄像头」素材，选择「OBS Virtual Camera」。详见使用说明。"
!define MUI_FINISHPAGE_SHOWREADME "$INSTDIR\README-Windows.txt"
!define MUI_FINISHPAGE_SHOWREADME_TEXT "查看使用说明（含抖音直播伴侣设置）"
!insertmacro MUI_PAGE_FINISH

!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES

!insertmacro MUI_LANGUAGE "SimpChinese"

; ---------------- 安装 ----------------

Section "主程序（必选）" SecMain
  SectionIn RO
  ; 覆盖安装：先删掉旧版本的程序文件和模型（ort-cuda 由 CUDA 组件决定是否保留）
  Delete "$INSTDIR\bin\64bit\*.dll"
  RMDir /r "$INSTDIR\bin\64bit\ort-dml"
  RMDir /r "$INSTDIR\data"

  SetOutPath "$INSTDIR"
  File /r "${PKG}\obs-face-mosaic\*.*"
  File "/oname=README-Windows.txt" "${README}"

  WriteUninstaller "$INSTDIR\uninstall.exe"
  SetRegView 64
  WriteRegStr HKLM "${UNINST_KEY}" "DisplayName" "obs-auto-mosaic（OBS 自动马赛克插件）"
  WriteRegStr HKLM "${UNINST_KEY}" "DisplayVersion" "${VERSION}"
  WriteRegStr HKLM "${UNINST_KEY}" "Publisher" "obs-auto-mosaic"
  WriteRegStr HKLM "${UNINST_KEY}" "URLInfoAbout" "https://github.com/lingchenfs1/obs-auto-mosaic"
  WriteRegStr HKLM "${UNINST_KEY}" "InstallLocation" "$INSTDIR"
  WriteRegStr HKLM "${UNINST_KEY}" "UninstallString" '"$INSTDIR\uninstall.exe"'
  WriteRegStr HKLM "${UNINST_KEY}" "QuietUninstallString" '"$INSTDIR\uninstall.exe" /S'
  WriteRegDWORD HKLM "${UNINST_KEY}" "NoModify" 1
  WriteRegDWORD HKLM "${UNINST_KEY}" "NoRepair" 1
SectionEnd

Section "英伟达 CUDA 加速（约 1.7 GB）" SecCuda
  RMDir /r "$INSTDIR\bin\64bit\ort-cuda"
  SetOutPath "$INSTDIR\bin\64bit\ort-cuda"
  File /r "${PKG_CUDA}\obs-face-mosaic\bin\64bit\ort-cuda\*.*"
SectionEnd

; 隐藏步骤：没选 CUDA 时删掉以前装过的 CUDA 组件；最后记录占用空间
Section "-收尾"
  ${IfNot} ${SectionIsSelected} ${SecCuda}
    RMDir /r "$INSTDIR\bin\64bit\ort-cuda"
  ${EndIf}
  ${GetSize} "$INSTDIR" "/S=0K" $0 $1 $2
  SetRegView 64
  WriteRegDWORD HKLM "${UNINST_KEY}" "EstimatedSize" $0
SectionEnd

!insertmacro MUI_FUNCTION_DESCRIPTION_BEGIN
  !insertmacro MUI_DESCRIPTION_TEXT ${SecMain} "插件本体、DirectML 显卡加速（英伟达 / AMD / Intel 通用）和全部检测模型。"
  !insertmacro MUI_DESCRIPTION_TEXT ${SecCuda} "英伟达显卡专用，比 DirectML 更快。已自带 CUDA 13 和 cuDNN 9，不用另装；需要 RTX 20 系列或更新的显卡、580 系列或更新的驱动。条件不满足时插件会自动改用 DirectML。"
!insertmacro MUI_FUNCTION_DESCRIPTION_END

Function .onInit
  ${IfNot} ${RunningX64}
    MessageBox MB_OK|MB_ICONSTOP "本插件只支持 64 位 Windows。"
    Abort
  ${EndIf}
  ; 全体用户上下文下 $APPDATA 就是 C:\ProgramData（OBS 第三方插件目录在它下面）
  SetShellVarContext all
  StrCpy $INSTDIR "$APPDATA\obs-studio\plugins\obs-face-mosaic"

  ; 检查是否装了 OBS（官方安装程序会写这个注册表项）
  SetRegView 64
  ReadRegStr $0 HKLM "SOFTWARE\OBS Studio" ""
  ${If} $0 == ""
    MessageBox MB_YESNO|MB_ICONQUESTION "没有检测到 OBS Studio。$\r$\n本插件需要 OBS Studio 33 或更新版本（64 位）。$\r$\n$\r$\n仍要继续安装吗？" IDYES +2
    Abort
  ${EndIf}

  ; 有英伟达显卡驱动（System32 下有 nvcuda.dll）时默认勾选 CUDA 组件
  ${DisableX64FSRedirection}
  ${If} ${FileExists} "$SYSDIR\nvcuda.dll"
    !insertmacro SelectSection ${SecCuda}
  ${Else}
    !insertmacro UnselectSection ${SecCuda}
  ${EndIf}
  ${EnableX64FSRedirection}
FunctionEnd

; OBS 正在运行时不让继续（文件被占用会装不进去）
Function CheckObsNotRunning
  check:
  nsExec::ExecToStack 'cmd /c tasklist /FI "IMAGENAME eq obs64.exe" /NH | find /I "obs64.exe"'
  Pop $0
  Pop $1
  ${If} $0 == 0
    MessageBox MB_RETRYCANCEL|MB_ICONEXCLAMATION "OBS 正在运行。请先完全退出 OBS（包括系统托盘里的图标），然后点「重试」。" IDRETRY check
    Abort
  ${EndIf}
FunctionEnd

; ---------------- 卸载 ----------------

Function un.onInit
  SetShellVarContext all
  check:
  nsExec::ExecToStack 'cmd /c tasklist /FI "IMAGENAME eq obs64.exe" /NH | find /I "obs64.exe"'
  Pop $0
  Pop $1
  ${If} $0 == 0
    MessageBox MB_RETRYCANCEL|MB_ICONEXCLAMATION "OBS 正在运行。请先完全退出 OBS，然后点「重试」。" IDRETRY check
    Abort
  ${EndIf}
FunctionEnd

Section "Uninstall"
  ; 只在确实是本插件目录时整个删除，防止卸载程序被挪到别处后误删
  ${If} ${FileExists} "$INSTDIR\bin\64bit\obs-face-mosaic.dll"
    RMDir /r "$INSTDIR"
  ${Else}
    Delete "$INSTDIR\uninstall.exe"
    RMDir "$INSTDIR"
  ${EndIf}
  SetRegView 64
  DeleteRegKey HKLM "${UNINST_KEY}"
SectionEnd
