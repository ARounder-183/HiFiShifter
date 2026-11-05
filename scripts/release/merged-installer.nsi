; HiFiShifter 合并安装器（NSIS）：独立 App + VST3/ARA 插件 + 共享模型库。
;
; 【为什么是自研脚本，而不是 Tauri 的 NSIS 模板或 installerHooks】
;
; - bundle.windows.nsis.installerHooks 只能往 Tauri 生成的脚本里**插代码**，
;   加不了组件页，也加不了"第二个安装目录"（VST3 目录）。模型仍然会被 Tauri 的
;   payload 打进 App 目录 —— 那正是要消除的第二份 150 MB。
; - 自定义 nsis.template 要把 700 行模板连同它的全部 !define 隐式契约一起抄下来，
;   然后跟着 tauri-cli 的版本漂移。模板里的每一处升级都会变成一次手工合并。
;
; 所以这里自己写一份：页面、组件、目录、模型段、卸载段都在眼前，构建脚本用 /D 把
; 每一个路径传进来（脚本里不出现任何相对路径猜测）。
;
; 【文案】不在这里：全部来自 $(HFS_*)，取值由
; scripts/release/build-installer-strings.mjs 从五语词表抽出。要改文案请改词表。
;
; 【必须原样保留的三段（都来自旧插件安装器，且是踩过的坑）】
;   1. REAPER 运行守卫：安装与卸载都查（加载中的 DLL 被覆盖 = 宿主崩溃）；
;   2. 目录对账：Vite 的 chunk 带内容哈希，File /r 只写不删，重装会单调增长；
;   3. ARP 注册：没有它，Windows「已安装的应用」里根本没有本产品。
;
; 【从 Tauri 模板移植的三段】
;   WebView2 运行时检测 / 安装（内嵌 bootstrapper）、主程序运行中检测、快捷方式。

Unicode true
!include "MUI2.nsh"
!include "LogicLib.nsh"
!include "nsDialogs.nsh"
!include "x64.nsh"
!include "Sections.nsh"
!include "StrFunc.nsh"
; 安装器需要 StrStr（进程名过滤）与 StrTrimNewLines（读回文件大小时去掉行尾）。
; 卸载器只需要 StrStr（卸载器里的字符串函数前缀是 un.）。
${StrStr}
${UnStrStr}
${StrTrimNewLines}

; ── 构建期参数（全部由 scripts/build/build.ps1 以 /D 传入）──────────────────
;
; 【为什么全部强制】缺一个参数时 File /r 只会报一句 "no files found"，那是最难定位
; 的一类构建错误。!error 让它在编译的第一秒就停下并说清楚缺了什么。
!ifndef APP_DIR
  !error "APP_DIR is required (staged standalone app payload)"
!endif
!ifndef VST3_DIR
  !error "VST3_DIR is required (assembled HiFiShifter.vst3)"
!endif
!ifndef MODELS_DIR
  !error "MODELS_DIR is required (full model weights)"
!endif
!ifndef MODELS_VERSION
  !error "MODELS_VERSION is required (content hash from models.json)"
!endif
!ifndef MODELS_VERIFY
  !error "MODELS_VERIFY is required (generated verification include)"
!endif
!ifndef HFS_LICENSE
  !error "HFS_LICENSE is required"
!endif
!ifndef HFS_LICENSE_PAGE
  !error "HFS_LICENSE_PAGE is required (UTF-8 with BOM)"
!endif
!ifndef HFS_LICENSES
  !error "HFS_LICENSES is required"
!endif
!ifndef OUTPUT_FILE
  !error "OUTPUT_FILE is required"
!endif
!ifndef APP_VERSION
  !error "APP_VERSION is required"
!endif
!ifndef WEBVIEW2_BOOTSTRAPPER
  !error "WEBVIEW2_BOOTSTRAPPER is required (MicrosoftEdgeWebview2Setup.exe)"
!endif

; 组件索引：页面回调与 SectionGetFlags 需要的是**数字**，所以在这里显式编号。
; 【编号必须与下面的声明顺序一致】Section 语句已经带了这个标识符时 NSIS 会报
; "already defined, can't assign section index"，因此声明处不再重复写它。
!define SEC_APP 0
!define SEC_VST3 1
!define SEC_MODELS 2
!define SEC_SHORTCUTS 3

!define VST3_SUBDIR "HiFiShifter.vst3"
!define ARP_KEY "Software\Microsoft\Windows\CurrentVersion\Uninstall\HiFiShifter"
; 旧版插件安装器留下的卸载项：本安装器接管模型库与插件目录，成功后把它清掉。
!define LEGACY_VST3_ARP_KEY "Software\Microsoft\Windows\CurrentVersion\Uninstall\HiFiShifterVST3"
!define UNINSTALLER "$INSTDIR\uninstall.exe"
; WebView2 Evergreen Runtime 的产品 GUID（微软固定值，与语言无关）。
!define WEBVIEW2_GUID "{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}"

; ── 本地测试开关（发布构建绝不定义）────────────────────────────────────────
;
; 【为什么需要它】安装器真的会写 Program Files 与 HKLM，而没有管理员权限的机器
; 跑不了它、REAPER 也可能正开着（那正是安装器要拒绝的情形）。没有这个开关，
; "安装 → 查 ARP → 重装（幂等）→ /repair → 卸载"这条闭环完全无法在测试机上跑一遍，
; 而未经执行的安装器正是最容易出错的东西。
;
; 打开后：装到当前用户的临时目录、不要求提权、ARP 写 HKCU、跳过 REAPER 检查。
; 其余逻辑（目录对账、卸载段、注册表字段、页面与文案）与发布构建逐字相同。
; scripts/checks/check-product-consistency.ps1 断言默认分支仍是 admin / HKLM /
; Common Files —— 防止有人把测试分支误当默认。
!ifdef HFS_TEST_MODE
  InstallDir "$LOCALAPPDATA\HiFiShifterInstallerTest\App"
  RequestExecutionLevel user
  !define HFS_ARP_ROOT HKCU
  !define HFS_DEFAULT_VST3_DIR "$LOCALAPPDATA\HiFiShifterInstallerTest\VST3"
!else
  InstallDir "$PROGRAMFILES64\HiFiShifter"
  RequestExecutionLevel admin
  !define HFS_ARP_ROOT HKLM
  !define HFS_DEFAULT_VST3_DIR "$COMMONFILES64\VST3"
  !define HFS_REAPER_GUARD
!endif

; SetCompressor 必须在语言声明之前：语言文件一加载就已经有数据被压缩。
OutFile "${OUTPUT_FILE}"
SetCompressor /SOLID lzma
ShowInstDetails show
!define MUI_ABORTWARNING

; ── 页面 ────────────────────────────────────────────────────────────────
; 安装器与卸载器的页面都必须在语言声明之前（MUI 在加载第一个语言时插入界面函数，
; 那一刻它要知道有没有卸载器）。
!insertmacro MUI_PAGE_WELCOME

; 【为什么许可页看的不是随包的那份 LICENSE】NSIS 的 LicenseData 在**编译期**把文件读成
; UTF-16 存进安装器：文件带 BOM 才按 UTF-8 解码，不带就按**编译所在机器的 ANSI 代码页**
; 解码。LICENSE 是无 BOM 的 UTF-8，于是在中文 Windows 上编译出来的安装包里 θ 变成"胃"、
; 日文假名整段丢失 —— 而且编译一声不响，只有打开许可页才看得见。
; 所以 build.ps1 另写一份带 BOM 的 license-page.txt 专供这一页；HFS_LICENSE 仍是与仓库
; 逐字节相同的那份，负责装进用户磁盘。
!insertmacro MUI_PAGE_LICENSE "${HFS_LICENSE_PAGE}"
!insertmacro MUI_PAGE_COMPONENTS
!insertmacro MUI_PAGE_DIRECTORY
Page custom HfsVst3Page HfsVst3PageLeave
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH

!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES

; 第一个是默认语言（系统语言不匹配任何一个时回退到它）。
!insertmacro MUI_LANGUAGE "English"
!insertmacro MUI_LANGUAGE "SimpChinese"
!insertmacro MUI_LANGUAGE "TradChinese"
!insertmacro MUI_LANGUAGE "Japanese"
!insertmacro MUI_LANGUAGE "Korean"

!include "installer\installer_strings.nsh"

Name "$(HFS_NAME)"

Var HfsVst3Dialog
Var HfsVst3Input
Var HfsVst3Dir
Var HfsIsRepair
Var HfsModelRoot
Var HfsVerifyRoot
Var HfsListFile

; ── VST3 目录页 ─────────────────────────────────────────────────────────
; 【为什么单独一页】App 与插件装在两个完全不同的地方（Program Files 与
; Common Files\VST3）。只给一个目录页会让另一半落在用户没预期的地方。
; 组件页里没勾 VST3 时这一页直接跳过。
Function HfsVst3Page
  SectionGetFlags ${SEC_VST3} $0
  IntOp $0 $0 & ${SF_SELECTED}
  ${If} $0 == 0
    Abort
  ${EndIf}
  !insertmacro MUI_HEADER_TEXT "$(HFS_COMPONENT_VST3)" "$(HFS_DIR_TEXT)"
  nsDialogs::Create 1018
  Pop $HfsVst3Dialog
  ${If} $HfsVst3Dialog == error
    Abort
  ${EndIf}
  ${NSD_CreateLabel} 0 0 100% 24u "$(HFS_DIR_TEXT)"
  Pop $0
  ${NSD_CreateDirRequest} 0 26u 100% 12u "$HfsVst3Dir"
  Pop $HfsVst3Input
  nsDialogs::Show
FunctionEnd

Function HfsVst3PageLeave
  ${NSD_GetText} $HfsVst3Input $HfsVst3Dir
  ${If} $HfsVst3Dir == ""
    StrCpy $HfsVst3Dir "${HFS_DEFAULT_VST3_DIR}"
  ${EndIf}
FunctionEnd

; ── REAPER 运行检查 ─────────────────────────────────────────────────────
;
; 加载中的 DLL 会被宿主锁住：覆盖它要么失败、要么在 REAPER 里造成难以解释的崩溃。
; 安装与卸载**都要**检查（卸载时宿主还开着，目录删不干净且插件行为未定义）。
;
; 【为什么必须用 /FI 过滤】不能把整个 tasklist 输出取回来再找 reaper.exe：
; nsExec::ExecToStack 只保留输出的**前 1023 字节**，机器上进程一多 reaper.exe 就落在
; 窗口之外 —— 检查会**静默失效**。加上 /FI 之后输出只剩一行，既不会截断，也不必依赖
; 本地化的"未找到"提示文案。三个进程名都查：REAPER 可能把插件放在独立的 host 进程里。
;
; 结果写进 $0：1 = 正在运行。
Function HfsReaperRunning
  StrCpy $0 0
  nsExec::ExecToStack /TIMEOUT=10000 '"$SYSDIR\tasklist.exe" /FI "IMAGENAME eq reaper.exe" /FO CSV /NH'
  Pop $1
  Pop $2
  ${StrStr} $3 $2 '"reaper.exe"'
  ${If} $3 != ""
    StrCpy $0 1
    Return
  ${EndIf}
  nsExec::ExecToStack /TIMEOUT=10000 '"$SYSDIR\tasklist.exe" /FI "IMAGENAME eq reaper_host64.exe" /FO CSV /NH'
  Pop $1
  Pop $2
  ${StrStr} $3 $2 '"reaper_host64.exe"'
  ${If} $3 != ""
    StrCpy $0 1
    Return
  ${EndIf}
  nsExec::ExecToStack /TIMEOUT=10000 '"$SYSDIR\tasklist.exe" /FI "IMAGENAME eq reaper_host32.exe" /FO CSV /NH'
  Pop $1
  Pop $2
  ${StrStr} $3 $2 '"reaper_host32.exe"'
  ${If} $3 != ""
    StrCpy $0 1
  ${EndIf}
FunctionEnd

; 卸载器的同款检查（字符串函数前缀是 un.，因此必须另写一份）。
Function un.HfsReaperRunning
  StrCpy $0 0
  nsExec::ExecToStack /TIMEOUT=10000 '"$SYSDIR\tasklist.exe" /FI "IMAGENAME eq reaper.exe" /FO CSV /NH'
  Pop $1
  Pop $2
  ${UnStrStr} $3 $2 '"reaper.exe"'
  ${If} $3 != ""
    StrCpy $0 1
    Return
  ${EndIf}
  nsExec::ExecToStack /TIMEOUT=10000 '"$SYSDIR\tasklist.exe" /FI "IMAGENAME eq reaper_host64.exe" /FO CSV /NH'
  Pop $1
  Pop $2
  ${UnStrStr} $3 $2 '"reaper_host64.exe"'
  ${If} $3 != ""
    StrCpy $0 1
    Return
  ${EndIf}
  nsExec::ExecToStack /TIMEOUT=10000 '"$SYSDIR\tasklist.exe" /FI "IMAGENAME eq reaper_host32.exe" /FO CSV /NH'
  Pop $1
  Pop $2
  ${UnStrStr} $3 $2 '"reaper_host32.exe"'
  ${If} $3 != ""
    StrCpy $0 1
  ${EndIf}
FunctionEnd

; ── 主程序运行检查（移植自 Tauri 模板）────────────────────────────────────
;
; 【为什么用 /FI 而不是把 tasklist 输出接给 findstr】同上：只取一行，不依赖本地化文案。
Function HfsAppRunning
  StrCpy $0 0
  nsExec::ExecToStack /TIMEOUT=10000 '"$SYSDIR\tasklist.exe" /FI "IMAGENAME eq HiFiShifter.exe" /FO CSV /NH'
  Pop $1
  Pop $2
  ${StrStr} $3 $2 '"HiFiShifter.exe"'
  ${If} $3 != ""
    StrCpy $0 1
  ${EndIf}
FunctionEnd

; ── WebView2 运行时（移植自 Tauri 模板的 embedBootstrapper 分支）──────────
;
; 【为什么安装器必须自己管这件事】没有 WebView2 的 Windows 10 上，装完打开就是一片
; 空白窗口 —— 没有任何错误对话框，用户只会说"你们的软件打不开"。Windows 11 自带
; 运行时，但 Win10 未必。
Function HfsEnsureWebView2
  SetRegView 64
  ; 三处注册表：per-machine（WOW6432Node）、per-user、以及 64 位视图下的 per-machine。
  ReadRegStr $0 HKLM "SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\${WEBVIEW2_GUID}" "pv"
  ${If} $0 != ""
  ${AndIf} $0 != "0.0.0.0"
    Return
  ${EndIf}
  ReadRegStr $0 HKCU "Software\Microsoft\EdgeUpdate\Clients\${WEBVIEW2_GUID}" "pv"
  ${If} $0 != ""
  ${AndIf} $0 != "0.0.0.0"
    Return
  ${EndIf}
  ReadRegStr $0 HKLM "SOFTWARE\Microsoft\EdgeUpdate\Clients\${WEBVIEW2_GUID}" "pv"
  ${If} $0 != ""
  ${AndIf} $0 != "0.0.0.0"
    Return
  ${EndIf}

  DetailPrint "Installing the Microsoft Edge WebView2 runtime..."
  SetOutPath "$PLUGINSDIR"
  File "/oname=$PLUGINSDIR\MicrosoftEdgeWebview2Setup.exe" "${WEBVIEW2_BOOTSTRAPPER}"
  nsExec::ExecToLog '"$PLUGINSDIR\MicrosoftEdgeWebview2Setup.exe" /silent /install'
  Pop $0
  SetOutPath "$INSTDIR"

  ReadRegStr $0 HKLM "SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\${WEBVIEW2_GUID}" "pv"
  ${If} $0 != ""
  ${AndIf} $0 != "0.0.0.0"
    Return
  ${EndIf}
  ReadRegStr $0 HKCU "Software\Microsoft\EdgeUpdate\Clients\${WEBVIEW2_GUID}" "pv"
  ${If} $0 == ""
  ${OrIf} $0 == "0.0.0.0"
    MessageBox MB_OK|MB_ICONSTOP "$(HFS_WEBVIEW2_INSTALL_FAILED)"
  ${EndIf}
FunctionEnd

; ── 模型库：完整性校验（与运行时 looks_complete 同一判据）──────────────────
;
; 【为什么按"清单里每个文件的字节数"校验】运行时的 model_store::looks_complete 就是
; 这么判的。安装器若用另一套判据（比如只看目录存在），就会出现"安装器说装好了、
; 运行时说不可用"——用户看到的只是"推理不可用"，而模型目录看起来完全正常。
; 期望值由 build.ps1 从 models.json 生成到 MODELS_VERIFY 里（逐文件一行）。
Function HfsCheckModelFile
  ; $R0 = 相对路径，$R1 = 期望字节数，$R2 = 1（目前仍然完整）
  StrCpy $R3 "$HfsVerifyRoot\$R0"
  ${IfNot} ${FileExists} "$R3"
    StrCpy $R2 0
    Return
  ${EndIf}
  nsExec::ExecToStack /TIMEOUT=10000 'cmd /c for %I in ("$R3") do @echo %~zI'
  Pop $R4
  Pop $R5
  ${StrTrimNewLines} $R5 $R5
  ${If} $R5 != "$R1"
    StrCpy $R2 0
  ${EndIf}
FunctionEnd

; 校验 $HfsVerifyRoot；结果写进 $0（1 = 完整）。
Function HfsVerifyModels
  StrCpy $R2 1
  Call HfsVerifyModelFiles
  StrCpy $0 $R2
FunctionEnd

!include "${MODELS_VERIFY}"

; ── 模型库：GC（保留当前版本与另一个版本）──────────────────────────────────
;
; 【策略】只保留**当前版本**与**最近修改的另一个版本**（正常情况就是上一版）：
; 多留一个目录才能覆盖"回退到上一版 App"的现实场景；而版本号是内容哈希、无法排序，
; 所以用 dir /o-d 的修改时间排序取第一个别的版本。其余目录删除并 DetailPrint。
Function HfsModelsGc
  StrCpy $HfsListFile "$TEMP\hfs_models_gc.txt"
  nsExec::ExecToStack /TIMEOUT=10000 'cmd /c dir /b /ad /o-d "$HfsModelRoot" > "$HfsListFile" 2>nul'
  Pop $0
  Pop $1
  ${IfNot} ${FileExists} "$HfsListFile"
    Return
  ${EndIf}

  StrCpy $R0 ""   ; 保留的"另一个版本"
  FileOpen $R1 "$HfsListFile" r
  ${Do}
    FileRead $R1 $R2
    ${If} $R2 == ""
      ${ExitDo}
    ${EndIf}
    ${StrTrimNewLines} $R2 $R2
    ${If} $R2 == ""
      ${Continue}
    ${EndIf}
    ${If} $R2 == "${MODELS_VERSION}"
      ${Continue}
    ${EndIf}
    ${If} $R0 == ""
      StrCpy $R0 $R2
    ${EndIf}
  ${Loop}
  FileClose $R1

  nsExec::ExecToStack /TIMEOUT=10000 'cmd /c dir /b /ad /o-d "$HfsModelRoot" > "$HfsListFile" 2>nul'
  Pop $0
  Pop $1
  FileOpen $R1 "$HfsListFile" r
  ${Do}
    FileRead $R1 $R2
    ${If} $R2 == ""
      ${ExitDo}
    ${EndIf}
    ${StrTrimNewLines} $R2 $R2
    ${If} $R2 == ""
      ${Continue}
    ${EndIf}
    ${If} $R2 == "${MODELS_VERSION}"
      ${Continue}
    ${EndIf}
    ${If} $R2 == "$R0"
      ${Continue}
    ${EndIf}
    DetailPrint "Removing old model library $R2"
    RMDir /r "$HfsModelRoot\$R2"
  ${Loop}
  FileClose $R1
  Delete "$HfsListFile"
FunctionEnd

; ── 初始化 ──────────────────────────────────────────────────────────────
Function .onInit
  ${IfNot} ${RunningX64}
    ; 唯一一条无法本地化的提示：此时还不知道用户选了哪种语言。
    !insertmacro HFS_FATAL_NOT_X64
    Abort
  ${EndIf}
  SetRegView 64
  SetShellVarContext all
  StrCpy $HfsModelRoot "$APPDATA\HiFiShifter\models"
  SetShellVarContext current
  StrCpy $HfsVst3Dir "${HFS_DEFAULT_VST3_DIR}"

  ; /repair：只做校验与补齐，不弹任何页面（文件是幂等覆盖，模型段自己会跳过）。
  ${StrStr} $0 $CMDLINE "/repair"
  ${If} $0 != ""
    StrCpy $HfsIsRepair 1
    SetSilent silent
  ${EndIf}

  ; 语言选择：MUI 会把选择记在注册表里，下次直接沿用。
  !insertmacro MUI_LANGDLL_DISPLAY
  !ifdef HFS_REAPER_GUARD
    Call HfsReaperRunning
    ${If} $0 == 1
      MessageBox MB_OK|MB_ICONSTOP "$(HFS_ERR_REAPER_OPEN)"
      Abort
    ${EndIf}
  !endif
FunctionEnd

Function un.onInit
  SetRegView 64
  ; 【为什么不能直接用 $INSTDIR】卸载器会把自己复制到临时目录再运行，并传入 _?=
  ; —— 那个值是**卸载器所在目录**，不是可信来源。从安装时写下的 InstallLocation
  ; 读回来才是权威值；读不到时退回卸载器所在目录。
  ReadRegStr $INSTDIR ${HFS_ARP_ROOT} "${ARP_KEY}" "InstallLocation"
  ${If} $INSTDIR == ""
    StrCpy $INSTDIR "$EXEDIR"
  ${EndIf}
  ReadRegStr $HfsVst3Dir ${HFS_ARP_ROOT} "${ARP_KEY}" "Vst3Dir"
  ${If} $HfsVst3Dir == ""
    StrCpy $HfsVst3Dir "${HFS_DEFAULT_VST3_DIR}"
  ${EndIf}

  !ifdef HFS_REAPER_GUARD
    Call un.HfsReaperRunning
    ${If} $0 == 1
      MessageBox MB_OK|MB_ICONSTOP "$(HFS_UNINSTALL_REAPER_OPEN)"
      Abort
    ${EndIf}
  !endif
FunctionEnd

; ═══ 组件 ═══════════════════════════════════════════════════════════════

; ── 1. 独立 App ─────────────────────────────────────────────────────────
Section "$(HFS_COMPONENT_APP)"
  SectionIn RO
  !ifdef HFS_REAPER_GUARD
    Call HfsReaperRunning
    ${If} $0 == 1
      MessageBox MB_OK|MB_ICONSTOP "$(HFS_ERR_REAPER_OPEN)"
      Abort
    ${EndIf}
  !endif
  Call HfsAppRunning
  ${If} $0 == 1
    MessageBox MB_OK|MB_ICONSTOP "$(HFS_ERR_APP_RUNNING)"
    Abort
  ${EndIf}

  DetailPrint "$(HFS_INSTALLING)"
  Call HfsEnsureWebView2

  ; ── 目录对账：先清掉上一代产物，再写入 ──────────────────────────────
  ;
  ; 【为什么必须清 $INSTDIR\models】旧安装器（Tauri 模板）把 150 MB 权重装在这里；
  ; 新布局只在这里放 models.json（权重进 ProgramData 的共享库）。不删就等于升级后
  ; 悄悄多占 150 MB，而且 App 还会优先读到这份旧权重。
  ;
  ; 【为什么删 licenses 与 LICENSE】旧版本把它们放在安装根；现在整组随 App 重装，
  ; 先删可以保证"文档少了一份"不会靠覆盖而静默延续。
  RMDir /r "$INSTDIR\models"
  RMDir /r "$INSTDIR\licenses"
  Delete "$INSTDIR\LICENSE"

  SetOutPath "$INSTDIR"
  File /r "${APP_DIR}\*"

  ; 许可全文：App 与插件各自带一份完整副本（用户可能只装其中一个）。
  SetOutPath "$INSTDIR"
  File "${HFS_LICENSE}"
  SetOutPath "$INSTDIR\licenses"
  File /r "${HFS_LICENSES}\*"

  ; ── 卸载信息 ────────────────────────────────────────────────────────
  ;
  ; 【为什么复用 \Uninstall\HiFiShifter 这个键】旧版 App（Tauri 的 NSIS 安装器）
  ; 用的就是它。复用之后升级是"接管"，而不是在「已安装的应用」里出现两个 HiFiShifter。
  WriteUninstaller "${UNINSTALLER}"
  WriteRegStr ${HFS_ARP_ROOT} "${ARP_KEY}" "DisplayName" "$(HFS_NAME)"
  WriteRegStr ${HFS_ARP_ROOT} "${ARP_KEY}" "DisplayVersion" "${APP_VERSION}"
  WriteRegStr ${HFS_ARP_ROOT} "${ARP_KEY}" "DisplayIcon" "$INSTDIR\HiFiShifter.exe"
  WriteRegStr ${HFS_ARP_ROOT} "${ARP_KEY}" "Publisher" "$(HFS_PUBLISHER)"
  WriteRegStr ${HFS_ARP_ROOT} "${ARP_KEY}" "InstallLocation" "$INSTDIR"
  WriteRegStr ${HFS_ARP_ROOT} "${ARP_KEY}" "Vst3Dir" "$HfsVst3Dir"
  ; 不带 _?=：让卸载器按默认行为把自己复制到临时目录再运行，从而能删掉自身。
  WriteRegStr ${HFS_ARP_ROOT} "${ARP_KEY}" "UninstallString" '"${UNINSTALLER}"'
  WriteRegStr ${HFS_ARP_ROOT} "${ARP_KEY}" "QuietUninstallString" '"${UNINSTALLER}" /S'
  WriteRegDWORD ${HFS_ARP_ROOT} "${ARP_KEY}" "NoModify" 1
  WriteRegDWORD ${HFS_ARP_ROOT} "${ARP_KEY}" "NoRepair" 1
SectionEnd

; ── 2. VST3 / ARA 插件 ──────────────────────────────────────────────────
Section "$(HFS_COMPONENT_VST3)"
  !ifdef HFS_REAPER_GUARD
    Call HfsReaperRunning
    ${If} $0 == 1
      MessageBox MB_OK|MB_ICONSTOP "$(HFS_ERR_REAPER_OPEN)"
      Abort
    ${EndIf}
  !endif

  DetailPrint "$(HFS_INSTALLING)"

  ; ── 目录对账（与旧插件安装器逐字相同）──────────────────────────────
  ;
  ; 【为什么必须清】Vite 给每个 chunk 加了内容哈希，内容一变文件名就变；而 File /r
  ; 只写不删。于是每次重装都会再堆一层，目录单调增长。这两个目录是**纯构建产出**，
  ; 删除是幂等且安全的。
  RMDir /r "$HfsVst3Dir\${VST3_SUBDIR}\Contents\Resources\frontend"
  RMDir /r "$HfsVst3Dir\${VST3_SUBDIR}\Contents\x86_64-win"
  ; 早期版本把许可全文与变更说明放在 Resources 根目录；它们已不再交付。
  Delete "$HfsVst3Dir\${VST3_SUBDIR}\Contents\Resources\COPYING.LGPL-2.1"
  Delete "$HfsVst3Dir\${VST3_SUBDIR}\Contents\Resources\SOUNDTOUCH-CHANGES.md"
  Delete "$HfsVst3Dir\${VST3_SUBDIR}\Contents\Resources\VST3-BUILD.md"
  Delete "$HfsVst3Dir\${VST3_SUBDIR}\Contents\Resources\frontend\index.html"
  Delete "$HfsVst3Dir\${VST3_SUBDIR}\Contents\Resources\frontend\detached.html"
  Delete "$HfsVst3Dir\${VST3_SUBDIR}\Contents\Resources\frontend\waveform-test.html"

  SetOutPath "$HfsVst3Dir\${VST3_SUBDIR}"
  File /r "${VST3_DIR}\*"
  SetOutPath "$HfsVst3Dir\${VST3_SUBDIR}\Contents\Resources"
  File "${HFS_LICENSE}"
  SetOutPath "$HfsVst3Dir\${VST3_SUBDIR}\Contents\Resources\licenses"
  File /r "${HFS_LICENSES}\*"

  ; 旧版插件安装器有自己的卸载项（\Uninstall\HiFiShifterVST3）。本安装器接管了插件
  ; 的安装位置与模型库，成功后把它清掉，避免「已安装的应用」里出现两条。
  DeleteRegKey HKLM "${LEGACY_VST3_ARP_KEY}"
  DeleteRegKey HKCU "${LEGACY_VST3_ARP_KEY}"
SectionEnd

; ── 3. 共享模型库 ───────────────────────────────────────────────────────
;
; 【为什么由安装器独占写入】模型是唯一"两个形态都要用、又只该存一份"的东西。
; 装进 %PROGRAMDATA% 之后，App 与插件都从这一份读（见 model_store），用户只为完全
; 相同的 152 MB 付一次磁盘。
;
; 【为什么目录名带版本号】版本由模型内容派生（scripts/setup/fetch-models.py --write-manifest）。
; 升级时新版本落新目录，旧代码不会读到新权重；也正因为如此，同版本已存在就能安全跳过
; ——重装不必再写 152 MB。
Section "$(HFS_COMPONENT_MODELS)"
  SectionIn RO

  StrCpy $HfsVerifyRoot "$HfsModelRoot\${MODELS_VERSION}"
  Call HfsVerifyModels
  ${If} $0 == 1
    DetailPrint "$(HFS_COMPONENT_MODELS_SKIP)"
    Goto hfs_models_done
  ${EndIf}

  DetailPrint "$(HFS_COMPONENT_MODELS_INSTALL)"
  ; 先写 staging 再整体改名：中途失败不会留下一个"看起来存在但缺文件"的库，那会让
  ; 运行时误判为可用（与 model_store::publish 是同一套做法）。
  RMDir /r "$HfsModelRoot\${MODELS_VERSION}.staging"
  CreateDirectory "$HfsModelRoot\${MODELS_VERSION}.staging"
  SetOutPath "$HfsModelRoot\${MODELS_VERSION}.staging"
  File /r "${MODELS_DIR}\*"

  StrCpy $HfsVerifyRoot "$HfsModelRoot\${MODELS_VERSION}.staging"
  Call HfsVerifyModels
  ${If} $0 != 1
    RMDir /r "$HfsModelRoot\${MODELS_VERSION}.staging"
    MessageBox MB_OK|MB_ICONSTOP "$(HFS_MODELS_VERIFY_FAILED)"
    Abort
  ${EndIf}
  RMDir /r "$HfsModelRoot\${MODELS_VERSION}"
  Rename "$HfsModelRoot\${MODELS_VERSION}.staging" "$HfsModelRoot\${MODELS_VERSION}"

  hfs_models_done:
  ; 权限：ProgramData 的继承在部分机器上不完整，显式给 Users 读权限。用 SID
  ; （S-1-5-32-545 = Users）而不是本地化的组名 —— 组名在非英文系统上匹配不到。
  nsExec::ExecToLog 'icacls "$HfsModelRoot" /grant *S-1-5-32-545:(OI)(CI)RX /T /Q'
  Call HfsModelsGc

  SetOutPath "$INSTDIR"
  ${If} $HfsIsRepair == 1
    DetailPrint "$(HFS_REPAIR_DONE)"
  ${EndIf}
SectionEnd

; ── 4. 快捷方式 ─────────────────────────────────────────────────────────
Section "$(HFS_COMPONENT_SHORTCUTS)"
  SetShellVarContext all
  CreateDirectory "$SMPROGRAMS\HiFiShifter"
  CreateShortCut "$SMPROGRAMS\HiFiShifter\HiFiShifter.lnk" "$INSTDIR\HiFiShifter.exe"
  CreateShortCut "$DESKTOP\HiFiShifter.lnk" "$INSTDIR\HiFiShifter.exe"
  SetShellVarContext current
SectionEnd

; ── 卸载 ────────────────────────────────────────────────────────────────
Section "Uninstall"
  ; 只删自己：$HfsVst3Dir 是 Common Files\VST3，里面还有别人的插件。
  RMDir /r "$HfsVst3Dir\${VST3_SUBDIR}"
  ; App 目录整个属于本产品（与 Tauri 安装器的行为一致）。
  RMDir /r "$INSTDIR"

  SetShellVarContext all
  RMDir /r "$SMPROGRAMS\HiFiShifter"
  Delete "$DESKTOP\HiFiShifter.lnk"
  SetShellVarContext current

  DeleteRegKey ${HFS_ARP_ROOT} "${ARP_KEY}"
  ; 旧版安装器可能还留着键（本安装器成功时会删，但用户可能先卸载再手动清理）。
  DeleteRegKey HKLM "${LEGACY_VST3_ARP_KEY}"
  DeleteRegKey HKCU "${LEGACY_VST3_ARP_KEY}"

  ; ── 模型库：还有别的 HiFiShifter 安装才保留 ─────────────────────────
  Call un.HfsOtherInstallExists
  ${If} $0 == 0
    SetShellVarContext all
    StrCpy $HfsModelRoot "$APPDATA\HiFiShifter\models"
    SetShellVarContext current
    RMDir /r "$HfsModelRoot\${MODELS_VERSION}"
    ; 空壳目录也顺手清掉（只有它确实空了才会成功）。
    RMDir "$HfsModelRoot"
    RMDir "$APPDATA\HiFiShifter"
  ${EndIf}

  ; 【刻意不删的东西】
  ; - %APPDATA%\HiFiShifter：用户设置、最近工程、快捷键 —— 删了等于让用户重配一次；
  ; - %LOCALAPPDATA%\HiFiShifter：缓存、日志、未保存会话。前者是派生物，后者可能是
  ;   用户还没保存的工程素材。
  ; 两者都只在卸载详情里留下说明。
  DetailPrint "Settings, caches and logs were kept in your user profile."
SectionEnd

; 还有别的 HiFiShifter 安装项吗？结果写进 $0（1 = 有）。
Function un.HfsOtherInstallExists
  StrCpy $0 0
  StrCpy $R1 "HKLM"
  Call un.HfsScanUninstallRoot
  ${If} $0 == 1
    Return
  ${EndIf}
  StrCpy $R1 "HKCU"
  Call un.HfsScanUninstallRoot
FunctionEnd

Function un.HfsScanUninstallRoot
  StrCpy $R2 0
  ${Do}
    ${If} $R1 == "HKLM"
      EnumRegKey $R3 HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall" $R2
    ${Else}
      EnumRegKey $R3 HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall" $R2
    ${EndIf}
    ${If} $R3 == ""
      ${ExitDo}
    ${EndIf}
    IntOp $R2 $R2 + 1
    ; 自己（HiFiShifter）已经被删掉了，这里只需要看有没有别的同族项
    ; （旧版的 HiFiShifterVST3，或者将来拆出来的其它形态）。
    StrCpy $R4 $R3 11
    ${If} $R4 == "HiFiShifter"
      StrCpy $0 1
      ${ExitDo}
    ${EndIf}
  ${Loop}
FunctionEnd
