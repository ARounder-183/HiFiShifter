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
;   1. 宿主运行守卫：安装与卸载都查（已加载插件的 DAW 会锁住 DLL，覆盖 = 宿主崩溃）；
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
  !error "VST3_DIR is required (contents of the assembled HiFiShifter.vst3)"
!endif
!ifndef CLAP_FILE
  !error "CLAP_FILE is required (assembled HiFiShifter.clap module)"
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
!ifndef HFS_ICON
  !error "HFS_ICON is required (HiFiShifter icon.ico from the Tauri app icons)"
!endif

; 组件索引：自定义页函数写在 Section 声明**之前**，那时 NSIS 自动生成的
; ${SEC_*} 符号还不存在（实测会报 "unknown variable/constant"），所以这里仍用
; 显式数字。SectionGroup 自身也会占一个索引，且末尾的 SectionGroupEnd 再占一个
; —— 实测顺序为 0=组、1/2/3=子项、4=组结束、5=组、6/7=子项。
; 【改动声明顺序必须同步改这里】漏改不会编译失败，只会让 VST3/CLAP 目录页判断错
; 组件是否被勾选 —— 那是最难从 UI 上察觉的一类回归。
!define SECGRP_APP 0
!define SEC_APP 1
!define SEC_MODELS 2
!define SEC_SHORTCUTS 3
; 4 = 第一个 SectionGroupEnd
!define SECGRP_PLUGINS 5
!define SEC_VST3 6
!define SEC_CLAP 7


!define VST3_SUBDIR "HiFiShifter.vst3"
!define CLAP_FILENAME "HiFiShifter.clap"
!define ARP_KEY "Software\Microsoft\Windows\CurrentVersion\Uninstall\HiFiShifter"
; 旧版插件安装器留下的卸载项：本安装器接管模型库与插件目录，成功后把它清掉。
!define LEGACY_VST3_ARP_KEY "Software\Microsoft\Windows\CurrentVersion\Uninstall\HiFiShifterVST3"
!define UNINSTALLER "$INSTDIR\uninstall.exe"
; WebView2 Evergreen Runtime 的产品 GUID（微软固定值，与语言无关）。
!define WEBVIEW2_GUID "{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}"

; ── 本地测试开关（发布构建绝不定义）────────────────────────────────────────
;
; 【为什么需要它】安装器真的会写 Program Files 与 HKLM，而没有管理员权限的机器
; 跑不了它、也常有 DAW 正开着（那正是安装器要拒绝的情形）。没有这个开关，
; "安装 → 查 ARP → 重装（幂等）→ /repair → 卸载"这条闭环完全无法在测试机上跑一遍，
; 而未经执行的安装器正是最容易出错的东西。
;
; 打开后：装到当前用户的临时目录、不要求提权、ARP 写 HKCU、跳过宿主运行检查。
; 其余逻辑（目录对账、卸载段、注册表字段、页面与文案）与发布构建逐字相同。
; scripts/checks/check-product-consistency.ps1 断言默认分支仍是 admin / HKLM /
; Common Files —— 防止有人把测试分支误当默认。
!ifdef HFS_TEST_MODE
  InstallDir "$LOCALAPPDATA\HiFiShifterInstallerTest\App"
  RequestExecutionLevel user
  !define HFS_ARP_ROOT HKCU
  !define HFS_DEFAULT_VST3_DIR "$LOCALAPPDATA\HiFiShifterInstallerTest\VST3"
  !define HFS_DEFAULT_CLAP_DIR "$LOCALAPPDATA\HiFiShifterInstallerTest\CLAP"
  ; 【为什么测试模式换模型根】测试安装器是 user 权限，写 %ProgramData% 会触发 UAC
  ; 虚拟化：File/SetOutPath 落到 VirtualStore，而 Rename/FileExists 走真实路径 ——
  ; 模型永远停在 <版本>.staging，安装与卸载都判错。改用 current 上下文的 $APPDATA
  ; （用户漫游目录）就不虚拟化，门禁才能真实验证模型闭环。
  !define HFS_MODEL_ROOT "$APPDATA\HiFiShifterTest\models"
  !define HFS_MODEL_ROOT_CONTEXT current
!else
  InstallDir "$PROGRAMFILES64\HiFiShifter"
  RequestExecutionLevel admin
  !define HFS_ARP_ROOT HKLM
  !define HFS_DEFAULT_VST3_DIR "$COMMONFILES64\VST3"
  !define HFS_DEFAULT_CLAP_DIR "$COMMONFILES64\CLAP"
  ; 发布安装器是 admin，写 %ProgramData% 不虚拟化；all 上下文的 $APPDATA 正是它。
  !define HFS_MODEL_ROOT "$APPDATA\HiFiShifter\models"
  !define HFS_MODEL_ROOT_CONTEXT all
  !define HFS_HOST_GUARD
!endif

; SetCompressor 必须在语言声明之前：语言文件一加载就已经有数据被压缩。
OutFile "${OUTPUT_FILE}"
SetCompressor /SOLID lzma
ShowInstDetails show
!define MUI_ABORTWARNING

; 【为什么安装器用自己的图标】MUI2 默认用 NSIS 自带的 modern-install.ico ——
; 右上角那个与产品无关的圆形 logo。HiFiShifter 的图标随 Tauri App 一起维护在
; backend\src-tauri\icons\icon.ico，由构建脚本以 /D 传入（与其它路径一样不在
; 脚本里猜相对位置）。安装器与卸载器都换成它，打开向导就能认出是本产品。
!define MUI_ICON "${HFS_ICON}"
!define MUI_UNICON "${HFS_ICON}"

; 【为什么这些文案必须在页面宏之前定义】MUI 在 !insertmacro MUI_PAGE_* 时就把
; 页面文案烘进界面函数；$(HFS_...) 是 LangString 的前向引用，运行时才取值，
; 因此定义放在语言声明之前没有问题（已用最小脚本验证）。此前没有定义，欢迎页、
; 组件页与目录页都退回 MUI 自带的英文默认文案，五语词表里的文案从未被显示。
!define MUI_WELCOMEPAGE_TITLE "$(HFS_WELCOME_TITLE)"
!define MUI_WELCOMEPAGE_TEXT "$(HFS_WELCOME_TEXT)"
!define MUI_COMPONENTSPAGE_TEXT_TOP "$(HFS_COMPONENTS_TOP)"
!define MUI_DIRECTORYPAGE_TEXT_TOP "$(HFS_APP_DIR_TEXT)"

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
!define MUI_PAGE_CUSTOMFUNCTION_SHOW HfsComponentsPageShow
; ── 组件页布局：把树加宽、描述框改成底部通栏 ─────────────────────────
; 【为什么在运行时改】MUI2 自带的 IDD_SELCOM 把描述框放在右侧，树只剩 147px，
; 中文组件名被截断（用户实测截图）；换 SMALLDESC 又会把树压到 5 行、底部两项
; 被裁掉。页面 SHOW 时用 Win32 MoveWindow 重排：纯 NSIS、无自定义 UI 资源、
; 无二进制文件、构建脚本不用改。实测子对话框客户区 450x210，下面的坐标严格
; 落在其中；控件 ID 与 MUI2 的 GetDlgItem 一致（1006/1022/1023/1032/1042/1043）。
Function HfsComponentsPageShow
  FindWindow $0 "#32770" "" $HWNDPARENT
  ${If} $0 == 0
    Return
  ${EndIf}
  ; 顶部说明 → 组件列表标签 → 树（全宽、7 行）→ 所需空间 + 描述标题 → 描述通栏
  GetDlgItem $1 $0 1006
  System::Call 'user32::MoveWindow(p $1, i 8, i 2, i 434, i 24, i 1)'
  GetDlgItem $1 $0 1022
  System::Call 'user32::MoveWindow(p $1, i 8, i 28, i 434, i 10, i 1)'
  GetDlgItem $1 $0 1032
  System::Call 'user32::MoveWindow(p $1, i 8, i 40, i 434, i 118, i 1)'
  GetDlgItem $1 $0 1023
  System::Call 'user32::MoveWindow(p $1, i 8, i 162, i 110, i 10, i 1)'
  GetDlgItem $1 $0 1042
  System::Call 'user32::MoveWindow(p $1, i 122, i 162, i 320, i 10, i 1)'
  GetDlgItem $1 $0 1043
  System::Call 'user32::MoveWindow(p $1, i 8, i 174, i 434, i 34, i 1)'
FunctionEnd

!insertmacro MUI_PAGE_COMPONENTS
!insertmacro MUI_PAGE_DIRECTORY
Page custom HfsVst3Page HfsVst3PageLeave
Page custom HfsClapPage HfsClapPageLeave
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH

!insertmacro MUI_UNPAGE_CONFIRM
; 【为什么用 UninstPage custom 而不是 MUI 的组件页】卸载器只需要两个可勾选项，
; 复用安装侧的组件页会带来整套 section 索引与组件页布局回调；两复选框的自定义页
; 更轻，而且 un.onInit 里的默认值在静默卸载时不经页面直接生效。
UninstPage custom un.HfsUninstallOptionsCreate un.HfsUninstallOptionsLeave
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
Var HfsClapDialog
Var HfsClapInput
Var HfsClapDir
Var HfsIsRepair
Var HfsModelRoot
Var HfsVerifyRoot
Var HfsListFile
; 卸载选项页：两个复选框的状态（1 = 清理）。un.onInit 默认置 0 —— 静默卸载
; 不走选项页，因此默认是"都保留"；只有用户在选项页主动勾选才会清理。
Var HfsUnOptModels
Var HfsUnOptCache
Var HfsUnChkModels
Var HfsUnChkCache
Var HfsUnOtherInstall

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

Function HfsClapPage
  SectionGetFlags ${SEC_CLAP} $0
  IntOp $0 $0 & ${SF_SELECTED}
  ${If} $0 == 0
    Abort
  ${EndIf}
  !insertmacro MUI_HEADER_TEXT "$(HFS_COMPONENT_CLAP)" "$(HFS_CLAP_DIR_TEXT)"
  nsDialogs::Create 1018
  Pop $HfsClapDialog
  ${If} $HfsClapDialog == error
    Abort
  ${EndIf}
  ${NSD_CreateLabel} 0 0 100% 24u "$(HFS_CLAP_DIR_TEXT)"
  Pop $0
  ${NSD_CreateDirRequest} 0 26u 100% 12u "$HfsClapDir"
  Pop $HfsClapInput
  nsDialogs::Show
FunctionEnd

Function HfsClapPageLeave
  ${NSD_GetText} $HfsClapInput $HfsClapDir
  ${If} $HfsClapDir == ""
    StrCpy $HfsClapDir "${HFS_DEFAULT_CLAP_DIR}"
  ${EndIf}
FunctionEnd

Function HfsVst3PageLeave
  ${NSD_GetText} $HfsVst3Input $HfsVst3Dir
  ${If} $HfsVst3Dir == ""
    StrCpy $HfsVst3Dir "${HFS_DEFAULT_VST3_DIR}"
  ${EndIf}
FunctionEnd

; ── 宿主运行检查（安装与卸载都要查）────────────────────────────────────
;
; 【为什么不再按 DAW 名字枚举进程】本项目的目标是让所有支持 VST3 / CLAP 的宿主
; 都能用；写死某个 DAW 的进程名等于告诉别的用户"这个提示与你无关"，也会漏掉
; 那些把插件放在独立宿主进程里的 DAW。改为**按模块名**查：谁加载了本插件的
; DLL，谁就是要先退出的宿主，与它叫什么名字无关。
;
; 【为什么用 Sysnative】NSIS 是 32 位进程，直接调 tasklist.exe 会被 WOW64 重定向
; 到 SysWOW64 的 32 位版本，而 32 位 tasklist 列不出 64 位 DAW 加载的模块 ——
; 检查会静默失效。Sysnative 是 WOW64 给 32 位进程访问原生 System32 的虚拟别名
; （32 位系统上不存在，因此下面先判文件是否存在，再退回 System32）。
;
; 【为什么用 /FI 而不是把输出接给 findstr】nsExec::ExecToStack 只保留输出的
; 前 1023 字节，进程一多就截断；/FI 让 tasklist 自己过滤，输出只剩命中行，
; 也不依赖本地化的"未找到"文案。三个模块名覆盖 VST3 / CLAP 两种形态：
; HiFiShifter.vst3 是 VST3 外壳加载器，HiFiShifter.clap 是 CLAP 模块，
; HiFiShifterEngine.dll 是两者共同加载的核心（有些宿主只列出后者）。
;
; 结果写进 $0：1 = 仍有宿主加载着本插件。
Function HfsHostRunning
  StrCpy $0 0
  StrCpy $R6 "$WINDIR\Sysnative\tasklist.exe"
  ${IfNot} ${FileExists} "$R6"
    StrCpy $R6 "$SYSDIR\tasklist.exe"
  ${EndIf}
  StrCpy $R7 "HiFiShifter.vst3"
  Call HfsHostModuleProbe
  ${If} $0 == 0
    StrCpy $R7 "HiFiShifter.clap"
    Call HfsHostModuleProbe
  ${EndIf}
  ${If} $0 == 0
    StrCpy $R7 "HiFiShifterEngine.dll"
    Call HfsHostModuleProbe
  ${EndIf}
FunctionEnd

; $R7 = 模块名，$R6 = tasklist 路径；命中时把 $0 置 1。
Function HfsHostModuleProbe
  nsExec::ExecToStack /TIMEOUT=20000 '"$R6" /FI "MODULES eq $R7" /FO CSV /NH'
  Pop $1
  Pop $2
  ${StrStr} $3 $2 '"'
  ${If} $3 != ""
    StrCpy $0 1
  ${EndIf}
FunctionEnd

; 卸载器的同款检查（字符串函数前缀是 un.，因此必须另写一份）。
Function un.HfsHostRunning
  StrCpy $0 0
  StrCpy $R6 "$WINDIR\Sysnative\tasklist.exe"
  ${IfNot} ${FileExists} "$R6"
    StrCpy $R6 "$SYSDIR\tasklist.exe"
  ${EndIf}
  StrCpy $R7 "HiFiShifter.vst3"
  Call un.HfsHostModuleProbe
  ${If} $0 == 0
    StrCpy $R7 "HiFiShifter.clap"
    Call un.HfsHostModuleProbe
  ${EndIf}
  ${If} $0 == 0
    StrCpy $R7 "HiFiShifterEngine.dll"
    Call un.HfsHostModuleProbe
  ${EndIf}
FunctionEnd

Function un.HfsHostModuleProbe
  nsExec::ExecToStack /TIMEOUT=20000 '"$R6" /FI "MODULES eq $R7" /FO CSV /NH'
  Pop $1
  Pop $2
  ${UnStrStr} $3 $2 '"'
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
  SetShellVarContext ${HFS_MODEL_ROOT_CONTEXT}
  StrCpy $HfsModelRoot "${HFS_MODEL_ROOT}"
  SetShellVarContext current
  StrCpy $HfsVst3Dir "${HFS_DEFAULT_VST3_DIR}"
  ; 【为什么两个目录都要在 .onInit 初始化】自定义页的 Leave 回调只在用户
  ; 真正走过那一页时运行；静默安装（/S）跳过所有页面。此前只初始化了 VST3，
  ; CLAP 目录因此是空串，File 会落到当前目录（VirtualStore）而不是目标目录。
  StrCpy $HfsClapDir "${HFS_DEFAULT_CLAP_DIR}"

  ; /repair：只做校验与补齐，不弹任何页面（文件是幂等覆盖，模型段自己会跳过）。
  ${StrStr} $0 $CMDLINE "/repair"
  ${If} $0 != ""
    StrCpy $HfsIsRepair 1
    SetSilent silent
  ${EndIf}

  ; 语言选择：MUI 会把选择记在注册表里，下次直接沿用。
  !insertmacro MUI_LANGDLL_DISPLAY
  !ifdef HFS_HOST_GUARD
    Call HfsHostRunning
    ${If} $0 == 1
      MessageBox MB_OK|MB_ICONSTOP "$(HFS_ERR_HOST_OPEN)"
      Abort
    ${EndIf}
  !endif
FunctionEnd

Function un.onInit
  ; 【为什么默认都不勾选】卸载只是移除程序，不该抹掉还能用的东西：
  ; - 渲染缓存删了会让"卸载重装原版本"排障时首开变慢，运行时本就有
  ;   temp_sweep / clear_render_cache 按年龄回收，磁盘清理不需要卸载器代劳；
  ; - 日志是定位"上次为什么没工作"的唯一线索，删掉等于毁掉证据；
  ; - 模型库保留后，重装同版本可直接复用，装新版时 HfsModelsGc 自己会 GC。
  ; 用户想做一次"干净卸载"时，在选项页主动勾选即可。静默卸载（/S）不走选项页，
  ; 这两个 0 直接生效 —— 于是 CI/门禁也是"保留一切"，符合最小惊讶原则。
  StrCpy $HfsUnOptModels 0
  StrCpy $HfsUnOptCache 0
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
  ReadRegStr $HfsClapDir ${HFS_ARP_ROOT} "${ARP_KEY}" "ClapDir"
  ${If} $HfsClapDir == ""
    StrCpy $HfsClapDir "${HFS_DEFAULT_CLAP_DIR}"
  ${EndIf}

  !ifdef HFS_HOST_GUARD
    Call un.HfsHostRunning
    ${If} $0 == 1
      MessageBox MB_OK|MB_ICONSTOP "$(HFS_UNINSTALL_HOST_OPEN)"
      Abort
    ${EndIf}
  !endif
FunctionEnd

; ═══ 组件 ═══════════════════════════════════════════════════════════════
;
; 【为什么分两组】此前五个条目平铺，App、共享模型库、快捷方式与两种插件格式
; 混在一起，用户看不出哪些是主程序、哪些是给 DAW 用的。现在按用途分成两棵可
; 展开的树：主程序一组，DAW 插件一组。分组节点也占一个 section 索引（见文件
; 顶部 SECGRP_* 的编号说明）。
;
; 【必装项放进分组是安全的】实测：把分组整体取消勾选时，带 SectionIn RO 的
; App 与共享模型库仍保持选中并照常执行，只有可选的快捷方式被连带取消。
;
; ── 主程序 ───────────────────────────────────────────────────────────
SectionGroup /e "$(HFS_GROUP_APP)"
Section "$(HFS_COMPONENT_APP)"
  SectionIn RO
  !ifdef HFS_HOST_GUARD
    Call HfsHostRunning
    ${If} $0 == 1
      MessageBox MB_OK|MB_ICONSTOP "$(HFS_ERR_HOST_OPEN)"
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
  WriteRegStr ${HFS_ARP_ROOT} "${ARP_KEY}" "ClapDir" "$HfsClapDir"
  ; 不带 _?=：让卸载器按默认行为把自己复制到临时目录再运行，从而能删掉自身。
  WriteRegStr ${HFS_ARP_ROOT} "${ARP_KEY}" "UninstallString" '"${UNINSTALLER}"'
  WriteRegStr ${HFS_ARP_ROOT} "${ARP_KEY}" "QuietUninstallString" '"${UNINSTALLER}" /S'
  WriteRegDWORD ${HFS_ARP_ROOT} "${ARP_KEY}" "NoModify" 1
  WriteRegDWORD ${HFS_ARP_ROOT} "${ARP_KEY}" "NoRepair" 1
SectionEnd
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
  ; 【为什么必须先切走 SetOutPath】NSIS 的 SetOutPath 会 SetCurrentDirectory；
  ; 上面 File /r 把当前目录设成了 .staging，直接 Rename 它就是"重命名当前工作目录"，
  ; Windows 必然失败且 Rename 不报错 —— 模型库永远停在 .staging，安装显示成功、
  ; 运行时却读不到权重（实测 RenameError=1）。先切回 $INSTDIR 再改名。
  SetOutPath "$INSTDIR"
  RMDir /r "$HfsModelRoot\${MODELS_VERSION}"
  Rename "$HfsModelRoot\${MODELS_VERSION}.staging" "$HfsModelRoot\${MODELS_VERSION}"
  ${If} ${Errors}
    MessageBox MB_OK|MB_ICONSTOP "$(HFS_MODELS_VERIFY_FAILED)"
    Abort
  ${EndIf}
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
Section "$(HFS_COMPONENT_SHORTCUTS)"
  SetShellVarContext all
  CreateDirectory "$SMPROGRAMS\HiFiShifter"
  CreateShortCut "$SMPROGRAMS\HiFiShifter\HiFiShifter.lnk" "$INSTDIR\HiFiShifter.exe"
  CreateShortCut "$DESKTOP\HiFiShifter.lnk" "$INSTDIR\HiFiShifter.exe"
  SetShellVarContext current
SectionEnd
SectionGroupEnd

; ── DAW 插件 ─────────────────────────────────────────────────────────
SectionGroup /e "$(HFS_GROUP_PLUGINS)"
Section "$(HFS_COMPONENT_VST3)"
  !ifdef HFS_HOST_GUARD
    Call HfsHostRunning
    ${If} $0 == 1
      MessageBox MB_OK|MB_ICONSTOP "$(HFS_ERR_HOST_OPEN)"
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
  ; 【为什么还要清别平台架构】dist\HiFiShifter.vst3 是跨平台共用的 bundle，早期
  ; 打包逻辑用 Copy-Item * -Recurse，把 Linux 构建留下的 x86_64-linux（约 57 MB
  ; 的 .so）一起装到了 Windows 上。File /r 只写不删，重装时这些旧目录会一直留着；
  ; 明确删掉已知的非 Windows 架构目录，才能让已经中招的用户升级后恢复干净。
  RMDir /r "$HfsVst3Dir\${VST3_SUBDIR}\Contents\x86_64-linux"
  RMDir /r "$HfsVst3Dir\${VST3_SUBDIR}\Contents\aarch64-linux"
  RMDir /r "$HfsVst3Dir\${VST3_SUBDIR}\Contents\MacOS"
  RMDir /r "$HfsVst3Dir\${VST3_SUBDIR}\Contents\Frameworks"
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
Section "$(HFS_COMPONENT_CLAP)"
  SetOutPath "$HfsClapDir"
  Delete "$HfsClapDir\${CLAP_FILENAME}"
  File "/oname=${CLAP_FILENAME}" "${CLAP_FILE}"
  WriteRegStr ${HFS_ARP_ROOT} "${ARP_KEY}" "ClapDir" "$HfsClapDir"
SectionEnd
SectionGroupEnd

; ── 组件悬停描述 ─────────────────────────────────────────────────────
; 【为什么这一段非有不可】MUI2 的组件页默认带描述框，但描述内容只来自
; .onMouseOverSection 回调；脚本此前完全没有 MUI_FUNCTION_DESCRIPTION_*，
; 于是悬停时描述框始终只显示 MUI 自带的静态提示（截图里的"将光标悬停…"）。
; 每个 section 与每个分组都登记一条，五语文案来自 installer_desc_* 词表。
!insertmacro MUI_FUNCTION_DESCRIPTION_BEGIN
  !insertmacro MUI_DESCRIPTION_TEXT ${SECGRP_APP} "$(HFS_DESC_GROUP_APP)"
  !insertmacro MUI_DESCRIPTION_TEXT ${SEC_APP} "$(HFS_DESC_APP)"
  !insertmacro MUI_DESCRIPTION_TEXT ${SEC_MODELS} "$(HFS_DESC_MODELS)"
  !insertmacro MUI_DESCRIPTION_TEXT ${SEC_SHORTCUTS} "$(HFS_DESC_SHORTCUTS)"
  !insertmacro MUI_DESCRIPTION_TEXT ${SECGRP_PLUGINS} "$(HFS_DESC_GROUP_PLUGINS)"
  !insertmacro MUI_DESCRIPTION_TEXT ${SEC_VST3} "$(HFS_DESC_VST3)"
  !insertmacro MUI_DESCRIPTION_TEXT ${SEC_CLAP} "$(HFS_DESC_CLAP)"
!insertmacro MUI_FUNCTION_DESCRIPTION_END


; ── 卸载选项页 ─────────────────────────────────────────────────────────
; 两个复选框默认都不勾选（un.onInit 置 0），用户想清理时主动勾上：
;   - 清理共享模型库（约 150 MB，所有安装共用）
;   - 清理缓存与日志（波形/渲染缓存、日志、WebView 数据、临时文件）
; 无论怎么勾，设置与未保存工程始终保留；若还有别的 HiFiShifter 安装，
; 这两项都不生效（避免卸载 App 把仍在用的插件数据扫掉）。
Function un.HfsUninstallOptionsCreate
  !insertmacro MUI_HEADER_TEXT "$(HFS_UNINSTALL_OPTIONS_TITLE)" "$(HFS_UNINSTALL_OPTIONS_SUBTITLE)"
  nsDialogs::Create 1018
  Pop $0
  ${If} $0 == error
    Abort
  ${EndIf}
  ${NSD_CreateLabel} 0 0 100% 24u "$(HFS_UNINSTALL_OPTIONS_TOP)"
  Pop $0
  ${NSD_CreateCheckbox} 0 30u 100% 12u "$(HFS_UNINSTALL_OPT_MODELS)"
  Pop $HfsUnChkModels
  ${NSD_CreateLabel} 12u 44u 88% 20u "$(HFS_UNINSTALL_OPT_MODELS_DESC)"
  Pop $0
  ${NSD_CreateCheckbox} 0 72u 100% 12u "$(HFS_UNINSTALL_OPT_CACHE)"
  Pop $HfsUnChkCache
  ${NSD_CreateLabel} 12u 86u 88% 20u "$(HFS_UNINSTALL_OPT_CACHE_DESC)"
  Pop $0
  ${If} $HfsUnOptModels == 1
    ${NSD_Check} $HfsUnChkModels
  ${EndIf}
  ${If} $HfsUnOptCache == 1
    ${NSD_Check} $HfsUnChkCache
  ${EndIf}
  nsDialogs::Show
FunctionEnd

Function un.HfsUninstallOptionsLeave
  ${NSD_GetState} $HfsUnChkModels $HfsUnOptModels
  ${NSD_GetState} $HfsUnChkCache $HfsUnOptCache
FunctionEnd

Section "Uninstall"
  ; 只删自己：$HfsVst3Dir 是 Common Files\VST3，里面还有别人的插件。
  RMDir /r "$HfsVst3Dir\${VST3_SUBDIR}"
  Delete "$HfsClapDir\${CLAP_FILENAME}"
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

  ; ── 可选清理：只有"没有别的 HiFiShifter 安装"时才可能生效 ──────────────
  ; App 与仍在使用的 VST3/CLAP 插件共享模型库与缓存目录；卸载 App 时不该把
  ; 插件还在用的数据扫掉。判定结果存进专用变量，避免被 LogicLib 宏复用 $0。
  Call un.HfsOtherInstallExists
  StrCpy $HfsUnOtherInstall $0

  ; ── 模型库：勾了「清理共享模型库」且没有别的安装才删 ──────────────────
  ; 【为什么删整个根，而不是只删 ${MODELS_VERSION}】安装时 HfsModelsGc 会刻意保留
  ; 当前版本 + 上一个版本，升级过的机器上必然有两个版本目录；失败的安装还会留下
  ; <版本>.staging。只删一个版本目录时，剩下的会让 RMDir "$HfsModelRoot" 永远失败
  ; —— 表现为"卸载了但模型文件还在"。整个根目录都是本产品写的，没有别的程序会在
  ; 这里放东西，因此整目录删除是安全的。
  ${If} $HfsUnOptModels == 1
  ${AndIf} $HfsUnOtherInstall == 0
    SetShellVarContext ${HFS_MODEL_ROOT_CONTEXT}
    StrCpy $HfsModelRoot "${HFS_MODEL_ROOT}"
    SetShellVarContext current
    ; 【为什么先逐子目录删，再删根】RMDir /r 遇到被占用或权限异常的子目录会**静默跳过**
    ; —— 整个卸载显示成功，模型文件却还留在磁盘上。先逐个删，某个失败也不影响其它。
    FindFirst $R3 $R4 "$HfsModelRoot\*"
    StrCmp $R4 "" hfs_models_children_done
    hfs_models_children_loop:
    StrCmp $R4 "." hfs_models_children_next
    StrCmp $R4 ".." hfs_models_children_next
    ${If} ${FileExists} "$HfsModelRoot\$R4\*"
      RMDir /r "$HfsModelRoot\$R4"
    ${Else}
      Delete "$HfsModelRoot\$R4"
    ${EndIf}
    hfs_models_children_next:
    ClearErrors
    FindNext $R3 $R4
    IfErrors hfs_models_children_done
    StrCmp $R4 "" hfs_models_children_done hfs_models_children_loop
    hfs_models_children_done:
    FindClose $R3
    RMDir /r "$HfsModelRoot"
    ; 删掉模型根的上级空壳（只在它真的空了才成功），两种模式都适用。
    RMDir "$HfsModelRoot\.."
  ${EndIf}

  ; ── 缓存与日志：勾了「清理缓存与日志」才删，按可再生成性分级 ──────────
  ; 【为什么不是"全删"也不是"全留"】%APPDATA%\HiFiShifter 里是设置、快捷键、
  ; 最近工程 —— 删了不可恢复；%LOCALAPPDATA%\HiFiShifter 里既有几 GB 可重新生成的
  ; 缓存/日志/WebView profile，也有用户尚未保存的工程素材与插件记事本。白名单只列
  ; app_paths.rs / temp_sweep.rs 明确定义为派生物的子目录，其余默认保留。
  ${If} $HfsUnOptCache == 1
  ${AndIf} $HfsUnOtherInstall == 0
    Call un.HfsSweepDisposableDirs
    DetailPrint "Removed caches, logs and temporary data. Settings and any unsaved work were kept in %APPDATA%\HiFiShifter and %LOCALAPPDATA%\HiFiShifter."
  ${ElseIf} $HfsUnOtherInstall == 1
    DetailPrint "Another HiFiShifter installation is still present; shared data was kept."
  ${Else}
    DetailPrint "Kept the shared model library and cached data as requested."
  ${EndIf}
SectionEnd

; 删除 %LOCALAPPDATA%\HiFiShifter 下**明确可再生成**的子目录。
; 这份清单与 backend/hifishifter-kernel/src/app_paths.rs 的 cache_dir()/log_dir()/
; local_data_subdir() 以及 hifishifter-plugin/src/editor/temp_sweep.rs 的回收目标一一对应：
;   cache / logs —— 缓存与日志（app_paths.rs 明确说"删掉只会变慢，不会丢数据"）
;   EBWebView / webview —— WebView2 用户数据目录（TempSweep 自己也在回收）
;   pcm / capture / transfer / links —— PCM 暂存、捕获、传输、host-link 工作区标记
;   plugin-content-v1 —— 宿主 PCM 缓存（host_pcm_cache.rs）
; 【刻意保留】unsaved（未保存工程素材）、plugin-notebook（插件记事本正文与附件）、
; ara-content-v1 / ara-instances（宿主写入的 ARA 内容）、以及任何源码未列出的新目录。
Function un.HfsSweepDisposableDirs
  SetShellVarContext current
  StrCpy $R0 "$LOCALAPPDATA\HiFiShifter"
  ${IfNot} ${FileExists} "$R0\*"
    Return
  ${EndIf}
  ; 逐个删：某个目录被占用/权限异常时 RMDir /r 会静默跳过，但不影响其它目录。
  ${If} ${FileExists} "$R0\cache\*"
    DetailPrint "Removing disposable data: cache"
    RMDir /r "$R0\cache"
  ${EndIf}
  ${If} ${FileExists} "$R0\logs\*"
    DetailPrint "Removing disposable data: logs"
    RMDir /r "$R0\logs"
  ${EndIf}
  ${If} ${FileExists} "$R0\EBWebView\*"
    DetailPrint "Removing disposable data: EBWebView"
    RMDir /r "$R0\EBWebView"
  ${EndIf}
  ${If} ${FileExists} "$R0\webview\*"
    DetailPrint "Removing disposable data: webview"
    RMDir /r "$R0\webview"
  ${EndIf}
  ${If} ${FileExists} "$R0\pcm\*"
    DetailPrint "Removing disposable data: pcm"
    RMDir /r "$R0\pcm"
  ${EndIf}
  ${If} ${FileExists} "$R0\capture\*"
    DetailPrint "Removing disposable data: capture"
    RMDir /r "$R0\capture"
  ${EndIf}
  ${If} ${FileExists} "$R0\transfer\*"
    DetailPrint "Removing disposable data: transfer"
    RMDir /r "$R0\transfer"
  ${EndIf}
  ${If} ${FileExists} "$R0\links\*"
    DetailPrint "Removing disposable data: links"
    RMDir /r "$R0\links"
  ${EndIf}
  ${If} ${FileExists} "$R0\plugin-content-v1\*"
    DetailPrint "Removing disposable data: plugin-content-v1"
    RMDir /r "$R0\plugin-content-v1"
  ${EndIf}
  ; 目录空了才删根；删不掉说明还有用户数据要留，RMDir 不带 /r 会安静地什么也不做。
  RMDir "$R0"
FunctionEnd

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
    ; 只认"其它形态"的键：本产品自己的键名（HiFiShifter）明确排除。
    ; 【为什么必须排除自己】DeleteRegKey 可能因权限失败没删掉自己的键；若把它也算作
    ; "还有别的安装"，模型库就会永远被保留 —— 正是"卸载了模型还在"的表现。
    ; 旧版插件安装器留下的 HiFiShifterVST3、以及将来拆出的其它形态才算数。
    ${If} $R3 == "HiFiShifter"
      ${Continue}
    ${EndIf}
    StrCpy $R4 $R3 11
    ${If} $R4 == "HiFiShifter"
      StrCpy $0 1
      ${ExitDo}
    ${EndIf}
  ${Loop}
FunctionEnd
