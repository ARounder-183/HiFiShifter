; HiFiShifter VST3 安装器（NSIS）。
;
; 【多语言】文案**不在这里**：本文件只引用 `$(HFS_*)`，取值来自
; `tools/installer/installer_strings.nsh`，而那份又由
; `tools/build-installer-strings.mjs` 从五语词表抽出。要改文案请改词表。
;
; 【为什么语言声明在页面之后】MUI2 要求 `MUI_LANGUAGE` 排在 `MUI_[UN]PAGE_*`
; 之后（否则告警）。代价是 `MUI_WELCOMEPAGE_TEXT` / `MUI_UNCONFIRMPAGE_TEXT_TOP`
; 这类**编译期**读取的页面文案覆盖用不了 —— 它们必须在页面宏之前，而那时语言串还没
; 解析。项目特有的说明因此放在一个自定义页面上（`HfsNotesPage`）：它的函数体在语言
; 声明之后，`$(HFS_*)` 才取得到值。其余页面沿用 MUI 自带的本地化文案。
;
; 【为什么自定义页放在目录页之前】要告诉用户"装到 REAPER 会扫描的 VST3 目录"——
; MUI 自带的目录页文案是通用的，用户可能随手装到别处，然后发现 REAPER 找不到插件。
;
; 【为什么"需要 64 位 Windows"是五语并列】它在语言选择之前就要显示，那时
; `$LANGUAGE` 还没定，`$(HFS_*)` 取不到值。生成器为此给出宏 `HFS_FATAL_NOT_X64`。
Unicode true
!include "MUI2.nsh"
!include "LogicLib.nsh"
!include "nsDialogs.nsh"
!include "x64.nsh"
!include "StrFunc.nsh"
; 安装器与卸载器各自需要一份 StrStr（卸载器的函数前缀是 `un.`）。
${StrStr}
${UnStrStr}

!define PLUGIN_SUBDIR "HiFiShifter.vst3"
!define ARP_KEY "Software\Microsoft\Windows\CurrentVersion\Uninstall\HiFiShifterVST3"
!define UNINSTALLER "$INSTDIR\${PLUGIN_SUBDIR}\uninstall.exe"

; ── 本机验证开关（发布构建绝不定义）────────────────────────────────────
;
; 【为什么需要它】安装器真的会写 `Program Files\Common Files` 与 HKLM，而开发机上
; 通常没有管理员权限、REAPER 也可能正开着（那正是安装器要拒绝的情形）。没有这个
; 开关，"安装 → 查 ARP → 重装 → 卸载"这条闭环就完全无法在本机跑一遍 —— 而未经执行
; 的安装器正是最容易出错的东西。
;
; 打开后：装到当前用户的临时目录、不要求提权、ARP 写 HKCU、跳过 REAPER 检查。
; 其余逻辑（目录对账、卸载段、注册表字段、页面与文案）与发布构建逐字相同。
; `scripts/check-product-consistency.ps1` 断言默认分支仍是 admin / HKLM /
; Common Files —— 防止有人把测试分支误当默认。
!ifdef HFS_TEST_MODE
  InstallDir "$LOCALAPPDATA\HiFiShifterInstallerTest\VST3"
  RequestExecutionLevel user
  !define HFS_ARP_ROOT HKCU
!else
  InstallDir "$COMMONFILES64\VST3"
  RequestExecutionLevel admin
  !define HFS_ARP_ROOT HKLM
  !define HFS_REAPER_GUARD
!endif

; LICENSE 的路径由打包脚本以绝对路径传入（`/DHFS_LICENSE=…`）。默认值 `..\LICENSE`
; 只在从仓库的 `tools\` 目录编译时成立 —— 那种"靠相对位置猜"的写法在测试副本上
; 会指向不存在的文件，报出来的却只是一句 `no files found`。
!ifndef HFS_LICENSE
  !define HFS_LICENSE "..\LICENSE"
!endif

; `SetCompressor` 必须在语言声明之前：语言文件一加载就已经有数据被压缩，
; 之后再改压缩器会直接报错。
OutFile "${OUTPUT_FILE}"
SetCompressor /SOLID lzma
ShowInstDetails show
!define MUI_ABORTWARNING

; ── 页面 ────────────────────────────────────────────────────────────────
; 安装器与卸载器的页面都必须在语言声明之前（MUI 在加载第一个语言时插入界面函数，
; 那一刻它要知道有没有卸载器）。
!insertmacro MUI_PAGE_WELCOME
Page custom HfsNotesPage HfsNotesPageLeave
!insertmacro MUI_PAGE_DIRECTORY
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

Var HfsNotesDialog

; 项目说明页：装到哪、装之前要做什么。文案全部来自词表。
Function HfsNotesPage
  !insertmacro MUI_HEADER_TEXT "$(HFS_WELCOME_TITLE)" "$(HFS_DIR_TEXT)"
  nsDialogs::Create 1018
  Pop $HfsNotesDialog
  ${If} $HfsNotesDialog == error
    Abort
  ${EndIf}
  ${NSD_CreateLabel} 0 0 100% 100% "$(HFS_WELCOME_TEXT)"
  Pop $0
  nsDialogs::Show
FunctionEnd

Function HfsNotesPageLeave
FunctionEnd

; ── REAPER 运行检查 ─────────────────────────────────────────────────────
;
; 加载中的 DLL 会被宿主锁住：覆盖它要么失败、要么在 REAPER 里造成难以解释的崩溃。
; 安装与卸载**都要**检查（卸载时宿主还开着，目录删不干净且插件行为未定义）。
;
; 【为什么必须用 `/FI` 过滤】不能把整个 `tasklist` 输出取回来再找 `"reaper.exe"`：
; `nsExec::ExecToStack` 只保留输出的**前 1023 字节**，机器上进程一多，reaper.exe
; 就落在窗口之外 —— 检查会**静默失效**，用户照样能覆盖正在加载的插件。加上
; `/FI` 之后输出只剩一行，既不会截断，也不必依赖本地化的"未找到"提示文案。
; 三个进程名都查：REAPER 可能把插件放在独立的 host 进程里。
;
; 结果写进 `$0`：1 = 正在运行。
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

; 卸载器的同款检查（字符串函数前缀是 `un.`，因此必须另写一份）。
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

Function .onInit
  ${IfNot} ${RunningX64}
    ; 唯一一条无法本地化的提示：此时还不知道用户选了哪种语言。
    !insertmacro HFS_FATAL_NOT_X64
    Abort
  ${EndIf}
  SetRegView 64
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
  ; 【为什么不能直接用 `$INSTDIR`】卸载器会把自己复制到临时目录再运行，并传入
  ; `_?=` —— 而那个值是**卸载器所在目录**（`$INSTDIR\HiFiShifter.vst3`），不是安装
  ; 根目录。于是 `RMDir /r "$INSTDIR\HiFiShifter.vst3"` 会指向一个不存在的路径，
  ; 卸载**静默地什么都不删**（实测：注册表项被删掉、目录原封不动）。从安装时写下的
  ; `InstallLocation` 读回来才是权威值；读不到时退回卸载器所在目录的上一层。
  ReadRegStr $INSTDIR ${HFS_ARP_ROOT} "${ARP_KEY}" "InstallLocation"
  ${If} $INSTDIR == ""
    ; 卸载器住在 `$INSTDIR\HiFiShifter.vst3` 里，所以上一层就是安装根目录。
    ; 用字面量 `..` 而不是引入 FileFunc.nsh 的 GetParent：Windows 路径解析本来
    ; 就接受 `..`，少一个 include 就少一处"宏没声明"的编译错误。
    StrCpy $INSTDIR "$EXEDIR\.."
  ${EndIf}

  !ifdef HFS_REAPER_GUARD
    Call un.HfsReaperRunning
    ${If} $0 == 1
      MessageBox MB_OK|MB_ICONSTOP "$(HFS_UNINSTALL_REAPER_OPEN)"
      Abort
    ${EndIf}
  !endif
FunctionEnd

Section "$(HFS_NAME)" SEC_PLUGIN
  ; 写入前再查一次：用户在向导里停留期间可能又打开了 REAPER。
  !ifdef HFS_REAPER_GUARD
    Call HfsReaperRunning
    ${If} $0 == 1
      MessageBox MB_OK|MB_ICONSTOP "$(HFS_ERR_REAPER_OPEN)"
      Abort
    ${EndIf}
  !endif

  DetailPrint "$(HFS_INSTALLING)"

  ; ── 目录对账：先清掉上一代产物，再写入 ──────────────────────────────
  ;
  ; 【为什么必须清】Vite 给每个 chunk 加了内容哈希（`main-D6fulCAv.js`），内容一变
  ; 文件名就变；而 `File /r` 只写不删。于是每次重装都会再堆一层，目录单调增长
  ; （实测一台机器上 `assets` 已从 20 个涨到 36 个，新旧两代并存）。
  ;
  ; 【为什么删整目录而不是逐文件比对】`File /r` 之后无法可靠区分"我刚写的"和
  ; "上一代遗留的"；而这两个目录是**纯构建产出**，删除是幂等且安全的。逐文件比对
  ; 要维护一份清单，反而更容易出错。
  ;
  ; 【为什么不动 `models\`】用户可能自行替换模型（换声码器等）。那是数据不是产物。
  RMDir /r "$INSTDIR\${PLUGIN_SUBDIR}\Contents\Resources\frontend"
  RMDir /r "$INSTDIR\${PLUGIN_SUBDIR}\Contents\x86_64-win"
  ; 早期版本把构建文档与独立 App 的页面一起塞进了 Resources；它们已经不再交付，
  ; 但**已安装**的副本不会自己消失，因此在这里顺手清掉。
  Delete "$INSTDIR\${PLUGIN_SUBDIR}\Contents\Resources\VST3-BUILD.md"
  Delete "$INSTDIR\${PLUGIN_SUBDIR}\Contents\Resources\frontend\index.html"
  Delete "$INSTDIR\${PLUGIN_SUBDIR}\Contents\Resources\frontend\detached.html"
  Delete "$INSTDIR\${PLUGIN_SUBDIR}\Contents\Resources\frontend\waveform-test.html"
  Delete "$INSTDIR\${PLUGIN_SUBDIR}\Contents\Resources\frontend\vite.svg"

  SetOutPath "$INSTDIR\${PLUGIN_SUBDIR}"
  File /r "${PLUGIN_BUNDLE}\*"
  SetOutPath "$INSTDIR\${PLUGIN_SUBDIR}\Contents\Resources"
  File "${HFS_LICENSE}"

  ; ── 卸载信息 ────────────────────────────────────────────────────────
  ;
  ; 【为什么必须写】没有这一段，Windows「程序和功能 / 已安装的应用」里根本不会出现
  ; HiFiShifter —— 用户唯一的卸载办法是手动删文件夹。`DisplayVersion` 用产品版本号
  ; （与独立 App 同源），将来才能做"已装版本"检测；`InstallLocation` 是卸载器唯一
  ; 可信的安装位置来源（见 `un.onInit` 的说明）。
  WriteUninstaller "${UNINSTALLER}"
  WriteRegStr ${HFS_ARP_ROOT} "${ARP_KEY}" "DisplayName" "$(HFS_NAME)"
  WriteRegStr ${HFS_ARP_ROOT} "${ARP_KEY}" "DisplayVersion" "${PLUGIN_VERSION}"
  WriteRegStr ${HFS_ARP_ROOT} "${ARP_KEY}" "Publisher" "$(HFS_PUBLISHER)"
  WriteRegStr ${HFS_ARP_ROOT} "${ARP_KEY}" "InstallLocation" "$INSTDIR"
  ; 不带 `_?=`：让卸载器按默认行为把自己复制到临时目录再运行，从而能删掉自身。
  WriteRegStr ${HFS_ARP_ROOT} "${ARP_KEY}" "UninstallString" '"${UNINSTALLER}"'
  WriteRegStr ${HFS_ARP_ROOT} "${ARP_KEY}" "QuietUninstallString" '"${UNINSTALLER}" /S'
  WriteRegDWORD ${HFS_ARP_ROOT} "${ARP_KEY}" "NoModify" 1
  WriteRegDWORD ${HFS_ARP_ROOT} "${ARP_KEY}" "NoRepair" 1
  ${If} ${FileExists} "$INSTDIR\${PLUGIN_SUBDIR}\Contents\Resources\icon.ico"
    WriteRegStr ${HFS_ARP_ROOT} "${ARP_KEY}" "DisplayIcon" "$INSTDIR\${PLUGIN_SUBDIR}\Contents\Resources\icon.ico"
  ${EndIf}
SectionEnd

Section "Uninstall"
  ; 只删自己这一层：`$INSTDIR` 是 `Common Files\VST3`，里面还有别人的插件。
  RMDir /r "$INSTDIR\${PLUGIN_SUBDIR}"
  DeleteRegKey ${HFS_ARP_ROOT} "${ARP_KEY}"

  ; 【刻意不删的东西】
  ; - `%APPDATA%\com.arounder.hifishifter\HiFiShifter`：用户设置与独立 App 共用，
  ;   删了会连带清掉 App 的配置；
  ; - 共享模型库：同样被独立 App 使用。误删 150 MB 比多占 150 MB 严重得多。
  ; 两者都不在这里动，只在卸载详情里留下说明。
  DetailPrint "Settings and the shared model library were kept."
SectionEnd
