Unicode true
!include "MUI2.nsh"
!include "LogicLib.nsh"
!include "x64.nsh"
!include "StrFunc.nsh"
${StrStr}
Name "HiFiShifter VST3 ${PLUGIN_VERSION}"
OutFile "${OUTPUT_FILE}"
InstallDir "$COMMONFILES64\VST3"
RequestExecutionLevel admin
SetCompressor /SOLID lzma
ShowInstDetails show
!define MUI_ABORTWARNING
!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH
!insertmacro MUI_LANGUAGE "SimpChinese"

; 在选择目录前和写入前都检查，避免覆盖仍在 REAPER 中加载的插件。
Function CheckReaperClosed
  nsExec::ExecToStack /TIMEOUT=10000 '"$SYSDIR\tasklist.exe" /FO CSV /NH'
  Pop $0
  Pop $1
  ${If} $0 != 0
    MessageBox MB_OK|MB_ICONSTOP "无法确认 REAPER 已退出，请关闭 REAPER 后重新运行安装器。"
    Abort
  ${EndIf}
  ${StrStr} $2 $1 '"reaper.exe"'
  ${StrStr} $3 $1 '"reaper_host64.exe"'
  ${StrStr} $4 $1 '"reaper_host32.exe"'
  ${If} $2 != ""
  ${OrIf} $3 != ""
  ${OrIf} $4 != ""
    MessageBox MB_OK|MB_ICONSTOP "请完全退出 REAPER 及其插件宿主进程后再安装 HiFiShifter VST3。"
    Abort
  ${EndIf}
FunctionEnd

Function .onInit
  ${IfNot} ${RunningX64}
    MessageBox MB_OK|MB_ICONSTOP "此插件需要 64 位 Windows。"
    Abort
  ${EndIf}
  SetRegView 64
  Call CheckReaperClosed
FunctionEnd

Section "HiFiShifter VST3" SEC_PLUGIN
  Call CheckReaperClosed
  SetOutPath "$INSTDIR\HiFiShifter.vst3"
  File /r "${PLUGIN_BUNDLE}\*"
  SetOutPath "$INSTDIR\HiFiShifter.vst3\Contents\Resources"
  File "..\LICENSE"
SectionEnd
