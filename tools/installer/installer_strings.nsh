; 本文件由 tools/build-installer-strings.mjs 生成 —— 不要手工编辑。
;
; 文案的唯一来源是 frontend/src/i18n/*.ts 里 installer_ 前缀的词条；
; 直接改这里会在下一次生成时被覆盖，且 scripts/check-product-consistency.ps1
; 会因与本文件不一致而失败。

; 语言：en-US→English、zh-CN→SimpChinese、zh-TW→TradChinese、ja-JP→Japanese、ko-KR→Korean
; 词条：10 条 × 5 种语言

; installer_dir_text
LangString HFS_DIR_TEXT ${LANG_ENGLISH} "Choose the VST3 folder that REAPER scans. The plug-in is installed as a HiFiShifter.vst3 folder inside it."
LangString HFS_DIR_TEXT ${LANG_SIMPCHINESE} "请选择 REAPER 会扫描的 VST3 文件夹。插件会以 HiFiShifter.vst3 文件夹的形式装在其中。"
LangString HFS_DIR_TEXT ${LANG_TRADCHINESE} "請選擇 REAPER 會掃描的 VST3 資料夾。外掛會以 HiFiShifter.vst3 資料夾的形式安裝在其中。"
LangString HFS_DIR_TEXT ${LANG_JAPANESE} "REAPER がスキャンする VST3 フォルダーを選んでください。プラグインは HiFiShifter.vst3 フォルダーとしてインストールされます。"
LangString HFS_DIR_TEXT ${LANG_KOREAN} "REAPER가 검사하는 VST3 폴더를 선택하세요. 플러그인은 HiFiShifter.vst3 폴더로 설치됩니다."

; installer_err_not_x64
LangString HFS_ERR_NOT_X64 ${LANG_ENGLISH} "This plug-in requires 64-bit Windows."
LangString HFS_ERR_NOT_X64 ${LANG_SIMPCHINESE} "此插件需要 64 位 Windows。"
LangString HFS_ERR_NOT_X64 ${LANG_TRADCHINESE} "此外掛需要 64 位元 Windows。"
LangString HFS_ERR_NOT_X64 ${LANG_JAPANESE} "このプラグインには 64 ビット版 Windows が必要です。"
LangString HFS_ERR_NOT_X64 ${LANG_KOREAN} "이 플러그인은 64비트 Windows가 필요합니다."

; installer_err_reaper_open
LangString HFS_ERR_REAPER_OPEN ${LANG_ENGLISH} "Quit REAPER and its plug-in host processes before installing HiFiShifter VST3."
LangString HFS_ERR_REAPER_OPEN ${LANG_SIMPCHINESE} "请先完全退出 REAPER 及其插件宿主进程，再安装 HiFiShifter VST3。"
LangString HFS_ERR_REAPER_OPEN ${LANG_TRADCHINESE} "請先完全結束 REAPER 及其外掛裝載程序，再安裝 HiFiShifter VST3。"
LangString HFS_ERR_REAPER_OPEN ${LANG_JAPANESE} "HiFiShifter VST3 をインストールする前に、REAPER とそのプラグインホストを終了してください。"
LangString HFS_ERR_REAPER_OPEN ${LANG_KOREAN} "HiFiShifter VST3를 설치하기 전에 REAPER와 플러그인 호스트 프로세스를 종료하세요."

; installer_err_tasklist
LangString HFS_ERR_TASKLIST ${LANG_ENGLISH} "Could not confirm that REAPER has exited. Close REAPER and run the installer again."
LangString HFS_ERR_TASKLIST ${LANG_SIMPCHINESE} "无法确认 REAPER 已退出。请关闭 REAPER 后重新运行安装程序。"
LangString HFS_ERR_TASKLIST ${LANG_TRADCHINESE} "無法確認 REAPER 已結束。請關閉 REAPER 後重新執行安裝程式。"
LangString HFS_ERR_TASKLIST ${LANG_JAPANESE} "REAPER が終了したことを確認できませんでした。REAPER を閉じてからインストーラーを再実行してください。"
LangString HFS_ERR_TASKLIST ${LANG_KOREAN} "REAPER가 종료되었는지 확인할 수 없습니다. REAPER를 닫고 설치 프로그램을 다시 실행하세요."

; installer_installing
LangString HFS_INSTALLING ${LANG_ENGLISH} "Installing HiFiShifter VST3..."
LangString HFS_INSTALLING ${LANG_SIMPCHINESE} "正在安装 HiFiShifter VST3..."
LangString HFS_INSTALLING ${LANG_TRADCHINESE} "正在安裝 HiFiShifter VST3..."
LangString HFS_INSTALLING ${LANG_JAPANESE} "HiFiShifter VST3 をインストールしています..."
LangString HFS_INSTALLING ${LANG_KOREAN} "HiFiShifter VST3 설치 중..."

; installer_name
LangString HFS_NAME ${LANG_ENGLISH} "HiFiShifter VST3"
LangString HFS_NAME ${LANG_SIMPCHINESE} "HiFiShifter VST3"
LangString HFS_NAME ${LANG_TRADCHINESE} "HiFiShifter VST3"
LangString HFS_NAME ${LANG_JAPANESE} "HiFiShifter VST3"
LangString HFS_NAME ${LANG_KOREAN} "HiFiShifter VST3"

; installer_publisher
LangString HFS_PUBLISHER ${LANG_ENGLISH} "ARounder"
LangString HFS_PUBLISHER ${LANG_SIMPCHINESE} "ARounder"
LangString HFS_PUBLISHER ${LANG_TRADCHINESE} "ARounder"
LangString HFS_PUBLISHER ${LANG_JAPANESE} "ARounder"
LangString HFS_PUBLISHER ${LANG_KOREAN} "ARounder"

; installer_uninstall_reaper_open
LangString HFS_UNINSTALL_REAPER_OPEN ${LANG_ENGLISH} "Quit REAPER before uninstalling HiFiShifter VST3."
LangString HFS_UNINSTALL_REAPER_OPEN ${LANG_SIMPCHINESE} "请先退出 REAPER，再卸载 HiFiShifter VST3。"
LangString HFS_UNINSTALL_REAPER_OPEN ${LANG_TRADCHINESE} "請先結束 REAPER，再卸載 HiFiShifter VST3。"
LangString HFS_UNINSTALL_REAPER_OPEN ${LANG_JAPANESE} "HiFiShifter VST3 をアンインストールする前に REAPER を終了してください。"
LangString HFS_UNINSTALL_REAPER_OPEN ${LANG_KOREAN} "HiFiShifter VST3를 제거하기 전에 REAPER를 종료하세요."

; installer_welcome_text
LangString HFS_WELCOME_TEXT ${LANG_ENGLISH} "This installs the HiFiShifter VST3/ARA plug-in into the VST3 folder you choose.$\r$\nClose REAPER first: the installer must not replace a plug-in that is currently loaded.$\r$\nSettings are shared with the standalone app."
LangString HFS_WELCOME_TEXT ${LANG_SIMPCHINESE} "安装程序会把 HiFiShifter VST3/ARA 插件放进你选择的 VST3 文件夹。$\r$\n请先关闭 REAPER：安装程序不会覆盖正在被加载的插件。$\r$\n设置与独立 App 共用。"
LangString HFS_WELCOME_TEXT ${LANG_TRADCHINESE} "安裝程式會把 HiFiShifter VST3/ARA 外掛放進你選擇的 VST3 資料夾。$\r$\n請先關閉 REAPER：安裝程式不會覆寫正在被載入的外掛。$\r$\n設定與獨立 App 共用。"
LangString HFS_WELCOME_TEXT ${LANG_JAPANESE} "選択した VST3 フォルダーに HiFiShifter VST3/ARA プラグインをインストールします。$\r$\n先に REAPER を終了してください。読み込み中のプラグインは上書きされません。$\r$\n設定は単体アプリと共有されます。"
LangString HFS_WELCOME_TEXT ${LANG_KOREAN} "선택한 VST3 폴더에 HiFiShifter VST3/ARA 플러그인을 설치합니다.$\r$\n먼저 REAPER를 종료하세요. 로드 중인 플러그인은 덮어쓰지 않습니다.$\r$\n설정은 독립 실행형 앱과 공유됩니다."

; installer_welcome_title
LangString HFS_WELCOME_TITLE ${LANG_ENGLISH} "Welcome to the HiFiShifter VST3 Setup"
LangString HFS_WELCOME_TITLE ${LANG_SIMPCHINESE} "欢迎安装 HiFiShifter VST3"
LangString HFS_WELCOME_TITLE ${LANG_TRADCHINESE} "歡迎安裝 HiFiShifter VST3"
LangString HFS_WELCOME_TITLE ${LANG_JAPANESE} "HiFiShifter VST3 セットアップへようこそ"
LangString HFS_WELCOME_TITLE ${LANG_KOREAN} "HiFiShifter VST3 설치를 시작합니다"

; 「需要 64 位 Windows」是唯一在语言选择**之前**就要显示的提示：
; 那时 $LANGUAGE 还没定，$(HFS_...) 取不到值，只能五种语言并列。
!macro HFS_FATAL_NOT_X64
  ; en-US
  StrCpy $0 "$0This plug-in requires 64-bit Windows.$\r$\n"
  ; zh-CN
  StrCpy $0 "$0此插件需要 64 位 Windows。$\r$\n"
  ; zh-TW
  StrCpy $0 "$0此外掛需要 64 位元 Windows。$\r$\n"
  ; ja-JP
  StrCpy $0 "$0このプラグインには 64 ビット版 Windows が必要です。$\r$\n"
  ; ko-KR
  StrCpy $0 "$0이 플러그인은 64비트 Windows가 필요합니다.$\r$\n"
  MessageBox MB_OK|MB_ICONSTOP "$0"
!macroend
