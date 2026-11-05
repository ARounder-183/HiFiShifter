; 本文件由 scripts/release/build-installer-strings.mjs 生成 —— 不要手工编辑。
;
; 文案的唯一来源是 frontend/src/i18n/*.ts 里 installer_ 前缀的词条；
; 直接改这里会在下一次生成时被覆盖，且 scripts/checks/check-product-consistency.ps1
; 会因与本文件不一致而失败。

; 语言：en-US→English、zh-CN→SimpChinese、zh-TW→TradChinese、ja-JP→Japanese、ko-KR→Korean
; 词条：23 条 × 5 种语言

; installer_app_dir_text
LangString HFS_APP_DIR_TEXT ${LANG_ENGLISH} "Choose where the standalone app is installed."
LangString HFS_APP_DIR_TEXT ${LANG_SIMPCHINESE} "请选择独立 App 的安装位置。"
LangString HFS_APP_DIR_TEXT ${LANG_TRADCHINESE} "請選擇獨立 App 的安裝位置。"
LangString HFS_APP_DIR_TEXT ${LANG_JAPANESE} "スタンドアロン版のインストール先を選んでください。"
LangString HFS_APP_DIR_TEXT ${LANG_KOREAN} "독립 실행형 앱의 설치 위치를 선택하세요."

; installer_clap_dir_text
LangString HFS_CLAP_DIR_TEXT ${LANG_ENGLISH} "Choose the CLAP folder scanned by your DAW. HiFiShifter.clap is installed as a single file inside it."
LangString HFS_CLAP_DIR_TEXT ${LANG_SIMPCHINESE} "请选择 DAW 会扫描的 CLAP 文件夹。插件会以 HiFiShifter.clap 单文件的形式装在其中。"
LangString HFS_CLAP_DIR_TEXT ${LANG_TRADCHINESE} "請選擇 DAW 會掃描的 CLAP 資料夾。外掛會以 HiFiShifter.clap 單檔的形式裝在其中。"
LangString HFS_CLAP_DIR_TEXT ${LANG_JAPANESE} "DAW がスキャンする CLAP フォルダーを選んでください。プラグインは HiFiShifter.clap の単一ファイルとしてインストールされます。"
LangString HFS_CLAP_DIR_TEXT ${LANG_KOREAN} "DAW가 검사하는 CLAP 폴더를 선택하세요. 플러그인은 HiFiShifter.clap 단일 파일로 설치됩니다."

; installer_component_app
LangString HFS_COMPONENT_APP ${LANG_ENGLISH} "HiFiShifter (standalone app)"
LangString HFS_COMPONENT_APP ${LANG_SIMPCHINESE} "HiFiShifter（独立 App）"
LangString HFS_COMPONENT_APP ${LANG_TRADCHINESE} "HiFiShifter（獨立 App）"
LangString HFS_COMPONENT_APP ${LANG_JAPANESE} "HiFiShifter（スタンドアロン版）"
LangString HFS_COMPONENT_APP ${LANG_KOREAN} "HiFiShifter(독립 실행형 앱)"

; installer_component_clap
LangString HFS_COMPONENT_CLAP ${LANG_ENGLISH} "CLAP plug-in (Capture-only)"
LangString HFS_COMPONENT_CLAP ${LANG_SIMPCHINESE} "CLAP 插件（仅 Capture）"
LangString HFS_COMPONENT_CLAP ${LANG_TRADCHINESE} "CLAP 外掛（僅 Capture）"
LangString HFS_COMPONENT_CLAP ${LANG_JAPANESE} "CLAP プラグイン（Capture のみ）"
LangString HFS_COMPONENT_CLAP ${LANG_KOREAN} "CLAP 플러그인(Capture 전용)"

; installer_component_models
LangString HFS_COMPONENT_MODELS ${LANG_ENGLISH} "Shared model library"
LangString HFS_COMPONENT_MODELS ${LANG_SIMPCHINESE} "共享模型库"
LangString HFS_COMPONENT_MODELS ${LANG_TRADCHINESE} "共用模型庫"
LangString HFS_COMPONENT_MODELS ${LANG_JAPANESE} "共有モデルライブラリ"
LangString HFS_COMPONENT_MODELS ${LANG_KOREAN} "공유 모델 라이브러리"

; installer_component_models_install
LangString HFS_COMPONENT_MODELS_INSTALL ${LANG_ENGLISH} "Installing the shared model library..."
LangString HFS_COMPONENT_MODELS_INSTALL ${LANG_SIMPCHINESE} "正在安装共享模型库..."
LangString HFS_COMPONENT_MODELS_INSTALL ${LANG_TRADCHINESE} "正在安裝共用模型庫..."
LangString HFS_COMPONENT_MODELS_INSTALL ${LANG_JAPANESE} "共有モデルライブラリをインストールしています..."
LangString HFS_COMPONENT_MODELS_INSTALL ${LANG_KOREAN} "공유 모델 라이브러리 설치 중..."

; installer_component_models_skip
LangString HFS_COMPONENT_MODELS_SKIP ${LANG_ENGLISH} "The shared model library is already up to date; skipping."
LangString HFS_COMPONENT_MODELS_SKIP ${LANG_SIMPCHINESE} "共享模型库已是最新，跳过。"
LangString HFS_COMPONENT_MODELS_SKIP ${LANG_TRADCHINESE} "共用模型庫已是最新，略過。"
LangString HFS_COMPONENT_MODELS_SKIP ${LANG_JAPANESE} "共有モデルライブラリは最新です。スキップします。"
LangString HFS_COMPONENT_MODELS_SKIP ${LANG_KOREAN} "공유 모델 라이브러리가 최신입니다. 건너뜁니다."

; installer_component_shortcuts
LangString HFS_COMPONENT_SHORTCUTS ${LANG_ENGLISH} "Start menu and desktop shortcuts"
LangString HFS_COMPONENT_SHORTCUTS ${LANG_SIMPCHINESE} "开始菜单与桌面快捷方式"
LangString HFS_COMPONENT_SHORTCUTS ${LANG_TRADCHINESE} "開始功能表與桌面捷徑"
LangString HFS_COMPONENT_SHORTCUTS ${LANG_JAPANESE} "スタートメニューとデスクトップのショートカット"
LangString HFS_COMPONENT_SHORTCUTS ${LANG_KOREAN} "시작 메뉴 및 바탕 화면 바로 가기"

; installer_component_vst3
LangString HFS_COMPONENT_VST3 ${LANG_ENGLISH} "VST3 / ARA plug-in"
LangString HFS_COMPONENT_VST3 ${LANG_SIMPCHINESE} "VST3 / ARA 插件"
LangString HFS_COMPONENT_VST3 ${LANG_TRADCHINESE} "VST3 / ARA 外掛"
LangString HFS_COMPONENT_VST3 ${LANG_JAPANESE} "VST3 / ARA プラグイン"
LangString HFS_COMPONENT_VST3 ${LANG_KOREAN} "VST3 / ARA 플러그인"

; installer_dir_text
LangString HFS_DIR_TEXT ${LANG_ENGLISH} "Choose the VST3 folder that REAPER scans. The plug-in is installed as a HiFiShifter.vst3 folder inside it."
LangString HFS_DIR_TEXT ${LANG_SIMPCHINESE} "请选择 REAPER 会扫描的 VST3 文件夹。插件会以 HiFiShifter.vst3 文件夹的形式装在其中。"
LangString HFS_DIR_TEXT ${LANG_TRADCHINESE} "請選擇 REAPER 會掃描的 VST3 資料夾。外掛會以 HiFiShifter.vst3 資料夾的形式安裝在其中。"
LangString HFS_DIR_TEXT ${LANG_JAPANESE} "REAPER がスキャンする VST3 フォルダーを選んでください。プラグインは HiFiShifter.vst3 フォルダーとしてインストールされます。"
LangString HFS_DIR_TEXT ${LANG_KOREAN} "REAPER가 검사하는 VST3 폴더를 선택하세요. 플러그인은 HiFiShifter.vst3 폴더로 설치됩니다."

; installer_err_app_running
LangString HFS_ERR_APP_RUNNING ${LANG_ENGLISH} "HiFiShifter is running. Close it and run the installer again."
LangString HFS_ERR_APP_RUNNING ${LANG_SIMPCHINESE} "HiFiShifter 正在运行。请关闭它后重新运行安装程序。"
LangString HFS_ERR_APP_RUNNING ${LANG_TRADCHINESE} "HiFiShifter 正在執行。請關閉它後重新執行安裝程式。"
LangString HFS_ERR_APP_RUNNING ${LANG_JAPANESE} "HiFiShifter が実行中です。終了してからインストーラーを再実行してください。"
LangString HFS_ERR_APP_RUNNING ${LANG_KOREAN} "HiFiShifter가 실행 중입니다. 종료한 뒤 설치 프로그램을 다시 실행하세요."

; installer_err_not_x64
LangString HFS_ERR_NOT_X64 ${LANG_ENGLISH} "HiFiShifter requires 64-bit Windows."
LangString HFS_ERR_NOT_X64 ${LANG_SIMPCHINESE} "HiFiShifter 需要 64 位 Windows。"
LangString HFS_ERR_NOT_X64 ${LANG_TRADCHINESE} "HiFiShifter 需要 64 位元 Windows。"
LangString HFS_ERR_NOT_X64 ${LANG_JAPANESE} "HiFiShifter には 64 ビット版 Windows が必要です。"
LangString HFS_ERR_NOT_X64 ${LANG_KOREAN} "HiFiShifter는 64비트 Windows가 필요합니다."

; installer_err_reaper_open
LangString HFS_ERR_REAPER_OPEN ${LANG_ENGLISH} "Quit REAPER and its plug-in host processes before installing HiFiShifter."
LangString HFS_ERR_REAPER_OPEN ${LANG_SIMPCHINESE} "请先完全退出 REAPER 及其插件宿主进程，再安装 HiFiShifter。"
LangString HFS_ERR_REAPER_OPEN ${LANG_TRADCHINESE} "請先完全結束 REAPER 及其外掛裝載程序，再安裝 HiFiShifter。"
LangString HFS_ERR_REAPER_OPEN ${LANG_JAPANESE} "HiFiShifter をインストールする前に、REAPER とそのプラグインホストを終了してください。"
LangString HFS_ERR_REAPER_OPEN ${LANG_KOREAN} "HiFiShifter를 설치하기 전에 REAPER와 플러그인 호스트 프로세스를 종료하세요."

; installer_err_tasklist
LangString HFS_ERR_TASKLIST ${LANG_ENGLISH} "Could not confirm that REAPER has exited. Close REAPER and run the installer again."
LangString HFS_ERR_TASKLIST ${LANG_SIMPCHINESE} "无法确认 REAPER 已退出。请关闭 REAPER 后重新运行安装程序。"
LangString HFS_ERR_TASKLIST ${LANG_TRADCHINESE} "無法確認 REAPER 已結束。請關閉 REAPER 後重新執行安裝程式。"
LangString HFS_ERR_TASKLIST ${LANG_JAPANESE} "REAPER が終了したことを確認できませんでした。REAPER を閉じてからインストーラーを再実行してください。"
LangString HFS_ERR_TASKLIST ${LANG_KOREAN} "REAPER가 종료되었는지 확인할 수 없습니다. REAPER를 닫고 설치 프로그램을 다시 실행하세요."

; installer_installing
LangString HFS_INSTALLING ${LANG_ENGLISH} "Installing HiFiShifter..."
LangString HFS_INSTALLING ${LANG_SIMPCHINESE} "正在安装 HiFiShifter..."
LangString HFS_INSTALLING ${LANG_TRADCHINESE} "正在安裝 HiFiShifter..."
LangString HFS_INSTALLING ${LANG_JAPANESE} "HiFiShifter をインストールしています..."
LangString HFS_INSTALLING ${LANG_KOREAN} "HiFiShifter 설치 중..."

; installer_models_verify_failed
LangString HFS_MODELS_VERIFY_FAILED ${LANG_ENGLISH} "The shared model library could not be verified after copying. Run the installer again; if it keeps failing, check the free space on your system drive."
LangString HFS_MODELS_VERIFY_FAILED ${LANG_SIMPCHINESE} "共享模型库复制后校验失败。请重新运行安装程序；若反复失败，请检查系统盘剩余空间。"
LangString HFS_MODELS_VERIFY_FAILED ${LANG_TRADCHINESE} "共用模型庫複製後驗證失敗。請重新執行安裝程式；若反覆失敗，請檢查系統碟剩餘空間。"
LangString HFS_MODELS_VERIFY_FAILED ${LANG_JAPANESE} "共有モデルライブラリのコピー後の検証に失敗しました。もう一度インストーラーを実行してください。繰り返し失敗する場合はシステムドライブの空き容量を確認してください。"
LangString HFS_MODELS_VERIFY_FAILED ${LANG_KOREAN} "공유 모델 라이브러리를 복사한 뒤 검증에 실패했습니다. 설치 프로그램을 다시 실행하세요. 계속 실패하면 시스템 드라이브의 여유 공간을 확인하세요."

; installer_name
LangString HFS_NAME ${LANG_ENGLISH} "HiFiShifter"
LangString HFS_NAME ${LANG_SIMPCHINESE} "HiFiShifter"
LangString HFS_NAME ${LANG_TRADCHINESE} "HiFiShifter"
LangString HFS_NAME ${LANG_JAPANESE} "HiFiShifter"
LangString HFS_NAME ${LANG_KOREAN} "HiFiShifter"

; installer_publisher
LangString HFS_PUBLISHER ${LANG_ENGLISH} "ARounder"
LangString HFS_PUBLISHER ${LANG_SIMPCHINESE} "ARounder"
LangString HFS_PUBLISHER ${LANG_TRADCHINESE} "ARounder"
LangString HFS_PUBLISHER ${LANG_JAPANESE} "ARounder"
LangString HFS_PUBLISHER ${LANG_KOREAN} "ARounder"

; installer_repair_done
LangString HFS_REPAIR_DONE ${LANG_ENGLISH} "Repair finished. The shared model library is complete."
LangString HFS_REPAIR_DONE ${LANG_SIMPCHINESE} "修复完成。共享模型库已齐全。"
LangString HFS_REPAIR_DONE ${LANG_TRADCHINESE} "修復完成。共用模型庫已齊全。"
LangString HFS_REPAIR_DONE ${LANG_JAPANESE} "修復が完了しました。共有モデルライブラリは揃っています。"
LangString HFS_REPAIR_DONE ${LANG_KOREAN} "복구가 완료되었습니다. 공유 모델 라이브러리가 갖춰졌습니다."

; installer_uninstall_reaper_open
LangString HFS_UNINSTALL_REAPER_OPEN ${LANG_ENGLISH} "Quit REAPER before uninstalling HiFiShifter."
LangString HFS_UNINSTALL_REAPER_OPEN ${LANG_SIMPCHINESE} "请先退出 REAPER，再卸载 HiFiShifter。"
LangString HFS_UNINSTALL_REAPER_OPEN ${LANG_TRADCHINESE} "請先結束 REAPER，再卸載 HiFiShifter。"
LangString HFS_UNINSTALL_REAPER_OPEN ${LANG_JAPANESE} "HiFiShifter をアンインストールする前に REAPER を終了してください。"
LangString HFS_UNINSTALL_REAPER_OPEN ${LANG_KOREAN} "HiFiShifter를 제거하기 전에 REAPER를 종료하세요."

; installer_webview2_install_failed
LangString HFS_WEBVIEW2_INSTALL_FAILED ${LANG_ENGLISH} "The WebView2 runtime could not be installed. Install it from Microsoft's website and run the installer again."
LangString HFS_WEBVIEW2_INSTALL_FAILED ${LANG_SIMPCHINESE} "WebView2 运行时安装失败。请从微软官网安装后再运行安装程序。"
LangString HFS_WEBVIEW2_INSTALL_FAILED ${LANG_TRADCHINESE} "WebView2 執行階段安裝失敗。請從微軟官網安裝後再執行安裝程式。"
LangString HFS_WEBVIEW2_INSTALL_FAILED ${LANG_JAPANESE} "WebView2 ランタイムをインストールできませんでした。Microsoft のサイトからインストールしてから再実行してください。"
LangString HFS_WEBVIEW2_INSTALL_FAILED ${LANG_KOREAN} "WebView2 런타임을 설치하지 못했습니다. Microsoft 웹사이트에서 설치한 뒤 다시 실행하세요."

; installer_welcome_text
LangString HFS_WELCOME_TEXT ${LANG_ENGLISH} "This installs the HiFiShifter standalone app and the VST3/ARA plug-in. Both share one model library, so the 150 MB of weights is written once.$\r$\nClose REAPER and HiFiShifter before continuing: the installer must not replace files that are currently loaded.$\r$\nSettings, caches and logs live in your user profile - uninstalling keeps them.$\r$\nThe bundled NSF-HiFiGAN vocoder weights are CC BY-NC-SA 4.0 - non-commercial use only. License texts are installed as LICENSE and the licenses folder."
LangString HFS_WELCOME_TEXT ${LANG_SIMPCHINESE} "安装程序会装上 HiFiShifter 独立 App 与 VST3/ARA 插件。两者共用同一份模型库，150 MB 权重只写一遍。$\r$\n继续之前请先关闭 REAPER 与 HiFiShifter：安装程序不会覆盖正在被加载的文件。$\r$\n设置、缓存与日志位于你的用户目录，卸载时会保留。$\r$\n随包分发的 NSF-HiFiGAN 声码器权重为 CC BY-NC-SA 4.0 授权，仅限非商业使用；许可全文安装在 LICENSE 与 licenses 文件夹中。"
LangString HFS_WELCOME_TEXT ${LANG_TRADCHINESE} "安裝程式會裝上 HiFiShifter 獨立 App 與 VST3/ARA 外掛。兩者共用同一份模型庫，150 MB 權重只寫一次。$\r$\n繼續之前請先關閉 REAPER 與 HiFiShifter：安裝程式不會覆寫正在被載入的檔案。$\r$\n設定、快取與日誌位於你的使用者目錄，卸載時會保留。$\r$\n隨套件分發的 NSF-HiFiGAN 聲碼器權重為 CC BY-NC-SA 4.0 授權，僅限非商業使用；授權全文安裝在 LICENSE 與 licenses 資料夾中。"
LangString HFS_WELCOME_TEXT ${LANG_JAPANESE} "HiFiShifter のスタンドアロン版と VST3/ARA プラグインをインストールします。両者は同じモデルライブラリを共有するため、150 MB の重みは一度しか書き込まれません。$\r$\n続行する前に REAPER と HiFiShifter を終了してください。読み込み中のファイルは上書きされません。$\r$\n設定・キャッシュ・ログはユーザーフォルダーにあり、アンインストールしても残ります。$\r$\n同梱の NSF-HiFiGAN ボコーダー重みは CC BY-NC-SA 4.0 で、非商用用途に限ります。ライセンス全文は LICENSE と licenses フォルダーにインストールされます。"
LangString HFS_WELCOME_TEXT ${LANG_KOREAN} "HiFiShifter 독립 실행형 앱과 VST3/ARA 플러그인을 설치합니다. 두 형태는 하나의 모델 라이브러리를 공유하므로 150 MB 가중치는 한 번만 기록됩니다.$\r$\n계속하기 전에 REAPER와 HiFiShifter를 종료하세요. 로드 중인 파일은 덮어쓰지 않습니다.$\r$\n설정, 캐시, 로그는 사용자 폴더에 있으며 제거해도 유지됩니다.$\r$\n번들된 NSF-HiFiGAN 보코더 가중치는 CC BY-NC-SA 4.0 라이선스로 비상업적 용도로만 사용할 수 있습니다. 라이선스 전문은 LICENSE 및 licenses 폴더에 설치됩니다."

; installer_welcome_title
LangString HFS_WELCOME_TITLE ${LANG_ENGLISH} "Welcome to the HiFiShifter Setup"
LangString HFS_WELCOME_TITLE ${LANG_SIMPCHINESE} "欢迎安装 HiFiShifter"
LangString HFS_WELCOME_TITLE ${LANG_TRADCHINESE} "歡迎安裝 HiFiShifter"
LangString HFS_WELCOME_TITLE ${LANG_JAPANESE} "HiFiShifter セットアップへようこそ"
LangString HFS_WELCOME_TITLE ${LANG_KOREAN} "HiFiShifter 설치를 시작합니다"

; 「需要 64 位 Windows」是唯一在语言选择**之前**就要显示的提示：
; 那时 $LANGUAGE 还没定，$(HFS_...) 取不到值，只能五种语言并列。
!macro HFS_FATAL_NOT_X64
  ; en-US
  StrCpy $0 "$0HiFiShifter requires 64-bit Windows.$\r$\n"
  ; zh-CN
  StrCpy $0 "$0HiFiShifter 需要 64 位 Windows。$\r$\n"
  ; zh-TW
  StrCpy $0 "$0HiFiShifter 需要 64 位元 Windows。$\r$\n"
  ; ja-JP
  StrCpy $0 "$0HiFiShifter には 64 ビット版 Windows が必要です。$\r$\n"
  ; ko-KR
  StrCpy $0 "$0HiFiShifter는 64비트 Windows가 필요합니다.$\r$\n"
  MessageBox MB_OK|MB_ICONSTOP "$0"
!macroend
