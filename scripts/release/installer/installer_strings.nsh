; 本文件由 scripts/release/build-installer-strings.mjs 生成 —— 不要手工编辑。
;
; 文案的唯一来源是 frontend/src/i18n/*.ts 里 installer_ 前缀的词条；
; 直接改这里会在下一次生成时被覆盖，且 scripts/checks/check-product-consistency.ps1
; 会因与本文件不一致而失败。

; 语言：en-US→English、zh-CN→SimpChinese、zh-TW→TradChinese、ja-JP→Japanese、ko-KR→Korean
; 词条：43 条 × 5 种语言

; installer_app_dir_text
LangString HFS_APP_DIR_TEXT ${LANG_ENGLISH} "Choose the app install folder."
LangString HFS_APP_DIR_TEXT ${LANG_SIMPCHINESE} "选择独立 App 的安装文件夹。"
LangString HFS_APP_DIR_TEXT ${LANG_TRADCHINESE} "選擇獨立 App 的安裝資料夾。"
LangString HFS_APP_DIR_TEXT ${LANG_JAPANESE} "本体のインストール先フォルダーを選択。"
LangString HFS_APP_DIR_TEXT ${LANG_KOREAN} "앱을 설치할 폴더를 선택하세요."

; installer_clap_dir_text
LangString HFS_CLAP_DIR_TEXT ${LANG_ENGLISH} "Choose the CLAP folder scanned by your DAW."
LangString HFS_CLAP_DIR_TEXT ${LANG_SIMPCHINESE} "选择 DAW 扫描的 CLAP 文件夹。"
LangString HFS_CLAP_DIR_TEXT ${LANG_TRADCHINESE} "選擇 DAW 掃描的 CLAP 資料夾。"
LangString HFS_CLAP_DIR_TEXT ${LANG_JAPANESE} "DAW がスキャンする CLAP フォルダーを選択。"
LangString HFS_CLAP_DIR_TEXT ${LANG_KOREAN} "DAW가 검사하는 CLAP 폴더를 선택하세요."

; installer_component_app
LangString HFS_COMPONENT_APP ${LANG_ENGLISH} "Standalone app"
LangString HFS_COMPONENT_APP ${LANG_SIMPCHINESE} "独立 App"
LangString HFS_COMPONENT_APP ${LANG_TRADCHINESE} "獨立 App"
LangString HFS_COMPONENT_APP ${LANG_JAPANESE} "スタンドアロン版"
LangString HFS_COMPONENT_APP ${LANG_KOREAN} "독립 실행형 앱"

; installer_component_clap
LangString HFS_COMPONENT_CLAP ${LANG_ENGLISH} "CLAP"
LangString HFS_COMPONENT_CLAP ${LANG_SIMPCHINESE} "CLAP"
LangString HFS_COMPONENT_CLAP ${LANG_TRADCHINESE} "CLAP"
LangString HFS_COMPONENT_CLAP ${LANG_JAPANESE} "CLAP"
LangString HFS_COMPONENT_CLAP ${LANG_KOREAN} "CLAP"

; installer_component_file_assoc
LangString HFS_COMPONENT_FILE_ASSOC ${LANG_ENGLISH} "File association (.hshp/.hsp)"
LangString HFS_COMPONENT_FILE_ASSOC ${LANG_SIMPCHINESE} "文件关联（.hshp/.hsp）"
LangString HFS_COMPONENT_FILE_ASSOC ${LANG_TRADCHINESE} "檔案關聯（.hshp/.hsp）"
LangString HFS_COMPONENT_FILE_ASSOC ${LANG_JAPANESE} "ファイル関連付け（.hshp/.hsp）"
LangString HFS_COMPONENT_FILE_ASSOC ${LANG_KOREAN} "파일 연결(.hshp/.hsp)"

; installer_component_models
LangString HFS_COMPONENT_MODELS ${LANG_ENGLISH} "Model library"
LangString HFS_COMPONENT_MODELS ${LANG_SIMPCHINESE} "模型库"
LangString HFS_COMPONENT_MODELS ${LANG_TRADCHINESE} "模型庫"
LangString HFS_COMPONENT_MODELS ${LANG_JAPANESE} "モデルライブラリ"
LangString HFS_COMPONENT_MODELS ${LANG_KOREAN} "모델 라이브러리"

; installer_component_models_install
LangString HFS_COMPONENT_MODELS_INSTALL ${LANG_ENGLISH} "Installing model library..."
LangString HFS_COMPONENT_MODELS_INSTALL ${LANG_SIMPCHINESE} "正在安装模型库..."
LangString HFS_COMPONENT_MODELS_INSTALL ${LANG_TRADCHINESE} "正在安裝模型庫..."
LangString HFS_COMPONENT_MODELS_INSTALL ${LANG_JAPANESE} "モデルライブラリをインストール中..."
LangString HFS_COMPONENT_MODELS_INSTALL ${LANG_KOREAN} "모델 라이브러리 설치 중..."

; installer_component_models_skip
LangString HFS_COMPONENT_MODELS_SKIP ${LANG_ENGLISH} "Model library is up to date."
LangString HFS_COMPONENT_MODELS_SKIP ${LANG_SIMPCHINESE} "模型库已是最新。"
LangString HFS_COMPONENT_MODELS_SKIP ${LANG_TRADCHINESE} "模型庫已是最新。"
LangString HFS_COMPONENT_MODELS_SKIP ${LANG_JAPANESE} "モデルライブラリは最新です。"
LangString HFS_COMPONENT_MODELS_SKIP ${LANG_KOREAN} "모델 라이브러리가 최신입니다."

; installer_component_shortcuts
LangString HFS_COMPONENT_SHORTCUTS ${LANG_ENGLISH} "Shortcuts"
LangString HFS_COMPONENT_SHORTCUTS ${LANG_SIMPCHINESE} "快捷方式"
LangString HFS_COMPONENT_SHORTCUTS ${LANG_TRADCHINESE} "捷徑"
LangString HFS_COMPONENT_SHORTCUTS ${LANG_JAPANESE} "ショートカット"
LangString HFS_COMPONENT_SHORTCUTS ${LANG_KOREAN} "바로 가기"

; installer_component_vst3
LangString HFS_COMPONENT_VST3 ${LANG_ENGLISH} "VST3"
LangString HFS_COMPONENT_VST3 ${LANG_SIMPCHINESE} "VST3"
LangString HFS_COMPONENT_VST3 ${LANG_TRADCHINESE} "VST3"
LangString HFS_COMPONENT_VST3 ${LANG_JAPANESE} "VST3"
LangString HFS_COMPONENT_VST3 ${LANG_KOREAN} "VST3"

; installer_components_top
LangString HFS_COMPONENTS_TOP ${LANG_ENGLISH} "Grouped by purpose. Hover over an item for details."
LangString HFS_COMPONENTS_TOP ${LANG_SIMPCHINESE} "按用途分组，悬停查看说明。"
LangString HFS_COMPONENTS_TOP ${LANG_TRADCHINESE} "依用途分組，懸停查看說明。"
LangString HFS_COMPONENTS_TOP ${LANG_JAPANESE} "用途別にグループ化。カーソルを合わせると説明を表示。"
LangString HFS_COMPONENTS_TOP ${LANG_KOREAN} "용도별로 묶었습니다. 커서를 올리면 설명을 표시합니다."

; installer_desc_app
LangString HFS_DESC_APP ${LANG_ENGLISH} "The standalone editor. Always installed."
LangString HFS_DESC_APP ${LANG_SIMPCHINESE} "独立编辑器，必装。"
LangString HFS_DESC_APP ${LANG_TRADCHINESE} "獨立編輯器，必裝。"
LangString HFS_DESC_APP ${LANG_JAPANESE} "スタンドアロン版エディター。常にインストールされます。"
LangString HFS_DESC_APP ${LANG_KOREAN} "독립 실행형 편집기입니다. 항상 설치됩니다."

; installer_desc_clap
LangString HFS_DESC_CLAP ${LANG_ENGLISH} "CLAP plug-in (ARA + Capture). Installed into the CLAP folder you choose."
LangString HFS_DESC_CLAP ${LANG_SIMPCHINESE} "CLAP 插件（ARA + Capture），装入所选的 CLAP 文件夹。"
LangString HFS_DESC_CLAP ${LANG_TRADCHINESE} "CLAP 外掛（ARA + Capture），裝入所選的 CLAP 資料夾。"
LangString HFS_DESC_CLAP ${LANG_JAPANESE} "CLAP プラグイン（ARA + Capture）。選んだ CLAP フォルダーにインストール。"
LangString HFS_DESC_CLAP ${LANG_KOREAN} "CLAP 플러그인(ARA + Capture)입니다. 선택한 CLAP 폴더에 설치됩니다."

; installer_desc_file_assoc
LangString HFS_DESC_FILE_ASSOC ${LANG_ENGLISH} "Open HiFiShifter project files with HiFiShifter by default."
LangString HFS_DESC_FILE_ASSOC ${LANG_SIMPCHINESE} "将 HiFiShifter 工程文件默认用 HiFiShifter 打开。"
LangString HFS_DESC_FILE_ASSOC ${LANG_TRADCHINESE} "將 HiFiShifter 專案檔預設以 HiFiShifter 開啟。"
LangString HFS_DESC_FILE_ASSOC ${LANG_JAPANESE} "HiFiShifter のプロジェクトファイルを HiFiShifter で開くようにします。"
LangString HFS_DESC_FILE_ASSOC ${LANG_KOREAN} "HiFiShifter 프로젝트 파일을 기본적으로 HiFiShifter로 엽니다."

; installer_desc_group_app
LangString HFS_DESC_GROUP_APP ${LANG_ENGLISH} "The app, the model library, and optional shortcuts."
LangString HFS_DESC_GROUP_APP ${LANG_SIMPCHINESE} "主程序、模型库与可选的快捷方式。"
LangString HFS_DESC_GROUP_APP ${LANG_TRADCHINESE} "主程式、模型庫與可選的捷徑。"
LangString HFS_DESC_GROUP_APP ${LANG_JAPANESE} "本体、モデルライブラリ、任意のショートカット。"
LangString HFS_DESC_GROUP_APP ${LANG_KOREAN} "본체, 모델 라이브러리, 선택적 바로 가기."

; installer_desc_group_plugins
LangString HFS_DESC_GROUP_PLUGINS ${LANG_ENGLISH} "Plug-in formats for your DAW. Select the ones it supports."
LangString HFS_DESC_GROUP_PLUGINS ${LANG_SIMPCHINESE} "供 DAW 加载的插件格式，按你的 DAW 支持的格式勾选。"
LangString HFS_DESC_GROUP_PLUGINS ${LANG_TRADCHINESE} "供 DAW 載入的外掛格式，依你的 DAW 支援的格式勾選。"
LangString HFS_DESC_GROUP_PLUGINS ${LANG_JAPANESE} "DAW で読み込むプラグイン形式。お使いの DAW が対応するものを選択。"
LangString HFS_DESC_GROUP_PLUGINS ${LANG_KOREAN} "DAW에서 불러올 플러그인 형식입니다. 사용하는 DAW가 지원하는 형식을 선택하세요."

; installer_desc_models
LangString HFS_DESC_MODELS ${LANG_ENGLISH} "Model weights shared by the app and all plug-ins. Stored once."
LangString HFS_DESC_MODELS ${LANG_SIMPCHINESE} "各形态共用的模型权重，只存一份。"
LangString HFS_DESC_MODELS ${LANG_TRADCHINESE} "各形態共用的模型權重，只存一份。"
LangString HFS_DESC_MODELS ${LANG_JAPANESE} "本体と全プラグインで共有するモデル重み。1 か所にのみ保存。"
LangString HFS_DESC_MODELS ${LANG_KOREAN} "본체와 모든 플러그인이 공유하는 모델 가중치입니다. 한 번만 저장됩니다."

; installer_desc_shortcuts
LangString HFS_DESC_SHORTCUTS ${LANG_ENGLISH} "Adds a Start menu entry and a desktop icon."
LangString HFS_DESC_SHORTCUTS ${LANG_SIMPCHINESE} "添加开始菜单项和桌面图标。"
LangString HFS_DESC_SHORTCUTS ${LANG_TRADCHINESE} "新增開始功能表項目與桌面圖示。"
LangString HFS_DESC_SHORTCUTS ${LANG_JAPANESE} "スタートメニュー項目とデスクトップアイコンを追加。"
LangString HFS_DESC_SHORTCUTS ${LANG_KOREAN} "시작 메뉴 항목과 바탕 화면 아이콘을 추가합니다."

; installer_desc_vst3
LangString HFS_DESC_VST3 ${LANG_ENGLISH} "VST3 plug-in (ARA + Capture). Installed into the VST3 folder you choose next."
LangString HFS_DESC_VST3 ${LANG_SIMPCHINESE} "VST3 插件（ARA + Capture），装入下一页所选的 VST3 文件夹。"
LangString HFS_DESC_VST3 ${LANG_TRADCHINESE} "VST3 外掛（ARA + Capture），裝入下一頁所選的 VST3 資料夾。"
LangString HFS_DESC_VST3 ${LANG_JAPANESE} "VST3 プラグイン（ARA + Capture）。次のページで選ぶ VST3 フォルダーにインストール。"
LangString HFS_DESC_VST3 ${LANG_KOREAN} "VST3 플러그인(ARA + Capture)입니다. 다음 페이지에서 선택한 VST3 폴더에 설치됩니다."

; installer_dir_text
LangString HFS_DIR_TEXT ${LANG_ENGLISH} "Choose the VST3 folder scanned by your DAW."
LangString HFS_DIR_TEXT ${LANG_SIMPCHINESE} "选择 DAW 扫描的 VST3 文件夹。"
LangString HFS_DIR_TEXT ${LANG_TRADCHINESE} "選擇 DAW 掃描的 VST3 資料夾。"
LangString HFS_DIR_TEXT ${LANG_JAPANESE} "DAW がスキャンする VST3 フォルダーを選択。"
LangString HFS_DIR_TEXT ${LANG_KOREAN} "DAW가 검사하는 VST3 폴더를 선택하세요."

; installer_err_app_running
LangString HFS_ERR_APP_RUNNING ${LANG_ENGLISH} "HiFiShifter is running. Close it and try again."
LangString HFS_ERR_APP_RUNNING ${LANG_SIMPCHINESE} "HiFiShifter 正在运行，请关闭后重试。"
LangString HFS_ERR_APP_RUNNING ${LANG_TRADCHINESE} "HiFiShifter 正在執行，請關閉後重試。"
LangString HFS_ERR_APP_RUNNING ${LANG_JAPANESE} "HiFiShifter が実行中です。終了して再試行してください。"
LangString HFS_ERR_APP_RUNNING ${LANG_KOREAN} "HiFiShifter가 실행 중입니다. 종료하고 다시 시도하세요."

; installer_err_host_open
LangString HFS_ERR_HOST_OPEN ${LANG_ENGLISH} "A DAW is still using the plug-in. Close all DAWs and try again."
LangString HFS_ERR_HOST_OPEN ${LANG_SIMPCHINESE} "DAW 仍在加载插件，请关闭所有 DAW 后重试。"
LangString HFS_ERR_HOST_OPEN ${LANG_TRADCHINESE} "DAW 仍在載入外掛，請關閉所有 DAW 後重試。"
LangString HFS_ERR_HOST_OPEN ${LANG_JAPANESE} "プラグインを使用中の DAW があります。すべての DAW を閉じて再試行してください。"
LangString HFS_ERR_HOST_OPEN ${LANG_KOREAN} "플러그인을 사용 중인 DAW가 있습니다. 모든 DAW를 닫고 다시 시도하세요."

; installer_err_not_x64
LangString HFS_ERR_NOT_X64 ${LANG_ENGLISH} "HiFiShifter requires 64-bit Windows."
LangString HFS_ERR_NOT_X64 ${LANG_SIMPCHINESE} "HiFiShifter 需要 64 位 Windows。"
LangString HFS_ERR_NOT_X64 ${LANG_TRADCHINESE} "HiFiShifter 需要 64 位元 Windows。"
LangString HFS_ERR_NOT_X64 ${LANG_JAPANESE} "HiFiShifter には 64 ビット版 Windows が必要です。"
LangString HFS_ERR_NOT_X64 ${LANG_KOREAN} "HiFiShifter는 64비트 Windows가 필요합니다."

; installer_err_tasklist
LangString HFS_ERR_TASKLIST ${LANG_ENGLISH} "Could not confirm all DAWs are closed. Close them and try again."
LangString HFS_ERR_TASKLIST ${LANG_SIMPCHINESE} "无法确认 DAW 已全部关闭，请关闭后重试。"
LangString HFS_ERR_TASKLIST ${LANG_TRADCHINESE} "無法確認 DAW 已全部關閉，請關閉後重試。"
LangString HFS_ERR_TASKLIST ${LANG_JAPANESE} "DAW の終了を確認できません。閉じてから再試行してください。"
LangString HFS_ERR_TASKLIST ${LANG_KOREAN} "DAW 종료를 확인할 수 없습니다. 닫고 다시 시도하세요."

; installer_fileassoc_type_name
LangString HFS_FILEASSOC_TYPE_NAME ${LANG_ENGLISH} "HiFiShifter Project"
LangString HFS_FILEASSOC_TYPE_NAME ${LANG_SIMPCHINESE} "HiFiShifter 工程"
LangString HFS_FILEASSOC_TYPE_NAME ${LANG_TRADCHINESE} "HiFiShifter 專案"
LangString HFS_FILEASSOC_TYPE_NAME ${LANG_JAPANESE} "HiFiShifter プロジェクト"
LangString HFS_FILEASSOC_TYPE_NAME ${LANG_KOREAN} "HiFiShifter 프로젝트"

; installer_group_app
LangString HFS_GROUP_APP ${LANG_ENGLISH} "Application"
LangString HFS_GROUP_APP ${LANG_SIMPCHINESE} "主程序"
LangString HFS_GROUP_APP ${LANG_TRADCHINESE} "主程式"
LangString HFS_GROUP_APP ${LANG_JAPANESE} "本体"
LangString HFS_GROUP_APP ${LANG_KOREAN} "애플리케이션"

; installer_group_plugins
LangString HFS_GROUP_PLUGINS ${LANG_ENGLISH} "DAW plug-ins"
LangString HFS_GROUP_PLUGINS ${LANG_SIMPCHINESE} "DAW 插件"
LangString HFS_GROUP_PLUGINS ${LANG_TRADCHINESE} "DAW 外掛"
LangString HFS_GROUP_PLUGINS ${LANG_JAPANESE} "DAW プラグイン"
LangString HFS_GROUP_PLUGINS ${LANG_KOREAN} "DAW 플러그인"

; installer_installing
LangString HFS_INSTALLING ${LANG_ENGLISH} "Installing HiFiShifter..."
LangString HFS_INSTALLING ${LANG_SIMPCHINESE} "正在安装 HiFiShifter..."
LangString HFS_INSTALLING ${LANG_TRADCHINESE} "正在安裝 HiFiShifter..."
LangString HFS_INSTALLING ${LANG_JAPANESE} "HiFiShifter をインストール中..."
LangString HFS_INSTALLING ${LANG_KOREAN} "HiFiShifter 설치 중..."

; installer_models_verify_failed
LangString HFS_MODELS_VERIFY_FAILED ${LANG_ENGLISH} "Model library verification failed. Try again; if it persists, check disk space."
LangString HFS_MODELS_VERIFY_FAILED ${LANG_SIMPCHINESE} "模型库校验失败，请重试。若反复失败请检查磁盘空间。"
LangString HFS_MODELS_VERIFY_FAILED ${LANG_TRADCHINESE} "模型庫驗證失敗，請重試。若反覆失敗請檢查磁碟空間。"
LangString HFS_MODELS_VERIFY_FAILED ${LANG_JAPANESE} "モデルライブラリの検証に失敗しました。再試行し、繰り返す場合は空き容量を確認してください。"
LangString HFS_MODELS_VERIFY_FAILED ${LANG_KOREAN} "모델 라이브러리 검증에 실패했습니다. 다시 시도하고, 반복되면 디스크 여유 공간을 확인하세요."

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
LangString HFS_REPAIR_DONE ${LANG_ENGLISH} "Repair complete."
LangString HFS_REPAIR_DONE ${LANG_SIMPCHINESE} "修复完成。"
LangString HFS_REPAIR_DONE ${LANG_TRADCHINESE} "修復完成。"
LangString HFS_REPAIR_DONE ${LANG_JAPANESE} "修復が完了しました。"
LangString HFS_REPAIR_DONE ${LANG_KOREAN} "복구가 완료되었습니다."

; installer_uninstall_host_open
LangString HFS_UNINSTALL_HOST_OPEN ${LANG_ENGLISH} "Close all DAWs before uninstalling."
LangString HFS_UNINSTALL_HOST_OPEN ${LANG_SIMPCHINESE} "卸载前请关闭所有 DAW。"
LangString HFS_UNINSTALL_HOST_OPEN ${LANG_TRADCHINESE} "卸載前請關閉所有 DAW。"
LangString HFS_UNINSTALL_HOST_OPEN ${LANG_JAPANESE} "アンインストール前にすべての DAW を終了してください。"
LangString HFS_UNINSTALL_HOST_OPEN ${LANG_KOREAN} "제거하기 전에 모든 DAW를 종료하세요."

; installer_uninstall_opt_cache
LangString HFS_UNINSTALL_OPT_CACHE ${LANG_ENGLISH} "Remove caches and logs"
LangString HFS_UNINSTALL_OPT_CACHE ${LANG_SIMPCHINESE} "清理缓存与日志"
LangString HFS_UNINSTALL_OPT_CACHE ${LANG_TRADCHINESE} "清理快取與記錄檔"
LangString HFS_UNINSTALL_OPT_CACHE ${LANG_JAPANESE} "キャッシュとログを削除"
LangString HFS_UNINSTALL_OPT_CACHE ${LANG_KOREAN} "캐시 및 로그 삭제"

; installer_uninstall_opt_cache_desc
LangString HFS_UNINSTALL_OPT_CACHE_DESC ${LANG_ENGLISH} "Waveform/render caches, logs, WebView data and temporary files."
LangString HFS_UNINSTALL_OPT_CACHE_DESC ${LANG_SIMPCHINESE} "波形/渲染缓存、日志、WebView 数据与临时文件。"
LangString HFS_UNINSTALL_OPT_CACHE_DESC ${LANG_TRADCHINESE} "波形/渲染快取、記錄檔、WebView 資料與暫存檔。"
LangString HFS_UNINSTALL_OPT_CACHE_DESC ${LANG_JAPANESE} "波形/レンダリング キャッシュ、ログ、WebView データ、一時ファイル。"
LangString HFS_UNINSTALL_OPT_CACHE_DESC ${LANG_KOREAN} "파형/렌더 캐시, 로그, WebView 데이터 및 임시 파일."

; installer_uninstall_opt_models
LangString HFS_UNINSTALL_OPT_MODELS ${LANG_ENGLISH} "Remove the shared model library"
LangString HFS_UNINSTALL_OPT_MODELS ${LANG_SIMPCHINESE} "清理共享模型库"
LangString HFS_UNINSTALL_OPT_MODELS ${LANG_TRADCHINESE} "清理共用模型庫"
LangString HFS_UNINSTALL_OPT_MODELS ${LANG_JAPANESE} "共有モデル ライブラリを削除"
LangString HFS_UNINSTALL_OPT_MODELS ${LANG_KOREAN} "공유 모델 라이브러리 삭제"

; installer_uninstall_opt_models_desc
LangString HFS_UNINSTALL_OPT_MODELS_DESC ${LANG_ENGLISH} "Frees about 150 MB. Shared by every HiFiShifter installation."
LangString HFS_UNINSTALL_OPT_MODELS_DESC ${LANG_SIMPCHINESE} "约释放 150 MB，所有 HiFiShifter 安装共用这一份。"
LangString HFS_UNINSTALL_OPT_MODELS_DESC ${LANG_TRADCHINESE} "約釋出 150 MB，所有 HiFiShifter 安裝共用這一份。"
LangString HFS_UNINSTALL_OPT_MODELS_DESC ${LANG_JAPANESE} "約 150 MB を解放。すべての HiFiShifter インストールで共有されます。"
LangString HFS_UNINSTALL_OPT_MODELS_DESC ${LANG_KOREAN} "약 150 MB 확보. 모든 HiFiShifter 설치가 공유합니다."

; installer_uninstall_options_subtitle
LangString HFS_UNINSTALL_OPTIONS_SUBTITLE ${LANG_ENGLISH} "Choose what to remove along with HiFiShifter."
LangString HFS_UNINSTALL_OPTIONS_SUBTITLE ${LANG_SIMPCHINESE} "选择卸载 HiFiShifter 时要一并清理的内容。"
LangString HFS_UNINSTALL_OPTIONS_SUBTITLE ${LANG_TRADCHINESE} "選擇卸載 HiFiShifter 時要一併清理的內容。"
LangString HFS_UNINSTALL_OPTIONS_SUBTITLE ${LANG_JAPANESE} "HiFiShifter と一緒に削除する項目を選択してください。"
LangString HFS_UNINSTALL_OPTIONS_SUBTITLE ${LANG_KOREAN} "HiFiShifter와 함께 정리할 항목을 선택하세요."

; installer_uninstall_options_title
LangString HFS_UNINSTALL_OPTIONS_TITLE ${LANG_ENGLISH} "Uninstall options"
LangString HFS_UNINSTALL_OPTIONS_TITLE ${LANG_SIMPCHINESE} "卸载选项"
LangString HFS_UNINSTALL_OPTIONS_TITLE ${LANG_TRADCHINESE} "卸載選項"
LangString HFS_UNINSTALL_OPTIONS_TITLE ${LANG_JAPANESE} "アンインストール オプション"
LangString HFS_UNINSTALL_OPTIONS_TITLE ${LANG_KOREAN} "제거 옵션"

; installer_uninstall_options_top
LangString HFS_UNINSTALL_OPTIONS_TOP ${LANG_ENGLISH} "Also remove the data below. Your settings and unsaved work are always kept."
LangString HFS_UNINSTALL_OPTIONS_TOP ${LANG_SIMPCHINESE} "可一并清理下列数据。设置与未保存的工程始终保留。"
LangString HFS_UNINSTALL_OPTIONS_TOP ${LANG_TRADCHINESE} "可一併清理下列資料。設定與未儲存的工程一律保留。"
LangString HFS_UNINSTALL_OPTIONS_TOP ${LANG_JAPANESE} "以下をあわせて削除できます。設定と未保存の作業は常に保持されます。"
LangString HFS_UNINSTALL_OPTIONS_TOP ${LANG_KOREAN} "아래 데이터를 함께 정리할 수 있습니다. 설정과 저장하지 않은 작업은 항상 유지됩니다."

; installer_webview2_install_failed
LangString HFS_WEBVIEW2_INSTALL_FAILED ${LANG_ENGLISH} "WebView2 could not be installed. Install it from Microsoft's site and try again."
LangString HFS_WEBVIEW2_INSTALL_FAILED ${LANG_SIMPCHINESE} "WebView2 运行时安装失败，请从微软官网安装后重试。"
LangString HFS_WEBVIEW2_INSTALL_FAILED ${LANG_TRADCHINESE} "WebView2 執行階段安裝失敗，請從微軟官網安裝後重試。"
LangString HFS_WEBVIEW2_INSTALL_FAILED ${LANG_JAPANESE} "WebView2 のインストールに失敗しました。Microsoft のサイトからインストールして再試行してください。"
LangString HFS_WEBVIEW2_INSTALL_FAILED ${LANG_KOREAN} "WebView2 설치에 실패했습니다. Microsoft 사이트에서 설치한 뒤 다시 시도하세요."

; installer_welcome_text
LangString HFS_WELCOME_TEXT ${LANG_ENGLISH} "Installs the HiFiShifter app and its VST3 and CLAP plug-ins.$\r$\nClose HiFiShifter and any DAW using the plug-ins before continuing.$\r$\nThe NSF-HiFiGAN weights are CC BY-NC-SA 4.0, non-commercial use only; see LICENSE."
LangString HFS_WELCOME_TEXT ${LANG_SIMPCHINESE} "将安装 HiFiShifter 独立 App，以及 VST3 与 CLAP 插件。$\r$\n安装前请关闭 HiFiShifter 和所有正在使用插件的 DAW。$\r$\nNSF-HiFiGAN 权重采用 CC BY-NC-SA 4.0，仅限非商业使用。详见 LICENSE。"
LangString HFS_WELCOME_TEXT ${LANG_TRADCHINESE} "將安裝 HiFiShifter 獨立 App，以及 VST3 與 CLAP 外掛。$\r$\n安裝前請關閉 HiFiShifter 和所有正在使用外掛的 DAW。$\r$\nNSF-HiFiGAN 權重採用 CC BY-NC-SA 4.0，僅限非商業使用。詳見 LICENSE。"
LangString HFS_WELCOME_TEXT ${LANG_JAPANESE} "HiFiShifter 本体と VST3・CLAP プラグインをインストールします。$\r$\n続行する前に、HiFiShifter とプラグインを使用中の DAW を終了してください。$\r$\nNSF-HiFiGAN の重みは CC BY-NC-SA 4.0 で非商用のみ。詳細は LICENSE を参照。"
LangString HFS_WELCOME_TEXT ${LANG_KOREAN} "HiFiShifter 앱과 VST3·CLAP 플러그인을 설치합니다.$\r$\n계속하기 전에 HiFiShifter와 플러그인을 사용 중인 DAW를 종료하세요.$\r$\nNSF-HiFiGAN 가중치는 CC BY-NC-SA 4.0으로 비상업적 용도로만 사용할 수 있습니다. 자세한 내용은 LICENSE 참조."

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
