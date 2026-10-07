# HiFiShifter

[简体中文](../../README.md) | [繁體中文](README_zh-TW.md) | [English](README_en.md) | [日本語](README_ja.md) | [한국어](README_ko.md)

HiFiShifter 是一個圖形化人聲編輯與合成工具。它支援多軌道音訊塊處理，並以軌道組為單位，使用多種聲碼器完成人聲修音、人力調參功能，實現人力 VOCALOID 製作的拼調一體化。

**當前專案仍在開發迭代中，未對全鏈路進行測試，可能存在諸多 BUG 或不穩定問題。**

![預覽圖](../preview.png)

## 功能一覽

- **軌道編輯**：類 DAW 的多軌時間軸，支援音訊塊的裁切、拉伸、Slip、淡入淡出與交叉淡化、編組、Take 管理、靜音偵測、波紋編輯、速度映射（BPM / 拍號 / 音階）、節拍器與錄音等。
- **參數編輯**：以軌道群組為單位，透過鋼琴捲簾式參數編輯器調整音高、音量、動態、聲相、共振峰、氣聲、張力等參數線；支援繪製 / 直線 / 顫音工具、多選區編輯與顫音預設管理，子軌道可疊加 `音分差` / `度數差` / `共振峰差` 製作和聲。
- **三種聲碼器演算法**：nsf-hifigan（PC-NSF-HiFiGAN）、World、VsLib，詳見[演算法](#演算法)。
- **互操作性**：匯入 REAPER（`.rpp`）與 VocalShifter（`.vshp` / `.vsp`）專案，雙向讀寫 REAPER 與 VocalShifter 剪貼簿；MIDI 可匯入為音高參考塊 / 音高參數 / 速度映射，音訊塊與音高線可匯出為 MIDI。
- **匯入匯出**：常見音訊 / 視訊格式匯入（視訊檔自動提取音訊軌），專案與分軌匯出為 `wav` / `mp3` / `flac`。
- **其他**：內建檔案瀏覽器與快速搜尋、記事本、自動備份、多語言介面（简体中文 / 繁體中文 / English / 日本語 / 한국어）、深淺色主題、推理裝置選擇與基準測試。

## 安裝

從 [Releases](https://github.com/ARounder-183/HiFiShifter/releases) 頁面下載對應作業系統與架構的安裝包：

- **Windows**：NSIS 安裝包（`installer`）或可攜版壓縮包（`portable`），提供 x86_64 與 arm64 架構。
- **macOS**：未簽名 dmg（Apple Silicon 裝 `arm64`，Intel 裝 `x86_64`）。首次安裝需要手動放行；若提示「檔案已損毀」，請按照[使用手冊](USERMANUAL_zh-TW.md#一安裝)中的步驟處理。
- **Linux**：AppImage（x86_64 / arm64）。

GPU 加速：Windows 使用 DirectML（DirectX 12），macOS（Apple Silicon）使用 CoreML + WebGPU，Linux x86_64 使用 WebGPU（Dawn/Vulkan），其餘平台回退 CPU。可在應用內 `選項 → 推理裝置` 中切換裝置並執行基準測試。

## 基本原理

HiFiShifter 使用類似 UTAU 的離線渲染方式，對時間線中的每個音訊塊進行處理、渲染、快取，最後再輸入到播放系統中，因此其對短音訊塊有著更快的處理效率。

HiFiShifter 提供了一個統一的渲染介面，以便未來增添更多的演算法支援。

## 推薦工作流

我們推薦的工作流是：

1. 透過其他 DAW 或切片軟體準備好人力所需的短片段音源。
2. 在 HiFiShifter 中完成音訊的拼貼和調音。

當然，HiFiShifter 也支援以下操作方便從其他軟體的專案遷移：

1. 直接開啟 VocalShifter 專案。
2. 直接開啟 Reaper 專案。
3. 解析 VocalShifter 剪貼簿內容，支援將 VocalShifter 中的參數貼到 HiFiShifter 參數區中。
4. 解析 Reaper 剪貼簿內容，支援直接將 Reaper 的 Items 貼到 HiFiShifter 中。

## 介面與演算法

### 佈局

HiFiShifter 大致分為上部的軌道面板和下部的參數面板：軌道面板負責音訊塊的編輯與編排，參數面板負責對音訊進行調參處理。各面板可停靠、浮動與重排（`檢視 → 視窗 / 佈局`）。

軌道支援巢狀：將一個軌道拖曳到另一個軌道下，即可組成軌道群組（根軌道 + 子軌道）。軌道群組共用一個演算法和一套參數線，參數線會按位置作用到群組內每一個音訊塊上；調參前需要先按下軌道的合成按鈕 `C`。

### 演算法

目前 HiFiShifter 支援三種演算法進行處理：

- **World**：老牌聲碼器。支援 `音高`、`音量`、`動態`、`聲相` 參數的編輯。
- **PC-NSF-HiFiGAN**（介面演算法清單中顯示為 `nsf-hifigan`）：OpenVPI 開源、為歌聲特化的 HiFi-GAN 聲碼器，也是預設演算法。支援 `音高`、`共振峰偏移`、`氣聲音量`、`張力`、`音量`、`動態`、`聲相` 參數的編輯。其中 `氣聲音量` 與 `張力` 依賴 `氣聲分離` 開關：開啟後會把音訊塊分離為諧波與噪聲兩部分（使用 hnsep 模型），**會增加額外的渲染成本**（首次每個音訊塊都要做一次分離，可能較慢）；關閉時則完全不進行分離，這兩個參數會被置灰、其曲線保持可見但不可編輯，也不參與合成。
- **VsLib**（介面演算法清單中顯示為 `vslib`）：VocalShifter 官方提供的演算法庫。支援 `音高`、`共振峰偏移`、`氣聲強度`、`音量`、`動態`、`聲相` 與 `合成模式` 參數的編輯。**僅 Windows x86_64 版本可用**；由於官方提供的 dll 僅支援檔案 I/O，因此相對 VocalShifter 本體需要更多的時間處理。

軌道面板與參數面板的詳細操作（淡化編輯、吸附設定、顫音預設、匯出與錄音等）請閱讀[使用手冊](USERMANUAL_zh-TW.md)。

## 常用快捷鍵速查

> 快捷鍵均可在應用內 `選項 → 鍵盤快捷鍵...` 中自訂，下表為預設值；macOS 上 `Ctrl` 對應 `⌘`、`Alt` 對應 `⌥`。

| 操作                           | 快捷鍵 / 滑鼠                                                |
| :----------------------------- | :----------------------------------------------------------- |
| 播放 / 暫停（不返回起播點）    | `Space`                                                      |
| 播放 / 停止（返回起播點）      | `Enter`                                                      |
| 開關節拍器                     | `K`                                                          |
| 平移檢視                       | 滑鼠中鍵拖曳                                                 |
| 滾動時間軸                     | 滑鼠滾輪（雙軸自由滾動）                                     |
| 橫向 / 縱向滾動                | `Shift` / `Alt` + 滑鼠滾輪                                   |
| 縮放軌道高度                   | `Ctrl` + 滑鼠滾輪                                            |
| 復原 / 重做                    | `Ctrl + Z` / `Ctrl + Shift + Z`（或 `Ctrl + Y`）             |
| 剪下 / 複製 / 貼上             | `Ctrl + X` / `Ctrl + C` / `Ctrl + V`                         |
| 刪除選取音訊塊                 | `Delete`                                                     |
| 分割音訊塊（播放頭處）         | `S`                                                          |
| 編組 / 解組                    | `G` / `U`                                                    |
| 循環切換 Take                  | `T`（`Shift + T` 切換上一個）                                |
| 框選多選                       | 在時間軸空白處按住滑鼠右鍵拖曳                               |
| 複製拖動                       | 按住 `Ctrl` 拖曳音訊塊                                       |
| 拉伸音訊塊 / Slip 編輯         | 按住 `Alt` 拖曳音訊塊邊緣 / 主體                             |
| 暫時切換吸附                   | 按住 `Shift` 拖曳                                            |
| 新建專案 / 開啟專案            | `Ctrl + N` / `Ctrl + Shift + O`                              |
| 匯入媒體檔案 / 匯出音訊        | `Ctrl + O` / `Ctrl + E`                                      |
| 儲存 / 另存新檔                | `Ctrl + S` / `Ctrl + Shift + S`                              |
| 新增軌道 / 錄音                | `Ctrl + T` / `Ctrl + R`                                      |
| 模式切換（選取 ↔ 繪製類工具）  | `Tab`                                                        |
| 快速搜尋                       | `Ctrl + F`                                                   |
| 參數線整體上移 / 下移          | `=` / `-`（選取範圍用 `[` / `]`），`Shift` 大幅、`Ctrl` 微調 |
| 參數編輯器多選區               | 按住 `Ctrl` 拖曳追加選區、點擊已有選區取消                   |

完整快捷鍵與修飾鍵清單見[使用手冊](USERMANUAL_zh-TW.md#三軌道介面)或應用內快捷鍵設定。

## 開發

該部分內容為開發者提供，普通使用者可以跳過。

### 1. 克隆倉庫

```bash
git clone https://github.com/ARounder-183/HiFiShifter.git
cd HiFiShifter
```

### 2. 安裝依賴

#### Windows

請確保已安裝以下工具：

- **Node.js**（建議 20+，CI 使用 24）及 npm
- **Rust 工具鏈**（參見 `rust-toolchain.toml`）
- **Tauri 2 CLI**：`cargo install tauri-cli --version "^2"`
- **CMake**（用於編譯 SoundTouch 函式庫）

ONNX Runtime (DirectML) 由 ort crate 在編譯時自動下載，無需額外設定。

也可以執行 `scripts/install_deps_windows.ps1`，透過 Chocolatey 安裝 NSIS / CMake / LLVM 並安裝 tauri-cli（需要已安裝 Chocolatey）。

安裝前端依賴：

```bash
npm --prefix frontend install
```

#### macOS

```bash
chmod +x ./scripts/install_deps_macos.sh
SKIP_FRONTEND=0 bash ./scripts/install_deps_macos.sh
```

#### Linux

請確保已安裝以下工具：

- **Node.js**（建議 20+，CI 使用 24）及 npm
- **Rust 工具鏈**（參見 `rust-toolchain.toml`，專案會自動選擇對應平台的 stable 工具鏈）
- **Tauri 2 CLI**：`cargo install tauri-cli --version "^2"`
- **CMake**、**pkg-config** 及系統構建工具
- **GTK3、WebKit2GTK、ALSA** 等 Tauri 執行時開發函式庫（詳見安裝腳本）

執行一鍵安裝腳本：

```bash
chmod +x ./scripts/install_deps_linux.sh
bash ./scripts/install_deps_linux.sh
```

腳本會自動安裝系統依賴、Node.js（如未安裝）、appimagetool 及前端 npm 依賴。

安裝前端依賴（如未使用腳本）：

```bash
npm --prefix frontend ci
```

### 3. 第三方原始碼

SoundTouch、WORLD、Signalsmith Stretch 均在編譯時從原始碼構建，首次構建時會**自動克隆**，無需手動操作。

如需離線構建，可提前手動克隆，例如 SoundTouch：

```bash
cd backend/src-tauri/third_party/soundtouch-static
git clone --depth 1 --branch 2.3.3 https://codeberg.org/soundtouch/soundtouch.git soundtouch
```

### 4. 開發與構建

```bash
# 開發模式（熱更新）
cd backend
cargo tauri dev

# 構建 Release
# Windows / macOS（預設 features：onnx + vslib）
cargo tauri build

# Linux AppImage（vslib 僅限 Windows，需排除預設 feature）
cargo tauri build --bundles appimage -- --no-default-features --features onnx
# 或使用輔助腳本：bash scripts/build-linux-appimage.sh

# Windows 可攜版 ZIP
.\scripts\pack-portable.ps1 -SkipBuild
```

前端啟動模式可透過環境變數 `TAURI_UI_MODE` 切換：

- `dev`：開發模式（預設，使用 Vite dev server，支援熱更新）
- `build`：建置模式（先建置前端靜態資源，再啟動）

Linux/macOS（bash/zsh）：

```bash
cd backend
TAURI_UI_MODE=build cargo tauri dev
```

Windows PowerShell：

```powershell
cd backend
$env:TAURI_UI_MODE='build'; cargo tauri dev
```

**注意：** 首次編譯需要很長的時間，請耐心等待。

### 5. GPU 加速

HiFiShifter 在支援的平台上自動啟用 GPU 推理加速。你可以在選單列的**推理裝置（Inference Device）**中選擇 Auto / CPU / GPU，並透過**執行基準測試（Run Benchmark）**比較各裝置的推理延遲。

| 平台                        | GPU 技術                     | 說明                                                        |
| --------------------------- | ---------------------------- | ----------------------------------------------------------- |
| Windows x86_64 / ARM64      | DirectML (DirectX 12)        | 成熟穩定的 GPU 路徑，支援 NVIDIA / AMD / Intel Arc          |
| macOS ARM64 (Apple Silicon) | CoreML + WebGPU (Dawn/Metal) | CoreML 利用 Apple Neural Engine；WebGPU 作為補充 GPU 後端   |
| macOS x86_64 (Intel)        | —                            | 僅 CPU（使用 ort-tract 替代後端）                           |
| Linux x86_64                | WebGPU (Dawn/Vulkan)         | Dawn 透過 Vulkan API 存取 GPU；無 GPU 時自動回退到 CPU      |
| Linux ARM64                 | —                            | 僅 CPU（此目標尚無預編譯的 WebGPU ONNX Runtime 二進位檔案） |

> **注意**：Windows 平台未啟用 WebGPU。其 Dawn/D3D12 後端在部分 GPU/驅動組合上存在原生崩潰風險。DirectML 是 Windows 上成熟穩定的 GPU 路徑。

ONNX Runtime 二進位檔案由 ort crate 在編譯時透過 `download-binaries` 特性自動下載，無需手動設定。GPU 提供程序（DirectML / WebGPU / CoreML）在編譯時根據目標平台自動啟用，無需額外的 `--features` 標誌。

**WSL2 使用者**：

- WSL2 不向 Linux 子環境暴露硬體 Vulkan。WebGPU/Dawn 只能使用 Lavapipe（CPU 軟體渲染），效能極差。如需 GPU 加速，請使用 Windows 原生版本的 DirectML。
- 因缺少 FUSE 支援，Tauri bundler 的 linuxdeploy 步驟可能失敗（錯誤：`failed to run linuxdeploy`）。這是 WSL2 已知限制，不影響實際 AppImage 產出——AppDir 已正確組裝在 `target/release/bundle/appimage/` 中。可設定 `APPIMAGE_EXTRACT_AND_RUN=1` 後手動執行 `appimagetool` 打包。在真實 Linux 機器與 CI 中不存在此問題。

## 日誌與故障排除

應用程式會自動將執行日誌寫入系統標準日誌目錄，無需任何命令列參數：

| 系統 | 日誌目錄 |
| --- | --- |
| Windows | `%LOCALAPPDATA%\com.arounder.hifishifter\logs` |
| macOS | `~/Library/Logs/com.arounder.hifishifter` |
| Linux | `~/.local/share/com.arounder.hifishifter/logs` |

- 在應用程式內透過 **說明 → 開啟日誌資料夾** 可以直接定位日誌；**說明 → 匯出診斷資訊** 可以一鍵產生診斷套件（系統資訊 + 全部日誌 + 推理裝置基準測試結果），提交 issue 時附上即可。
- 日誌依大小自動輪替：單一檔案上限 8 MiB，預設保留 3 份歷史（`hifishifter.1.log` ~ `hifishifter.3.log`）。
  - 高頻重複的錯誤 / 警告會自動限流：同一位置的日誌預設每 10 秒最多輸出一條，被抑制的條數會在下一條輸出前以 `[throttled]` 彙總行補記。
- 前端與後端的錯誤都會統一記錄在同一份日誌檔案裡，方便按時間軸對照排查。

進階選項：

- 啟動參數 `--log-file=<path>`：將日誌寫入指定路徑；`--log-file=-` 明確關閉檔案日誌。
- 環境變數 `HIFISHIFTER_LOG=debug`（或 `trace` / `info` / `warn` / `error`）：調整日誌詳細程度。
- 環境變數 `HIFISHIFTER_LOG_DIR`：覆蓋預設日誌目錄。
- 在終端機執行 `HiFiShifter --benchmark`：直接執行推理裝置基準測試並輸出 JSON 結果。

## 文件

- 使用手冊：[简体中文](../../docs/i18n/USERMANUAL.md) · [繁體中文](USERMANUAL_zh-TW.md) · [English](USERMANUAL_en.md) · [日本語](USERMANUAL_ja.md) · [한국어](USERMANUAL_ko.md)
- 擴充功能（Extension）API：[docs/extension-api.md](../../docs/extension-api.md)
- 介面文案風格指南（i18n Style Guide）：[docs/i18n/style-guide.md](../../docs/i18n/style-guide.md)

## 致謝

本專案使用了以下開源函式庫的程式碼或模型結構：

- [WORLD](https://github.com/mmorise/World) - 高品質語音分析與合成系統
- [SoundTouch](https://www.surina.net/soundtouch/) - 音訊時間拉伸與變調函式庫（LGPL）
- [Signalsmith Stretch](https://github.com/Signalsmith-Audio/signalsmith-stretch) - 高品質音訊時間拉伸函式庫（MIT）
- [VocalShifter Library (vslib)](https://ackiesound.ifdef.jp/) - 語音解析與合成函式庫
- [SingingVocoders](https://github.com/openvpi/SingingVocoders) - 歌聲合成聲碼器（OpenVPI）
- [HiFi-GAN](https://github.com/jik876/hifi-gan) - 高保真生成對抗網路聲碼器
- [vocal-remover (hnsep)](https://github.com/stakira/vocal-remover) - 諧波 / 噪聲分離模型（用於氣聲分離）

## 授權條款

本專案基於 [MIT 授權條款](../../LICENSE) 發布。
