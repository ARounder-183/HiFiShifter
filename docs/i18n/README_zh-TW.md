# HiFiShifter

[简体中文](../../README.md) | [繁體中文](README_zh-TW.md) | [English](README_en.md) | [日本語](README_ja.md) | [한국어](README_ko.md)

HiFiShifter 是一個圖形化人聲編輯與合成工具。它支援多軌道音訊片段處理，並以軌道組為單位，使用多種聲碼器完成人聲修音、人力調參功能，實現人力 VOCALOID 製作的拼調一體化。

**當前專案仍在開發迭代中，未對全鏈路進行測試，可能存在諸多 BUG 或不穩定問題。**

![預覽圖](../preview.png)

## 安裝

請直接在倉庫側邊選擇適合自己系統的 Release 版本下載安裝。

## 基本原理

HiFiShifter 使用類似 UTAU 的離線渲染方式，對時間線中的每個音訊片段進行處理、渲染、快取，最後再輸入到播放系統中，因此其對短片段有著更快的處理效率。

HiFiShifter 提供了一個統一的渲染介面，以便未來增添更多的演算法支援。

## 工作流推薦

我們推薦的工作流是：

1. 透過其他 DAW 或片段軟體準備好人力所需的短片段音源。
2. 在 HiFiShifter 中完成音訊的拼貼和調音。

當然，HiFiShifter 也支援以下操作方便從其他軟體的專案遷移：

1. 直接開啟 VocalShifter 專案。
2. 直接開啟 Reaper 專案。
3. 解析 VocalShifter 剪貼簿內容，支援將 VocalShifter 中的參數貼到 HiFiShifter 參數區中。
4. 解析 Reaper 剪貼簿內容，支援直接將 Reaper 的 Items 貼到 HiFiShifter 中。

## 功能介紹

### 佈局介紹

HiFiShifter 可以大致分為兩個功能區，分別是上部的軌道面板和下部的參數面板。軌道面板主要負責對音訊的片段處理，參數面板則負責對音訊進行調參處理。

### 軌道面板

HiFiShifter 提供了一個基本完備的軌道面板與音訊片段功能。該功能與大多數現代 DAW 類似。

#### 音訊匯入

HiFiShifter 支援三種方式匯入音訊：

1. 直接從系統檔案管理器中拖曳音訊到軌道上。
2. 點擊工具列的資料夾圖示，開啟內建檔案瀏覽器並拖曳音訊到軌道上。
3. 按下 `Ctrl + F` 開啟快速搜尋，選擇音訊匯入到軌道上（快速搜尋的檔案路徑與內建檔案瀏覽器的目前路徑一致）。

#### 音訊編輯

- **吸附格線**：剪輯移動/裁切預設吸附格線；按住 `Shift` 可暫時關閉吸附。
- **裁切/伸縮範圍**：拖動剪輯左右邊界進行裁切或延長。
- **伸縮（Time Stretch）**：按住 `Alt` + 滑鼠左鍵拖動剪輯左右邊界，可伸縮音訊。
- **內部偏移（Slip-Edit）**：按住 `Alt` + 滑鼠左鍵拖動剪輯主體，可左右滑移剪輯的內部內容。
- **淡入淡出**：拖動剪輯左上角/右上角調整淡入/淡出時長。
- **增益（dB）**：拖動剪輯左上角的旋鈕（上下拖動）調整增益，剪輯右上角會顯示目前 dB。
- **剪輯靜音（M）**：剪輯左上角 `M` 按鈕可對該剪輯靜音，靜音後剪輯整體變灰。
- **框選多選**：在時間線空白處按住滑鼠右鍵拖曳可框選多個剪輯。
- **複製拖動**：按住 `Ctrl` 後拖曳剪輯，會在目標位置建立副本並保持原剪輯不動（複製完成在鬆手時生效）。
- **膠合**：右鍵剪輯開啟選單，選擇「膠合」（要求同一軌道且至少 2 個剪輯）。
- **切分**：選中剪輯後按 `S` 可在播放頭位置切分。
- **複製貼上**：選中剪輯後按 `Ctrl + C` 將選中剪輯複製到應用內剪貼簿。`Ctrl + V` 會把「所選剪輯中最靠左的起點」對齊到播放頭位置，其餘剪輯保持相對間距。

需要特別注意的，軌道支援巢狀，可以將軌道拖動到另一個軌道下成為該軌道的子軌道，形成一個軌道組。在接下來的調參過程中，軌道組將十分有用。

### 參數面板

HiFiShifter 的參數面板提供了類似 VocalShifter 的操作支援以方便使用者調整參數。

需要注意的是，HiFiShifter 的軌道上有一個特殊的 `C` 按鈕，只有按下這個按鈕，該軌道上的音訊才能被後續調參處理。

在調參中，HiFiShifter 以軌道組為單位，透過根軌道開啟 `C` 來決定，一個軌道組共用一個演算法和一套參數線。參數線會按位置作用到每一個音訊片段上。

HiFiShifter 中的每個演算法都有不同的參數可供調整，其中通用參數為音高。

在首次打開時，HiFiShifter 需要一些時間對片段的音高進行分析。分析完成後，面板中的實線表示該軌道組的整體目前音高，虛線表示整體原始音高，彩線表示每個片段自己的原始音高。

其他面板與音高面板類似，只是不會顯示片段自己的原始音高。

面板旁邊的小眼睛可以開啟該面板在未選中下的可見性。

### 演算法

目前 HiFiShifter 支援三種演算法進行處理。

#### World 演算法

老牌聲碼器。  
僅支援 `音高` 編輯。

#### PC-NSF-HiFiGAN

OpenVPI 開源的為歌聲特化的 hifigan 聲碼器。  
支援 `音高`、`氣聲`、`張力`、`共振峰偏移`、`音量` 參數的編輯。  
需要注意的是，氣聲的編輯需要額外開啟，將會使用 hnsep 的 UVR 模型對片段進行氣聲分離，首次需要較長的時間處理。如果需要編輯張力請務必開啟氣聲。

#### Vslib

VocalShifter 提供的演算法庫。  
支援 `音高`、`聲像`、`共振峰偏移`、`音量`、`氣聲` 參數的編輯。  
由於官方提供的 dll 僅支援檔案 I/O，因此相對 VocalShifter 本體需要更多的時間處理。

## 常用快捷鍵速查

| 操作                         | 快捷鍵 / 滑鼠                   |
| :--------------------------- | :------------------------------ |
| 平移檢視（時間軸）           | 滑鼠中鍵拖曳                    |
| 橫向縮放（時間軸）           | 滑鼠滾輪（以游標為中心）        |
| 縱向縮放（軌道高度，時間軸） | Ctrl + 滑鼠滾輪                 |
| 縱向縮放（參數軸，參數面板） | Ctrl + 滑鼠滾輪（參數面板內）   |
| 播放/暫停                    | Space（空格鍵）                 |
| 播放/停止                    | Enter                           |
| 復原/重做                    | Ctrl + Z / Ctrl + Y             |
| 新建專案                     | Ctrl + N                        |
| 開啟專案                     | Ctrl + Shift + O                |
| 儲存                         | Ctrl + S                        |
| 另存新檔                     | Ctrl + Shift + S                |
| 匯出音訊                     | Ctrl + E                        |
| 模式切換（選取/繪製）        | Tab                             |
| 刪除選中剪輯                 | Delete                          |
| 複製選中剪輯（應用內剪貼簿） | Ctrl + C                        |
| 貼到播放頭位置               | Ctrl + V                        |
| 參數面板複製選區曲線         | Ctrl + C（Select 模式）         |
| 參數面板貼到選區起點         | Ctrl + V（Select 模式）         |
| 分割剪輯                     | S（在播放頭位置分割選中的剪輯） |
| 新建軌道                     | Ctrl + T                        |
| 快速搜尋                     | Ctrl + F                        |

## 開發環境配置

該部分內容為開發者提供，普通使用者可以跳過。

### 1. 克隆倉庫

```bash
git clone https://github.com/ARounder-183/HiFiShifter.git
cd HiFiShifter
```

### 2. 安裝依賴

#### Windows

請確保已安裝以下工具：

- **Node.js**（建議 18+）及 npm
- **Rust 工具鏈**（參見 `rust-toolchain.toml`）
- **Tauri 2 CLI**：`cargo install tauri-cli --version "^2"`
- **CMake**（用於編譯 SoundTouch 函式庫）

ONNX Runtime (DirectML) 由 ort crate 在編譯時自動下載，無需額外設定。

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

- **Node.js**（建議 20+）及 npm
- **Rust 工具鏈**（參見 `rust-toolchain.toml`，專案會自動選擇對應平台的 stable 工具鏈）
- **Tauri 2 CLI**：`cargo install tauri-cli --version "^2"`
- **CMake**、**pkg-config** 及系統構建工具
- **GTK3、WebKit2GTK、ALSA** 等 Tauri 執行時開發函式庫（詳見安裝腳本）

執行一鍵安裝腳本：

```bash
chmod +x ./scripts/install_deps_linux.sh
sudo bash ./scripts/install_deps_linux.sh
```

腳本會自動安裝系統依賴、Node.js（如未安裝）、appimagetool 及前端 npm 依賴。

安裝前端依賴（如未使用腳本）：

```bash
npm --prefix frontend ci
```

#### Linux AppImage 構建

由於 `vslib` 演算法僅限 Windows，Linux 構建需要禁用預設 feature：

```bash
# 進入 backend 目錄執行（tauri.conf.json 中路徑相對於此目錄）
cd backend
cargo tauri build --bundles appimage -- --no-default-features --features onnx
```

或使用提供的輔助腳本：

```bash
bash scripts/build-linux-appimage.sh
```

> **注意：** WSL2 環境下因缺少 FUSE 支援，Tauri bundler 的 linuxdeploy 步驟可能失敗（錯誤：`failed to run linuxdeploy`）。這是 WSL2 已知限制，不影響實際 AppImage 產出——AppDir 已正確組裝在 `target/release/bundle/appimage/` 中。可設定 `APPIMAGE_EXTRACT_AND_RUN=1` 後手動執行 `appimagetool` 打包。在真實 Linux 機器與 CI 中不存在此問題。

### 3. SoundTouch 原始碼

SoundTouch 音訊時間拉伸函式庫在編譯時從原始碼構建。首次構建時會**自動克隆**，無需手動操作。

如需離線構建，可提前手動克隆：

```bash
cd backend/src-tauri/third_party/soundtouch-static
git clone --depth 1 --branch 2.3.3 https://codeberg.org/soundtouch/soundtouch.git soundtouch
```

### 4. GPU 加速

HiFiShifter 在支援的平台上自動啟用 GPU 推理加速。你可以在選單列的**推理裝置（Inference Device）**中選擇 Auto / CPU / GPU，並透過**執行基準測試（Run Benchmark）**比較各裝置的推理延遲。

| 平台                            | GPU 技術                             | 說明                                                                           |
| ------------------------------- | ------------------------------------ | ------------------------------------------------------------------------------ |
| Windows x86_64 / ARM64          | DirectML (DirectX 12)                | 成熟穩定的 GPU 路徑，支援 NVIDIA / AMD / Intel Arc                             |
| macOS ARM64 (Apple Silicon)     | CoreML + WebGPU (Dawn/Metal)         | CoreML 利用 Apple Neural Engine；WebGPU 作為補充 GPU 後端                      |
| macOS x86_64 (Intel)            | —                                    | 僅 CPU（使用 ort-tract 替代後端）                                              |
| Linux x86_64                    | WebGPU (Dawn/Vulkan)                 | Dawn 透過 Vulkan API 存取 GPU；無 GPU 時自動回退到 CPU                         |
| Linux ARM64                     | —                                    | 僅 CPU（此目標尚無預編譯的 WebGPU ONNX Runtime 二進位檔案）                    |

> **注意**：Windows 平台未啟用 WebGPU。其 Dawn/D3D12 後端在部分 GPU/驅動組合上存在原生崩潰風險。DirectML 是 Windows 上成熟穩定的 GPU 路徑。
>
> **WSL2 使用者**：WSL2 不向 Linux 子環境暴露硬體 Vulkan。WebGPU/Dawn 只能使用 Lavapipe（CPU 軟體渲染），效能極差。如需 GPU 加速，請使用 Windows 原生版本的 DirectML。

#### 所有平台

ONNX Runtime 二進位檔案由 ort crate 在編譯時透過 `download-binaries` 特性自動下載，無需手動設定。GPU 提供程序（DirectML / WebGPU / CoreML）在編譯時根據目標平台自動啟用，無需額外的 `--features` 標誌。

```bash
# 開發模式（熱更新）
cd backend
cargo tauri dev

# 構建 Release
# Windows / macOS（預設 features：onnx + vslib）
cargo tauri build

# Linux（vslib 僅限 Windows，需排除預設 feature）
cargo tauri build --bundles appimage -- --no-default-features --features onnx

# Windows 可攜版 ZIP
.\scripts\pack-portable.ps1 -SkipBuild
```

## 快速開始

### 執行開發模式

```bash
cd backend/src-tauri
cargo tauri dev
```

前端啟動模式可透過環境變數 `TAURI_UI_MODE` 切換：

- `dev`：開發模式（預設，使用 Vite dev server，支援熱更新）
- `build`：建置模式（先建置前端靜態資源，再啟動）

Linux/macOS（bash/zsh）：

```bash
cd backend/src-tauri
TAURI_UI_MODE=build cargo tauri dev
```

Windows PowerShell：

```powershell
cd backend/src-tauri
$env:TAURI_UI_MODE='build'; cargo tauri dev
```

**注意：** 首次編譯需要很長的時間，請耐心等待。

## 文件

- [使用手冊](USERMANUAL_zh-TW.md)
- [更新計畫](../../todo.md)

## 致謝

本專案使用了以下開源函式庫的程式碼或模型結構：

- [WORLD](https://github.com/mmorise/World) - 高品質語音分析與合成系統
- [SoundTouch](https://www.surina.net/soundtouch/) - 音訊時間拉伸與變調函式庫（LGPL）
- [Signalsmith Stretch](https://github.com/Signalsmith-Audio/signalsmith-stretch) - 高品質音訊時間拉伸函式庫（MIT）
- [VocalShifter Library (vslib)](https://ackiesound.ifdef.jp/) - 語音解析與合成函式庫
- [SingingVocoders](https://github.com/openvpi/SingingVocoders) - 歌聲合成聲碼器（OpenVPI）
- [HiFi-GAN](https://github.com/jik876/hifi-gan) - 高保真生成對抗網路聲碼器

## 授權條款

本專案基於 [MIT 授權條款](../../LICENSE) 發布。
