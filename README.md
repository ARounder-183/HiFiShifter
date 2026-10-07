# HiFiShifter

[简体中文](README.md) | [繁體中文](docs/i18n/README_zh-TW.md) | [English](docs/i18n/README_en.md) | [日本語](docs/i18n/README_ja.md) | [한국어](docs/i18n/README_ko.md)

HiFiShifter 是一个图形化人声编辑与合成工具。它支持多轨道音频块处理，并以轨道组为单位，使用多种声码器完成人声修音、人力调参功能，实现人力 VOCALOID 制作的拼调一体化。

**当前项目仍在开发迭代中，未对全链路进行测试，可能存在诸多 BUG 或不稳定问题。**

![预览图](docs/preview.png)

## 功能一览

- **轨道编辑**：类 DAW 的多轨时间轴，支持音频块的裁剪、拉伸、Slip、淡入淡出与交叉淡化、编组、Take 管理、静音检测、波纹编辑、速度映射（BPM / 拍号 / 音阶）、节拍器与录音等。
- **参数编辑**：以轨道组为单位，通过钢琴卷帘式参数编辑器调整音高、音量、动态、声像、共振峰、气声、张力等参数线；支持绘制 / 直线 / 颤音工具、多选区编辑与颤音预设管理，子轨道可叠加 `音分差` / `度数差` / `共振峰差` 制作和声。
- **三种声码器算法**：nsf-hifigan（PC-NSF-HiFiGAN）、World、VsLib，详见[算法](#算法)。
- **互操作性**：导入 REAPER（`.rpp`）与 VocalShifter（`.vshp` / `.vsp`）工程，双向读写 REAPER 与 VocalShifter 剪贴板；MIDI 可导入为音高参考块 / 音高参数 / 速度映射，音频块与音高线可导出为 MIDI。
- **导入导出**：常见音频 / 视频格式导入（视频自动提取音轨），工程与分轨导出为 `wav` / `mp3` / `flac`。
- **其他**：内置文件浏览器与快速搜索、记事本、自动备份、多语言界面（简体中文 / 繁體中文 / English / 日本語 / 한국어）、深浅色主题、推理设备选择与基准测试。

## 安装

从 [Releases](https://github.com/ARounder-183/HiFiShifter/releases) 页面下载对应操作系统与架构的安装包：

- **Windows**：NSIS 安装包（`installer`）或便携版压缩包（`portable`），提供 x86_64 与 arm64 架构。
- **macOS**：未签名 dmg（Apple Silicon 装 `arm64`，Intel 装 `x86_64`）。首次安装需要手动放行；若提示"文件已损坏"，请按[使用手册](docs/i18n/USERMANUAL.md#一安装)中的步骤处理。
- **Linux**：AppImage（x86_64 / arm64）。

GPU 加速：Windows 使用 DirectML（DirectX 12），macOS（Apple Silicon）使用 CoreML + WebGPU，Linux x86_64 使用 WebGPU（Dawn/Vulkan），其余平台回退 CPU。可在应用内 `选项 → 推理设备` 中切换设备并运行基准测试。

## 基本原理

HiFiShifter 使用类似 UTAU 的离线渲染方式，对时间线中的每个音频块进行处理、渲染、缓存，最后再输入到播放系统中，因此其对短音频块有着更快的处理效率。

HiFiShifter 提供了一个统一的渲染接口，以便未来增添更多的算法支持。

## 推荐工作流

我们推荐的工作流是：

1. 通过其他 DAW 或切片软件准备好人力所需的短切片音源
2. 在 HiFiShifter 中完成音频的拼贴和调音

当然，HiFiShifter 也支持以下操作方便从其他软件的工程迁移：

1. 直接打开 VocalShifter 工程
2. 直接打开 Reaper 工程
3. 解析 VocalShifter 剪贴板内容，支持将 VocalShifter 中的参数粘贴到 HiFiShifter 参数区中。
4. 解析 Reaper 剪贴板内容，支持直接将 Reaper 的 Items 粘贴到 HiFiShifter 中

## 界面与算法

### 布局

HiFiShifter 大致分为上部的轨道面板和下部的参数面板：轨道面板负责音频块的编辑与编排，参数面板负责对音频进行调参处理。各面板可停靠、浮动与重排（`视图 → 窗口 / 布局`）。

轨道支持嵌套：将一个轨道拖拽到另一个轨道下，即可组成轨道组（根轨道 + 子轨道）。轨道组共用一个算法和一套参数线，参数线会按位置作用到组内每一个音频块上；调参前需要先按下轨道的合成按钮 `C`。

### 算法

目前 HiFiShifter 支持三种算法进行处理：

- **World**：老牌声码器。支持 `音高`、`音量`、`动态`、`声像` 参数的编辑。
- **PC-NSF-HiFiGAN**（界面算法列表中显示为 `nsf-hifigan`）：OpenVPI 开源的歌声特化 HiFi-GAN 声码器，也是默认算法。支持 `音高`、`共振峰偏移`、`气声音量`、`张力`、`音量`、`动态`、`声像` 参数的编辑。其中 `气声音量` 与 `张力` 依赖 `气声分离` 开关：开启后会把音频块分离为谐波与噪声两部分（使用 hnsep 模型），**会增加额外的渲染成本**（首次每个音频块都要做一次分离，可能较慢）；关闭时则完全不进行分离，这两个参数会被置灰、其曲线保持可见但不可编辑，也不参与合成。
- **VsLib**（界面算法列表中显示为 `vslib`）：VocalShifter 官方提供的算法库。支持 `音高`、`共振峰偏移`、`气声强度`、`音量`、`动态`、`声像` 与 `合成模式` 参数的编辑。**仅 Windows x86_64 版本可用**；由于官方提供的 dll 仅支持文件 I/O，相对 VocalShifter 本体需要更多的时间处理。

轨道面板与参数面板的详细操作（淡化编辑、吸附设置、颤音预设、导出与录音等）请阅读[使用手册](docs/i18n/USERMANUAL.md)。

## 常用快捷键速查

> 快捷键均可在应用内 `选项 → 快捷键设置` 中自定义，下表为默认值；macOS 上 `Ctrl` 对应 `⌘`、`Alt` 对应 `⌥`。

| 操作                           | 快捷键 / 鼠标                                        |
| :----------------------------- | :--------------------------------------------------- |
| 播放 / 暂停（不返回起播点）    | `Space`                                              |
| 播放 / 停止（返回起播点）      | `Enter`                                              |
| 开关节拍器                     | `K`                                                  |
| 平移视图                       | 鼠标中键拖动                                         |
| 滚动时间轴                     | 鼠标滚轮（双轴自由滚动）                             |
| 横向 / 纵向滚动                | `Shift` / `Alt` + 鼠标滚轮                           |
| 缩放轨道高度                   | `Ctrl` + 鼠标滚轮                                    |
| 撤销 / 重做                    | `Ctrl + Z` / `Ctrl + Shift + Z`（或 `Ctrl + Y`）     |
| 剪切 / 复制 / 粘贴             | `Ctrl + X` / `Ctrl + C` / `Ctrl + V`                 |
| 删除选中音频块                 | `Delete`                                             |
| 分割音频块（播放头处）         | `S`                                                  |
| 编组 / 解组                    | `G` / `U`                                            |
| 循环切换 Take                  | `T`（`Shift + T` 切换上一个）                        |
| 框选多选                       | 在时间线空白处按住鼠标右键拖动                       |
| 复制拖动                       | 按住 `Ctrl` 拖动音频块                               |
| 拉伸音频块 / Slip 编辑         | 按住 `Alt` 拖动音频块边缘 / 主体                     |
| 临时切换吸附                   | 按住 `Shift` 拖动                                    |
| 新建工程 / 打开工程            | `Ctrl + N` / `Ctrl + Shift + O`                      |
| 导入媒体文件 / 导出音频        | `Ctrl + O` / `Ctrl + E`                              |
| 保存 / 另存为                  | `Ctrl + S` / `Ctrl + Shift + S`                      |
| 新建轨道 / 录音                | `Ctrl + T` / `Ctrl + R`                              |
| 模式切换（选择 ↔ 绘制类工具）  | `Tab`                                                |
| 快速搜索                       | `Ctrl + F`                                           |
| 参数线整体上移 / 下移          | `=` / `-`（选中范围用 `[` / `]`），`Shift` 大幅、`Ctrl` 微调 |
| 参数编辑器多选区               | 按住 `Ctrl` 拖动追加选区、点击已有选区取消           |

完整快捷键与修饰键列表见[使用手册](docs/i18n/USERMANUAL.md#三轨道界面)或应用内快捷键设置。

## 开发

该部分内容为开发者提供，普通用户可以跳过。

### 1. 克隆仓库

```bash
git clone https://github.com/ARounder-183/HiFiShifter.git
cd HiFiShifter
```

### 2. 安装依赖

#### Windows

请确保已安装以下工具：

- **Node.js**（建议 20+，CI 使用 24）及 npm
- **Rust 工具链**（参见 `rust-toolchain.toml`）
- **Tauri 2 CLI**：`cargo install tauri-cli --version "^2"`
- **CMake**（用于编译 SoundTouch 库）

ONNX Runtime (DirectML) 由 ort crate 在编译时自动下载，无需额外配置。

也可以运行 `scripts/install_deps_windows.ps1`，通过 Chocolatey 安装 NSIS / CMake / LLVM 并安装 tauri-cli（需要已安装 Chocolatey）。

安装前端依赖：

```bash
npm --prefix frontend install
```

#### macOS

```bash
chmod +x ./scripts/install_deps_macos.sh
SKIP_FRONTEND=0 bash ./scripts/install_deps_macos.sh
```

#### Linux

请确保已安装以下工具：

- **Node.js**（建议 20+，CI 使用 24）及 npm
- **Rust 工具链**（参见 `rust-toolchain.toml`，项目会自动选择对应平台的 stable 工具链）
- **Tauri 2 CLI**：`cargo install tauri-cli --version "^2"`
- **CMake**、**pkg-config** 及系统构建工具
- **GTK3、WebKit2GTK、ALSA** 等 Tauri 运行时开发库（详见安装脚本）

运行一键安装脚本：

```bash
chmod +x ./scripts/install_deps_linux.sh
bash ./scripts/install_deps_linux.sh
```

脚本会自动安装系统依赖、Node.js（如未安装）、appimagetool 及前端 npm 依赖。

安装前端依赖（如未使用脚本）：

```bash
npm --prefix frontend ci
```

### 3. 第三方源码

SoundTouch、WORLD、Signalsmith Stretch 均在编译时从源码构建，首次构建时会**自动克隆**，无需手动操作。

如需离线构建，可提前手动克隆，例如 SoundTouch：

```bash
cd backend/src-tauri/third_party/soundtouch-static
git clone --depth 1 --branch 2.3.3 https://codeberg.org/soundtouch/soundtouch.git soundtouch
```

### 4. 开发与构建

```bash
# 开发模式（热更新）
cd backend
cargo tauri dev

# 构建 Release
# Windows / macOS（默认 features：onnx + vslib）
cargo tauri build

# Linux AppImage（vslib 仅限 Windows，需排除默认 feature）
cargo tauri build --bundles appimage -- --no-default-features --features onnx
# 或使用辅助脚本：bash scripts/build-linux-appimage.sh

# Windows 便携版 ZIP
.\scripts\pack-portable.ps1 -SkipBuild

# Windows VST3 ZIP + 安装器（已有完整 Release 交付）
.\scripts\pack-portable.ps1 -PackageTarget Plugin -SkipBuild -Installer
```

双击 `pack-portable.bat` 可选择 App、VST3 或两者。插件构建、安装目录及 GitHub Actions
产物说明见 [VST3-BUILD.md](docs/VST3-BUILD.md)。

前端启动模式可通过环境变量 `TAURI_UI_MODE` 切换：

- `dev`：开发模式（默认，使用 Vite dev server，支持热更新）
- `build`：构建模式（先构建前端静态资源，再启动）

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

**注意：** 首次编译需要很长的时间，请耐心等待。

### 5. GPU 加速

HiFiShifter 在支持的平台上自动启用 GPU 推理加速。你可以在菜单栏中的 **推理设备（Inference Device）** 里选择 Auto / CPU / GPU，并通过 **运行基准测试（Run Benchmark）** 比较各设备的推理延迟。

| 平台                        | GPU 技术                     | 说明                                                      |
| --------------------------- | ---------------------------- | --------------------------------------------------------- |
| Windows x86_64 / ARM64      | DirectML (DirectX 12)        | 成熟稳定的 GPU 路径，支持 NVIDIA / AMD / Intel Arc        |
| macOS ARM64 (Apple Silicon) | CoreML + WebGPU (Dawn/Metal) | CoreML 利用 Apple Neural Engine；WebGPU 作为补充 GPU 后端 |
| macOS x86_64 (Intel)        | —                            | CPU only（使用 ort-tract 替代后端）                       |
| Linux x86_64                | WebGPU (Dawn/Vulkan)         | Dawn 通过 Vulkan API 使用 GPU；无 GPU 时自动回退到 CPU    |
| Linux ARM64                 | —                            | CPU only（暂无预编译 WebGPU ONNX Runtime 二进制文件）     |

> **注意**：Windows 平台暂不启用 WebGPU。其 Dawn/D3D12 后端在部分 GPU/驱动组合上存在原生崩溃风险。DirectML 是 Windows 上成熟稳定的 GPU 路径。

ONNX Runtime 二进制文件由 ort crate 在编译时通过 `download-binaries` 特性自动下载，无需手动安装。GPU 提供程序（DirectML / WebGPU / CoreML）的代码在编译时根据目标平台自动启用，无需额外的 `--features` 标志。

**WSL2 用户**：

- WSL2 不向 Linux 子环境暴露硬件 Vulkan。WebGPU/Dawn 只能使用 Lavapipe（CPU 软件渲染），性能极差。如需 GPU 加速，请使用 Windows 原生版本（DirectML）。
- 因缺少 FUSE 支持，Tauri bundler 的 linuxdeploy 步骤可能失败（错误：`failed to run linuxdeploy`）。这是 WSL2 已知限制，不影响实际 AppImage 产出——AppDir 已正确组装在 `target/release/bundle/appimage/` 中。可设置 `APPIMAGE_EXTRACT_AND_RUN=1` 后手动运行 `appimagetool` 打包。在真实 Linux 机器和 CI 中不存在此问题。

## 日志与故障排查

应用会自动把运行日志写入系统标准日志目录，无需任何命令行参数：

| 系统 | 日志目录 |
| --- | --- |
| Windows | `%LOCALAPPDATA%\com.arounder.hifishifter\logs` |
| macOS | `~/Library/Logs/com.arounder.hifishifter` |
| Linux | `~/.local/share/com.arounder.hifishifter/logs` |

- 在应用内通过 **帮助 → 打开日志文件夹** 可以直接定位日志；**帮助 → 导出诊断信息** 可以一键生成诊断包（系统信息 + 全部日志 + 推理设备基准测试结果），提交 issue 时附上即可。
- 日志按大小自动轮转：单个文件上限 8 MiB，默认保留 3 份历史（`hifishifter.1.log` ~ `hifishifter.3.log`）。
  - 高频重复的错误 / 警告会自动限流：同一位置的日志默认每 10 秒最多输出一条，被抑制的条数会在下一条输出前以 `[throttled]` 汇总行补记。
- 前端与后端的错误都会统一记录在同一份日志文件里，方便按时间轴对照排查。

高级选项：

- 启动参数 `--log-file=<path>`：把日志写到指定路径；`--log-file=-` 显式关闭文件日志。
- 环境变量 `HIFISHIFTER_LOG=debug`（或 `trace` / `info` / `warn` / `error`）：调整日志详细程度。
- 环境变量 `HIFISHIFTER_LOG_DIR`：覆盖默认日志目录。
- 终端运行 `HiFiShifter --benchmark`：直接执行推理设备基准测试并输出 JSON 结果。

## 文档

- 使用手册：[简体中文](docs/i18n/USERMANUAL.md) · [繁體中文](docs/i18n/USERMANUAL_zh-TW.md) · [English](docs/i18n/USERMANUAL_en.md) · [日本語](docs/i18n/USERMANUAL_ja.md) · [한국어](docs/i18n/USERMANUAL_ko.md)
- 扩展（Extension）API：[docs/extension-api.md](docs/extension-api.md)
- 界面文案风格指南（i18n Style Guide）：[docs/i18n/style-guide.md](docs/i18n/style-guide.md)

## 致谢

本项目使用了以下开源库的代码或模型结构：

- [WORLD](https://github.com/mmorise/World) - 高质量语音分析与合成系统
- [SoundTouch](https://www.surina.net/soundtouch/) - 音频时间拉伸与变调库（LGPL）
- [Signalsmith Stretch](https://github.com/Signalsmith-Audio/signalsmith-stretch) - 高质量音频时间拉伸库（MIT）
- [VocalShifter Library (vslib)](https://ackiesound.ifdef.jp/) - 音声解析与合成库
- [SingingVocoders](https://github.com/openvpi/SingingVocoders) - 歌声合成声码器（OpenVPI）
- [HiFi-GAN](https://github.com/jik876/hifi-gan) - 高保真生成对抗网络声码器
- [vocal-remover (hnsep)](https://github.com/stakira/vocal-remover) - 谐波 / 噪声分离模型（用于气声分离）

## License

本项目基于 [MIT License](LICENSE) 发布。
