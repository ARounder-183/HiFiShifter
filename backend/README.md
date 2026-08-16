# HiFiShifter Backend (Tauri 2.0)

本目录用于承载 HiFiShifter 的 Rust 后端与桌面壳，基于 **Tauri 2.0**。

当前阶段目标：

- 先跑通 Tauri 2.0 桌面壳（Rust commands + 事件），逐步替换原 Python/pywebview 的运行体系。
- 前端 UI 复用仓库根目录的 `frontend/`（Vite + React），本目录不维护独立的 Web UI。

## FFmpeg 依赖（视频媒体导入）

HiFiShifter 通过 `ffmpeg-next` **动态链接** FFmpeg（LGPL 共享库），用于解析视频容器中的音频轨。构建前需要准备 FFmpeg 8.1 的开发包（包含 `include` 与 `lib`）：

- Windows: `pwsh ../scripts/install_ffmpeg_windows.ps1 -Arch x64`（ARM64 使用 `-Arch arm64`）
- Linux: `bash ../scripts/install_ffmpeg_linux.sh x64`（ARM64 使用 `arm64`）
- macOS: `bash ../scripts/install_ffmpeg_macos.sh`

脚本会把库安装到 `third_party/ffmpeg`，并创建 `third_party/ffmpeg/current`（Windows 为目录联接，Unix 为符号链接）。`backend/src-tauri/.cargo/config.toml` 会强制 Cargo 从该固定路径读取 FFmpeg，避免旧的 `FFMPEG_DIR` 环境变量导致二进制链接错误版本的 DLL/dylib；因此构建前只需运行对应供应脚本即可。请勿启用 `ffmpeg-sys-next` 的 `static`/`build` 特性（MIT 许可证要求动态链接 LGPL 库）。

## 开发启动

在仓库根目录确保已安装前端依赖：

```bash
cd frontend
npm install
```

启动 Tauri（会自动执行 `frontend` 的 dev server）：

```bash
cd backend/src-tauri
cargo tauri dev
```

可通过环境变量切换前端模式：

- 默认（`dev`，热更新）：

```bash
cd backend/src-tauri
cargo tauri dev
```

- `build`（先完整构建，再用 preview 提供静态资源）：

```bash
cd backend/src-tauri
TAURI_UI_MODE=build cargo tauri dev
```

Windows PowerShell：

```powershell
cd backend/src-tauri
$env:TAURI_UI_MODE='build'; cargo tauri dev
```

## 最小后端接口（迁移起点）

- `ping` → `{ ok: true, message: "pong" }`
- `get_runtime_info` → 与现有前端类型对齐（`device/model_loaded/audio_loaded/has_synthesized/...`）
- `get_timeline_state` → 返回最小时间线工程（tracks/clips/bpm/playhead/project_beats）
- `set_transport` → `{ ok: true, playhead_beat, bpm }`
- `close_window` → `{ ok: true }`

后续迁移会以这些 commands 为起点，对齐现有 `hifi_shifter/web_api.py` 的接口形状。
