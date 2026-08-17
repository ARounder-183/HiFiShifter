# HiFiShifter Backend (Tauri 2.0)

本目录用于承载 HiFiShifter 的 Rust 后端与桌面壳，基于 **Tauri 2.0**。

当前阶段目标：

- 先跑通 Tauri 2.0 桌面壳（Rust commands + 事件），逐步替换原 Python/pywebview 的运行体系。
- 前端 UI 复用仓库根目录的 `frontend/`（Vite + React），本目录不维护独立的 Web UI。

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
