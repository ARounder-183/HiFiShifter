# HiFiShifter Backend Workspace

`backend/` 是 HiFiShifter 的统一 Rust Workspace（`resolver = "2"`），同时支撑**独立桌面 App（Tauri 2）**、**VST3 + ARA2 插件**、**CLAP 插件**以及**本机 Agent MCP 服务**，共用单一 `Cargo.lock` 与编译期 `profile` 配置。

## Workspace 成员一览

| 目录 | 包名 | 产物与职责 |
| :--- | :--- | :--- |
| `hifishifter-kernel/` | `hifishifter-kernel` | 离线核心：时间线与 Take 模型、参数权威、World / PC-NSF-HiFiGAN / VsLib 声码器、Signalsmith / SoundTouch 拉伸、离线混音 `mixdown`、宿主音高包络换算 `pitch_bridge`。不依赖 Tauri 或音频设备。 |
| `src-tauri/` | `HiFiShifter` (`backend_lib`) | 独立桌面 App 外壳（Tauri 2）、`cpal` 音频播放与录音引擎、前端 IPC 命令入口（`commands.rs`）及 App MCP 运行时（`src/mcp/`）。 |
| `hifishifter-plugin/` | `hifishifter-plugin` | VST3 与 CLAP 共用的格式无关插件实现层（`rlib`）：ARA 文档映射、捕获协同、内嵌 WebView 编辑器、REAPER 原生扩展写回。 |
| `hifishifter-plugin-core/` | `hifishifter-plugin-core` | 跨插件格式的常量 ID、资源根定位与毒化恢复锁原语（`MutexExt::lock_or_recover`）。 |
| `hifishifter-vst3/` | `hifishifter-vst3` | VST3 + ARA2 `cdylib` 外壳（需通过 `scripts/setup/prepare-plugin-sdks.ps1` 或 `scripts/vst3/unix/build.sh --stage=sdk` 准备 SDK）。 |
| `hifishifter-clap/` | `hifishifter-clap` | CLAP `cdylib` 外壳（ARA 定稿协议 + Capture 双模式；测试/构建时需独立 `-p hifishifter-clap`，避免与 `src-tauri` 合并构建污染 `vslib` 导入表）。 |
| `hifishifter-ara/` | `hifishifter-ara` | ARA2 绑定与桥接辅助层。 |
| `hifishifter-capture/` | `hifishifter-capture` | 实时效果器捕获录音、侧车持久化与循环段（Loop Pass）合成投影。 |
| `hifishifter-host-link/` | `hifishifter-host-link` | 无锁双文件原子工作区交接协议（`manifest.hfslink` + `edits.hfsedit` + 采样级 BLAKE3 去重浮点 WAV）。 |
| `hifishifter-clipboard/` | `hifishifter-clipboard` | REAPER、VocalShifter 与 MIDI 系统剪贴板双向编解码。 |
| `hifishifter-mcp/` | `hifishifter-mcp` | 基于 `rmcp` 的本机 Streamable HTTP + `stdio` 桥接 MCP 协议层、多实例发现注册表、统一句段音源标记契约 `voicebank` 与强类型工具 Schema。 |

> 另：`backend/hfs-gui/` 为独立于本 Workspace 的 Unix 插件独立渲染进程包，`cargo fmt` 时需单独包含其 `Cargo.toml`。

## 常用开发与验证命令

在仓库根目录执行：

```bash
# 1. 秒级跨平台静态门禁（FFI panic 守卫、锁与错误码基线、IPC 接线、UTF-8 等）
python3 scripts/checks/run-fast-gates.py

# 2. 按当前 git 改动自动路由最小测试集
python3 scripts/checks/verify-changed.py --run

# 3. 单独运行 MCP 协议与无头编辑器集成测试（零模型权重可跑）
cargo test --locked --manifest-path backend/Cargo.toml -p hifishifter-mcp
cargo test --locked --manifest-path backend/Cargo.toml -p HiFiShifter --no-default-features --lib mcp::

# 4. 启动独立桌面 App 开发模式（可选开启本机 MCP）
cd backend/src-tauri
cargo tauri dev -- -- --enable-mcp
```

完整线程/锁/FFI 铁律请参阅 [`../docs/thread-model.md`](../docs/thread-model.md)、宿主能力边界请参阅 [`../docs/host-extension-boundary.md`](../docs/host-extension-boundary.md)、Agent 协作规范请参阅 [`../AGENTS.md`](../AGENTS.md) 与 [`../docs/mcp.md`](../docs/mcp.md)。
