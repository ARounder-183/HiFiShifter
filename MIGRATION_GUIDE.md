# HiFiShifter macOS 迁移指南

本文档记录了将 HiFiShifter 从 Windows 架构移植到 macOS (Apple Silicon) 过程中的核心改造要点。作为后续升级或再次适配时的速查手册。

## 1. 核心架构与功能阉割 (重要前提)
- **vslib 模块**：原始项目依赖了 Windows 专有的 `vslib_x64.dll` 闭源动态库用于提取基频等分析。由于没有 Mac 源码/动态库，该模块被**彻底禁用**。
- **配置修改**：在 macOS 环境下，必须在 `backend/src-tauri/Cargo.toml` 中将 `vslib` 从 default features 列表中移除。
- **回退逻辑**：在解析 VocalShifter (.vsqxX/.vsp) 旧项目时，原指定 `PitchAnalysisAlgo::VocalShifterVslib` 的音轨会在反序列化被强行 Mapping 退化到 `WorldVocoder` (`state.rs::SynthPipelineKind::from_track_algo`)。

## 2. 编译链路与链接环境 (C++)
项目中包含了通过 C++ 编写的特定算法 (`WORLD` 库和 `Signalsmith Stretch`)，在跨平台中需要调整链接器。
- **C++ 库要求**：在 `build.rs` 中编译完被 `cc` wrapper 处理好的 C++ 模块后，使用 `cargo:rustc-link-lib=c++` 手式链接 libc++。因为 macOS 下 clang++ 默认使用 libc++ 而非 libstdc++。
- **vslib 动态库跳过**：`build.rs` 内针对 `cfg!(target_os = "macos")` 需阻止复制和链接不存在的 `.dll` / `.lib`。

## 3. Rust与平台细节
- **Tauri setup 函数中的 unsafe 问题**：Rust 1.83+ 版本以上强制要求 `std::env::set_var` 必须在 `unsafe {}` 块中调用（因为多线程修改环境的不安全性）。需对此类设置模型路径(HNSEP/NSF-HIFIGAN_MODEL_DIR)的方法添加 unsafe 标记。
- **剪贴板实现**：应用支持从 Reaper 或其他宿主粘贴轨信息，Windows 基于 `clipboard-win`，而在 macOS 侧已完美切换到使用系统的 `NSPasteboard` (`objc2-app-kit` crate)。

## 4. 前端UI适配
- **键盘修饰键**：基于使用习惯和原生体验要求，修改了快捷键显示逻辑 (`keybindingsSlice.ts:formatKeybinding`)。在检测到 navigator.platform 包含 mac 时，自动将 `Ctrl` 转换为 `⌘`，`Alt` 转换为 `⌥`，`Shift` 转换为 `⇧`。事件拦截中的 `.metaKey` 处理也与 `Ctrl` 绑定进行了同步。

## 5. 打包与发布选项
- **架构指定**：明确在 `rust-toolchain.toml` 或构建命令中锁定 `aarch64-apple-darwin` (Apple Silicon)。由于 ONNX 依赖中，macOS x86_64 需开启特定 `ort-tract` backend 参数，这在 aarch64 下容易引发依赖链失效。
- **tauri.conf.json 调整**：
  - 取消对 Windows 特有的 `nsis` 安装包支持机制，Bundle `targets` 修改仅保留 `["dmg"]`
  - 移除了 `resources` 资源打包指令中由于 vslib 环境带入的多余依赖

---
**推荐后续验证流程**：
- 执行 `npm install -g @tauri-apps/cli@^2` 安装无网络死锁风险的全平台 Tauri CLI 工具。
- 直接利用 `npx tauri build` 自动化编译并生成 `.app` 和 `.dmg` 分发包。
