# HiFiShifter VST3

GitHub Actions 的 **HiFiShifter VST3 (Windows x64)** 流程在 main/develop 推送、目标为
main/develop 的 PR、v* 标签及手动触发时构建。当前打包平台为 Windows x64；插件的
原生 WebView 编辑器尚未提供 macOS/Linux 交付流程。

下载 Actions 中的 `HiFiShifter-windows-x86_64-vst3` artifact，使用其中的 `-setup.exe`
安装器，或解开产品 ZIP 手动安装。安装器默认放入系统 VST3 目录，也可选择 D:\VST；
写入前会检查 REAPER 和它的插件宿主进程已退出。
将完整 `HiFiShifter.vst3` 目录放入宿主扫描的 VST3 目录，例如 `D:\VST` 或
`C:\Program Files\Common Files\VST3`。安装或替换前完全退出 REAPER。
请保留 Contents 内的 DLL、前端资源和模型，不能只复制同名二进制文件。

包内包含 LICENSE、构建说明和逐文件 SHA256 manifest；ZIP 旁另有 `.sha256`。
setup.exe 同样附带 SHA256；两种格式都包含完整的目录内容。
该流程验证前端、插件单元测试和完整资源包，不代替 REAPER 的真实 GUI 验收。
插件不包含独立 App 专用的 vslib 运行库。

## 快速打包

双击根目录 `pack-portable.bat` 可选择 App、VST3 或两者，然后选择重新构建还是直接
打包已有产物。默认生成 ZIP；加 `-Installer` 可同时生成 NSIS 安装器：

```powershell
.\pack-portable.bat -PackageTarget Plugin -SkipBuild -Installer
.\scripts\pack-portable.ps1 -PackageTarget All -SkipBuild
```

`-SkipBuild` 的插件路径选择最新完整 Release 交付，也可使用 `-DeliveryDirectory`
明确指定；`-NoZip` 保留目录。产物默认写入根目录 `dist`。已有的 App 命令和 CI
`-TargetTriple` 参数保持可用；workspace 的产物从 `backend/target` 读取，并兼容旧
`backend/src-tauri/target` 布局。完整构建 All 时前端只构建一次。

在 Windows checkout 中本地复现：

```powershell
.\tools\prepare-plugin-sdks.ps1
npm --prefix frontend ci
cargo fetch --manifest-path backend\Cargo.toml --locked
.\tools\build-hifishifter.ps1 -Target Plugin -Configuration Release -BuildName local-vst3
.\tools\package-vst3.ps1 `
  -DeliveryDirectory "$PWD\.build-tmp\deliveries\local-vst3" `
  -OutputDirectory "$PWD\.build-tmp\vst3-artifacts" -Installer
```

需要 Rust、Node.js、CMake 和 Visual Studio C++ 工具链。SDK commit/tree 锁定在
`tools/plugin-sdks.json`，与 `ara2-bridge-companion 0.3.0` 的身份检查一致；脚本仅安装
ARA_API 和 VST3 pluginterfaces 两个构建必需子模块，不替换已有不同版本的 SDK。
安装器额外需要 NSIS；GitHub Actions 自动安装该工具。
