# scripts/

构建、发布与门禁脚本。原则很简单：**一个平台一条构建入口，产物只有一处（`dist/`）**。
不建中间交付目录、不写构建清单、不算源码指纹 —— 构建产物的身份由它来自的那个 git 提交决定。

| 子目录     | 用途                                                                                                                                                                        |
| ---------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `env/`     | 构建环境：MSVC 导入（`msvc-env.ps1`）、插件 SDK 路径（`sdk-env.ps1`）、pin 清单（`plugin-sdks.json`）。                                                                     |
| `setup/`   | 前置准备：安装系统依赖、按 pin 拉取插件 SDK、取模型权重与生成清单（`fetch-models.py`，含打包/校验发布资产，见 [docs/models.md](../docs/models.md)）、导出离网用的本地镜像。 |
| `build/`   | 构建：Windows 的 `build.ps1`、Linux AppImage、macOS DMG、Unix 插件分发（`unix/package.sh`）、CoreML 补丁。                                                                  |
| `release/` | 发布物：VST3 的 NSIS 安装器（`.nsi` + 五语词表）、版本号同步（`version.py` + 落盘清单 `version-sources.json`）。                                                            |
| `checks/`  | CI 门禁：FFI panic 守卫、日志级别、命令接线、产品一致性、依赖通告扫描；`-Package hifishifter-plugin-core` 只跑格式无关层。                                                  |
| `clap/`    | CLAP 自检与构建：`probe.c`（黑盒 dlopen + 短流宿主）、`verify.sh`、`unix/build.sh`、`windows/build.ps1`；ABI 布局 oracle 在 `backend/hifishifter-clap/tests/clap_layout.{c,rs}`。                                                                              |
| `plugin/`  | 插件打包的共享规则：`unix/lib.sh`（运行时依赖 / rpath / 前端 / 许可证 / 签名）、`unix/install.sh`（随 Linux 插件包分发）。                                                  |
| `vst3/`    | 自检与 Unix 构建：`windows/verify.ps1`（导出表 + 依赖 + 黑盒加载）、`unix/build.sh`、`unix/verify.sh`。                                                                     |
| `dev/`     | 开发期辅助：`tauri dev` 的前端启动脚本、音高分析内存计量。                                                                                                                  |

## 常用入口

Windows 只有一条构建命令，它按顺序做完：前端 → App → VST3 bundle → 黑盒加载门禁 → zip / 安装器。

```powershell
.\build.bat                                # 双击入口：App + VST3
.\scripts\build\build.ps1 -Target Plugin   # 只出 VST3
.\scripts\build\build.ps1 -Target App      # 只出 App
.\scripts\build\build.ps1 -SkipFrontend    # frontend\dist 已是最新

.\scripts\setup\prepare-plugin-sdks.ps1    # 首次构建 VST3 前执行一次
.\scripts\checks\check-product-consistency.ps1
```

产物全在 `dist/`：`HiFiShifter.vst3`、`HiFiShifter-v<版本>-portable-win-<arch>.zip`、
`HiFiShifter_v<版本>_windows-<arch>-setup.exe`（App、VST3 与模型只装一份的合并安装器）。

macOS / Linux 的插件构建不经过 PowerShell。**构建按格式分开，分发统一**：

```bash
# 分发入口：一次编译 → 每个 OS 一个可安装的产物（Linux tar.gz / macOS pkg）
scripts/build/unix/package.sh                  # VST3 + CLAP
scripts/build/unix/package.sh --formats=vst3   # 只出 VST3

# 单格式构建（改一个格式时不必等另一个）
scripts/vst3/unix/build.sh                     # SDK → 前端 → 编译 → bundle → 自检 → dist/ 下的 tar.gz
scripts/vst3/unix/build.sh --skip-frontend     # 只改 Rust 时跳过 npm
scripts/vst3/unix/build.sh --install           # 顺手装进 ~/.vst3
scripts/clap/unix/build.sh                     # 编译 → 组装（依赖 + 资源）→ 自检 → dist/ 下的 tar.gz

# 只跑自检
scripts/vst3/unix/verify.sh <HiFiShifter.vst3>
scripts/clap/verify.sh <HiFiShifter.clap>
```

平台规则（运行时依赖怎么放、rpath 怎么断言、前端与许可证怎么进包、怎么签名）只有一份
实现：`scripts/plugin/unix/lib.sh`。格式脚本只描述自己的布局 —— VST3 在 Linux 上也是
`Contents/<arch>-linux/` 的 bundle，CLAP 在 Linux 上是单文件、在 macOS 上才是 bundle。
这条分界是踩出来的：CLAP 的 `build.rs` 早就声明了 `$ORIGIN`，而打包脚本没把库放进去，
`ldd` 与 `dyld` 同时红。

bundle 里带着前端（`Contents/Resources/frontend`），所以这条流水线需要
**Node ^20.19.0 或 >=22.12.0**（vite 8 / rolldown 的 engines）。版本不对时构建脚本在入口就报出来 ——
那两个真实的报错（`styleText` 缺失、`rolldown/binding-*` 找不到）看起来都像前端代码的问题。

两个 GUI 探针是手工工具而不是门禁（需要显示服务，CI 没有）：

```bash
gcc -o /tmp/gui_probe scripts/vst3/unix/gui_probe.c $(pkg-config --cflags --libs gtk+-3.0 webkit2gtk-4.1 x11)
GDK_BACKEND=x11 /tmp/gui_probe
gcc -o /tmp/host_harness scripts/vst3/unix/host_harness.c -ldl -lX11
HIFISHIFTER_PLUGIN_LOG=/tmp/hf.log /tmp/host_harness <bundle>/Contents/x86_64-linux/HiFiShifter.so
```

第三方源码的获取与离网构建见根 `README.md` 的「第三方源码」一节。
