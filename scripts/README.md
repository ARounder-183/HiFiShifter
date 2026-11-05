# scripts/

构建、发布与门禁脚本。原则很简单：**一个平台一条构建入口，产物只有一处（`dist/`）**。
不建中间交付目录、不写构建清单、不算源码指纹 —— 构建产物的身份由它来自的那个 git 提交决定。

| 子目录     | 用途                                                                                                                                                              |
| ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `env/`     | 构建环境：MSVC 导入（`msvc-env.ps1`）、插件 SDK 路径（`sdk-env.ps1`）、pin 清单（`plugin-sdks.json`）。                                                           |
| `setup/`   | 前置准备：安装系统依赖、按 pin 拉取插件 SDK、取模型权重（`fetch-models.py`，含打包/校验发布资产，见 [docs/models.md](../docs/models.md)）、导出离网用的本地镜像。 |
| `build/`   | 构建：Windows 的 `build.ps1`、Linux AppImage、macOS DMG、模型清单、CoreML 补丁。                                                                                  |
| `release/` | 发布物：VST3 的 NSIS 安装器（`.nsi` + 五语词表）、版本号同步。                                                                                                    |
| `checks/`  | CI 门禁：FFI panic 守卫、日志级别、命令接线、产品一致性、依赖通告扫描。                                                                                           |
| `vst3/`    | 自检与 Unix 构建：`windows/verify.ps1`（导出表 + 依赖 + 黑盒加载）、`unix/build.sh`、`unix/verify.sh`。                                                           |
| `dev/`     | 开发期辅助：`tauri dev` 的前端启动脚本、音高分析内存计量。                                                                                                        |

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
`HiFiShifter_v<版本>_windows-x86_64-vst3.zip`、两个 NSIS 安装器。

macOS / Linux 的 VST3 构建不经过 PowerShell：

```bash
scripts/vst3/unix/build.sh                     # SDK → 前端 → 编译 → bundle → 自检 → dist/ 下的 tar.gz
scripts/vst3/unix/build.sh --skip-frontend     # 只改 Rust 时跳过 npm
scripts/vst3/unix/build.sh --install           # 顺手装进 ~/.vst3
scripts/vst3/unix/verify.sh <HiFiShifter.vst3> # 只跑自检（布局/符号/依赖/dlopen 探针/前端产物）
```

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
