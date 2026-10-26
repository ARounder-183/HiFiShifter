# scripts/

本目录收纳项目的全部脚本，按用途分子目录。此前脚本分散在 `scripts/` 与 `tools/` 两处、
边界模糊（同样是构建脚本，却一半在这边一半在那边），现统一到这里，一个目录回答
"脚本在哪"。

| 子目录 | 用途 |
| --- | --- |
| `env/` | 构建环境与 SDK 钉死的唯一事实来源：MSVC 环境导入、SDK 路径解析（`sdk-env.ps1`）、pin 清单（`plugin-sdks.json`）。 |
| `setup/` | 前置准备：安装系统依赖、按 pin 拉取并校验插件 SDK、导出无网络构建用的本地镜像。 |
| `build/` | 构建与打包：App/插件统一构建、便携包、Linux/macOS 产物、模型清单、CoreML 补丁生成。 |
| `release/` | 发布物组装：VST3 打包与 NSIS 安装器（`.nsi` + 五语词表），版本号同步。 |
| `checks/` | CI 门禁：FFI panic 守卫、日志级别守卫（`check-log-levels.py`）、命令接线守卫（`check-command-gating.py`，经产品一致性脚本调用）、产品一致性守卫。 |
| `vst3/unix/` | **macOS / Linux** 的 VST3 构建：`build.sh`（SDK→前端→编译→bundle→自检→打包）与 `verify.sh` + `probe.c`（dlopen 黑盒加载 + 前端产物完整性验证）。Windows 的插件构建仍在 `build/`、`release/`。 |
| `dev/` | 开发期辅助：`tauri dev` 的前端启动脚本、音高分析内存计量。 |

前端专用的浏览器验证脚本（`dev-shot.mjs` 等）留在 `frontend/scripts/`，因为它们依赖
前端依赖树（playwright-core）且只服务于前端调试。

## 常用入口

```powershell
.\pack-portable.bat                                  # 双击入口：App / VST3 / 两者
.\scripts\build\pack-portable.ps1 -SkipBuild         # 只打包已有产物
.\scripts\setup\prepare-plugin-sdks.ps1              # 首次构建 VST3 插件前执行一次
.\scripts\checks\check-product-consistency.ps1       # 跨文件一致性守卫（CI 同款）
```

macOS / Linux 的 VST3 构建不经过 PowerShell（CI 的 Unix runner 上不该为此装 pwsh）：

```bash
scripts/vst3/unix/build.sh                     # 一条命令：SDK → 前端 → 编译 → bundle → 自检 → tar.gz
scripts/vst3/unix/build.sh --skip-frontend     # 只改 Rust 时跳过 npm（要求 frontend/dist 已存在）
scripts/vst3/unix/verify.sh <HiFiShifter.vst3> # 只跑自检（布局/符号/依赖/dlopen 探针/前端产物）

# 两个 GUI 探针（需要显示服务；CI 没有，所以它们是手工工具而不是门禁）
gcc -o /tmp/gui_probe scripts/vst3/unix/gui_probe.c $(pkg-config --cflags --libs gtk+-3.0 webkit2gtk-4.1 x11)
GDK_BACKEND=x11 /tmp/gui_probe                                        # GTK/WebKit/嵌入能不能用（无参数：加载内置测试页）
gcc -o /tmp/host_harness scripts/vst3/unix/host_harness.c -ldl -lX11
HIFISHIFTER_PLUGIN_LOG=/tmp/hf.log /tmp/host_harness <bundle>/Contents/x86_64-linux/HiFiShifter.so
                                                                     # 合成宿主：走一遍 createView → attached
```

Unix 的 bundle 里带着前端（`Contents/Resources/frontend`），所以那条流水线需要
**Node ^20.19.0 或 >=22.12.0**（vite 8 / rolldown 的 engines）。版本不对时构建脚本会在
入口就报出来 —— 那两个真实的报错（`styleText` 缺失、`@rolldown/binding-*` 找不到）
看起来都像是前端代码的问题。

各脚本的完整参数见其文件头注释；第三方源码的获取与无网络构建见根 `README.md` 的
「第三方源码」一节。
