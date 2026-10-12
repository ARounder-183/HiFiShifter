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
| `checks/` | CI 门禁：FFI panic 守卫、产品一致性守卫。 |
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

各脚本的完整参数见其文件头注释；第三方源码的获取与无网络构建见根 `README.md` 的
「第三方源码」一节。
