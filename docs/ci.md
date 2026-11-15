# CI 流水线

构建流水线**只负责构建与打包**，不运行任何测试：单元测试、集成测试、testkit、
安装器闭环（装 → 查 → 重装 → 卸载）与 AppImage 冒烟均已从工作流中移除。依赖漏洞
扫描也不再由 CI 承担，只在本地按需运行。需要跑测试时请在本地直接调用
`scripts/checks/` 下的脚本。

## 工作流

| 工作流                | 触发                  | 职责                                                                  | 产物                                                   |
| --------------------- | --------------------- | --------------------------------------------------------------------- | ------------------------------------------------------ |
| `frontend-build.yml` | 被调用 / 手动         | 构建前端 `dist`（typecheck + vite build），供各产物流水线复用        | `frontend-dist`                                        |
| `build.yml`          | PR、`v*` tag、手动    | 六平台 App 产物；Windows x64 出一个合并安装器 + 便携包；Unix 出插件包 | 安装器 / DMG / AppImage / 便携包 / 插件 `tar.gz`·`pkg` |
| `vst3-unix.yml`      | PR、`v*` tag、手动    | Linux/macOS 的 VST3 bundle                                            | `*.tar.gz` + `*.sha256`                                |
| `clap.yml`           | PR、`v*` tag、手动    | Linux/macOS/Windows 的 CLAP 模块                                      | `HiFiShifter.clap` / bundle                            |

## 构建脚本与自检

- 工作流只调用 `scripts/` 下的构建脚本，YAML 里不重复实现编译逻辑。
- 本地构建默认会跑产物的黑盒自检（dlopen / 导出符号 / ABI oracle / AppImage 冒烟）。
  CI 只构建，因此显式传入跳过自检的开关：
  - `scripts/vst3/unix/build.sh --no-verify`
  - `scripts/clap/unix/build.sh --no-verify`
  - `scripts/clap/windows/build.ps1 -SkipVerify`
  - `scripts/build/build.ps1 -SkipVerify`
  - `scripts/build/unix/package.sh --no-verify`
  - `linux-appimage.sh` 通过 `HIFISHIFTER_SKIP_SMOKE=1` 跳过启动冒烟。
- 想在本地复现 CI 的构建步骤，去掉上述开关即可；想在本地跑 Rust 测试，使用
  `pwsh scripts/checks/run-rust-gates.ps1`。依赖漏洞扫描可手动运行
  `python3 scripts/checks/check-dependency-advisories.py --verbose`。

## 前端产物

前端只构建一次，以 `frontend-dist` artifact 分发给同一工作流内的各平台腿。
生产构建命令是 `npm --prefix frontend run build`（typecheck + vite build），
不包含 `vitest`。

## 触发与矩阵

- PR：`build.yml` 只跑三个平台（windows-x86_64 / macos-arm64 / linux-x86_64）。
- `v*` tag：全量六平台 + 全量 Unix 矩阵。
- tag 与版本号必须一致：`build.yml` 的 `assert-version` 用
  `backend/src-tauri/tauri.conf.json` 的 `version` 比对 tag，不一致直接停。
- 矩阵用 `fromJSON` 直接展开，不使用 `exclude` 里的表达式（GitHub 对后者不会报错，
  只会让整个矩阵一个实例都不创建）。

## 模型权重

权重不在仓库里。每条产物流水线在编译前都会调 `scripts/setup/fetch-models.py` 从
release 资产取（缓存键是 `models.json` 的哈希，模型一换键就变），因此**清单的改动
必须在上传资产之后合并**：资产不存在时脚本退出 1，六平台会一起红。

发布流程（打包、`gh release`、验证、顺序）见 [`models.md`](models.md)。

## 失败之后

- **构建类失败**：先看对应 job 的 diagnostics artifact（`*-diagnostics`），里面有
  构建日志与 bundle 目录清单。`ara2-bridge-companion` 的 build.rs panic 写在日志
  中段、后面跟着几百行 World C++ 告警，按 `^error` 定位才看得见；
  `vst3/unix/build.sh` 的 `sync_submodule` 负责初始化 `pluginterfaces` 与 ARA_API
  两个 gitlink。
- **测试类失败**：CI 不再产生这类红灯；请本地运行
  `pwsh scripts/checks/run-rust-gates.ps1`（Rust）或
  `bash scripts/checks/run-frontend-gates.sh`（前端）。

## 分支保护

原 required checks（`gates / static`、`gates / rust-test`、
`gates / frontend`）已随测试工作流删除而不复存在。启用本改动前，请在 GitHub 仓库
设置中移除这些 required checks，只保留需要的构建 job（例如
`build / Build Backend (windows-x86_64)`），否则 PR 会一直等待永远不会出现的检查。
