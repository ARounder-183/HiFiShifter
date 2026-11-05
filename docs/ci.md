# CI 流水线

五条工作流，各管一件事。判断"我的改动该看哪条"只需问两个问题：**它是不是源码级约定**、
**它要不要产物**。

| 工作流 | 触发 | 职责 | 产物 |
| --- | --- | --- | --- |
| `gates.yml` | 被调用 / 手动 | 源码级门禁：静态约定、前端、fmt、clippy、单测、testkit | `frontend-dist` |
| `plugin-gates.yml` | push `develop`、PR | 直接 push 时的插件快速反馈（static + 插件并发单测） | 无 |
| `build.yml` | PR、`v*` tag、手动 | 六平台 App 产物；Windows x64 出一个合并安装器 + 便携包 | 安装器 / DMG / AppImage / 便携包 |
| `vst3-unix.yml` | PR、`v*` tag、手动 | Linux/macOS 的 VST3 bundle（dlopen 黑盒验证） | `*.tar.gz` + `*.sha256` |
| `security.yml` | 每周 + 依赖变更 | 依赖漏洞扫描 | 无 |

## 为什么门禁只有一处

同一组 cargo 命令此前在三条工作流里各写一遍，参数并不相同（`--offline`、`--jobs`、
`--skip`、`--test-threads`）。结果是同一条用例可以在一条流水线上绿、在另一条上红，
而"哪条才算数"没人说得清 —— 更糟的是，某条流水线会在一棵 build.yml 会拒绝的树上产出
绿灯产物。

现在：

- **命令的唯一来源**是 `scripts/checks/run-{static,rust,frontend}-gates.{ps1,sh}`；
  工作流只调用它们。
- **测试跳过的唯一来源**是 `scripts/checks/test-skips.txt`。工作流的 YAML 里不再出现
  任何 `--skip`。清单条目按 `[unix]` / `[windows]` 前缀分平台生效。
- `run-rust-gates.ps1 -PrintCommand` 会打印它解析出的完整命令行，本地可以拿来与 CI 日志
  逐字对照（"为什么这条在 CI 上与本地不一样"从此是一句话能回答的问题）。

## 门禁档位

- `-Profile pr`（默认）：fmt + clippy + 单测。
- `-Profile full`：再加两件事 —— `hifishifter-kernel` 的 `--no-default-features`
  clippy（纯 Rust 形态从未被编译过），以及一次 `--test-threads=4` 的并发复跑。
  串行跑得通不代表正确：本仓最容易出事的地方（重算与自动应用的版本竞态、快照发布与
  读者回收）全部是并发敏感的。
- `-Only fmt|clippy|test`：拆成独立 job 时用。
- `-Package <crate>`：只跑一个 crate（插件并发门禁用）；`hifishifter-plugin-core` 是格式无关层，新增格式 crate 后一并加入依赖树/版本守卫。

## 产物门禁留在产物流水线

安装器闭环（装 → 查 → 重装 → 卸载 → 查）、许可页里多语言文本的编码、AppImage 冒烟、
DMG 里的 dylib 校验都必须**先有产物**。它们属于 `build.yml` / `vst3-unix.yml`：搬进
`gates.yml` 只会让门禁要么什么都验不了、要么把六平台全编一遍。

## 触发与矩阵

- PR：`build.yml` 只跑三个平台（windows-x86_64 / macos-arm64 / linux-x86_64）——
  够证明"改动没把某个平台编坏"；`vst3-unix.yml` 只跑 linux-x86_64 / macos-arm64。
- `v*` tag：全量六平台 + 全量 Unix 矩阵。
- tag 与版本号必须一致：`build.yml` 的 `assert-version` 用
  `backend/src-tauri/tauri.conf.json` 的 `version` 比对 tag，不一致直接停。
  版本号的五个副本由 `check-product-consistency.ps1` 保证一致。

## 失败之后

- **门禁类失败**（gates / plugin-gates）：本地跑同一个脚本即可复现，例如
  `pwsh scripts/checks/run-rust-gates.ps1 -Only clippy`。
- **产物类失败**（build / vst3-unix）：先看对应 job 的 diagnostics artifact
  （`*-diagnostics`），里面有构建日志与 bundle 目录清单。
- 偶发失败**不要**直接重跑掩盖：先确认它是不是 `test-skips.txt` 里已登记的那一条。

## 已知的测试跳过

`scripts/checks/test-skips.txt` 是权威清单。当前三类（全部是 `[unix]`，都只在
Linux/macOS 暴露、且不是这轮适配引入的）：

- `task34_tail_and_shared_undo_save_restore_both_world_tracks_without_gui`：
  `clear_renderers()` 之后请求仍被受理。
- `effective_item_mute_projects_to_gui_and_gates_real_process_without_inference`：
  静音→有声的去咔没有落成淡入（缓冲原样留在静音值）。
- `automatic_apply_releases_the_transaction_and_rechecks_model_and_edit_versions`：
  偶发（单独连跑 3 次：1 过 2 挂）—— 快照已发布、块却还没被后台渲染填上。Windows 侧
  根因已修，Unix 夹具尚未按修复后的形态复验。

外加一条全平台的 `pending_pitch_analysis_does_not_publish_raw_audio_as_applied`
（间歇挂死，根因按 `docs/plans/2026-10-27-audit-repair-round2.md` §1.3 / §7 批次 8 跟踪）。

**删条目的时机**：修好之后连同原因一起删。清单里的每一条都应当指向一个正在跟进的
解释，否则它就是"我们不打算修"的另一种说法。

## 模型权重从哪来

权重不在仓库里。每条产物流水线在编译前都会调 `scripts/setup/fetch-models.py` 从 release
资产取（缓存键是 `models.json` 的哈希，模型一换键就变），因此**清单的改动必须在上传资产
之后合并**：资产不存在时脚本退出 1，六平台会一起红而不是某条 lint 红。发布流程（打包、
`gh release`、验证、顺序）见 [`models.md`](models.md)。
## 分支保护

required checks（仓库设置里配，不在本文件里）：

- `gates / static`
- `gates / rust-test`
- `gates / frontend`
- `build / windows-x86_64`

其余 job 与 `vst3-unix.yml` 视情况保留为"必须绿但不阻塞合并"（它们的运行时间更长，
且历史上从没有单独红过）。
