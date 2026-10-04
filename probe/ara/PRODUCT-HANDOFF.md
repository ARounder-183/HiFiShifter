# ARA 产品开发当前交接

中文工作记录，更新于 2026-10-04。此文件记录产品分支，历史探针仍见 HANDOFF.md。

## 工作位置与授权

- 仅 `E:\code\HiFiShifter\.worktrees\ara-plugin`，分支 `codex/ara-plugin`。
- 用户授权按建议持续分批推进，无需中途选择执行方式；不 push，不 git add -A。
- 主工作区 develop、用户 REAPER 工程均不改；SDK 检出不修改。
- 当前 v1 是原本独立 app 图形界面联动插件，不是把完整 Tauri UI 嵌入 REAPER FX 窗口。

## 已完成

Phase 1 已抽内核并保持 app/内核测试合计。Phase 2 Task 10/11 提交 `a9d80b06`：
真实 REAPER UI 移动/切片、实际拉伸、官方 action 41051 倒放并读 ARA PCM；原始证据
已进 git（两个 *.log 用 explicit force-stage）。U1 仍是当前映射的方向表达缺口。

Task 12 提交 `e36889be`：完整 VST3 音频 ABI 原生布局 oracle、安全输出初始化、非法
输入/格式/布局拒绝、process/setProcessing 日志守卫。当前 process 只输出零。
上位设计已更正不存在的 storeAudioSourceContent，head/tail 查询不是预渲染调度。

Task 13 的部分检查点：RegionOwners、真实 region key 注册/撤销、扩展 add/remove
通知、entry builder Arc owner（无 Box::leak）、Processor 工厂初始 COM 引用修复。
完整插件构建及 38 测试通过；采集验证器 6 通过。审查无部分检查点阻塞，**并未批准
Task 13 全部完成或产品发布**。新 DLL 尚未部署到 REAPER。

## 下一步，按顺序

1. 阅读 Phase 3a spec/plan 的 Task 13 “当前部分检查点”。实际 destroy_document 只撤销
   RegionOwners，没有通知 ExtensionControllerLease 或清空 ExtensionOwner.assignments。
   必须找真实 controller -> document -> owner 的关联，不使用“最后一个文档”猜测。
2. 补 bound-extension teardown、observer 重入，以及 editor sequence assignment 通知。
   独立 FFI teardown 测试销毁手动 lease，不能证明产品接线；native entry lifetime
   测试目前是不带文档绑定的 entry。
3. Task 13 完成后才做 Task 14：scope 内宿主 PCM、不可变快照、项目 sample 时间读取。
   不直接按 persistentID 打开文件，不在 process 读 host reader/推理/IO/等待锁。
4. Task 15 在隔离 REAPER 输出波形验证普通/裁切/seek 与真实倒放；未测输出不关闭 U1。
5. 其后 Phase 3b 完整内核注入式修音与供音/cache 风险，Phase 4 原 app 参数通道及持久化，
   Phase 5 打包。不要宣称插件已经能修音或 GUI 联动。

## 环境与验证

每个 PowerShell 独立构建环境：点源 tools/msvc-env.ps1，之后设置 TEMP/TMP 到
.build-tmp/cl；两 SDK 变量指向 probe/ara/rust-path/.third-party 下对应检出。
全套命令见 Phase 3a plan；cargo 均 --offline --jobs 1。
产物为 backend/target/debug/hifishifter_plugin.dll，不是 crate 自己的 target 目录。

最后已停止 task10-clean 隔离 REAPER。重启前检查进程与命令行；不向已有用户实例发
脚本，不使用 -nonewinst。REAPER 会补扫系统 VST3；用已有隔离扫描缓存避免激活弹窗。
38 条插件测试与 6 条验证脚本回归无失败；本批不改 kernel/app，没重跑其完整测试，
四条既有 Windows /tmp 失败不修。既有未用 macro/mut/sequence index 警告保留。
