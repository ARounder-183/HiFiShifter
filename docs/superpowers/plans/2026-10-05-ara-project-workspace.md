# ARA Project Workspace Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans。
> 用户已要求本会话自主分批、不要逐步询问；回归用例先写，测试集中批末，减少 review。

**Goal:** 同一原 HiFiShifter FX 工作区编辑同工程全部已接入轨道，自动回送各自轨道，保留独立 app。

**Architecture:** 文档拥有唯一原编辑 actor；真实组件路由授权进入它。工作区投影取活 renderer
区域并集，DSP 和组件 state 仍按各自分配隔离；不在前端拼接多套客户端。

**Tech Stack:** Rust / ARA2 / VST3 / WebView2 COM / 原 React GUI 与共享 kernel。

**Spec:** `../specs/2026-10-05-ara-project-workspace-design.md`

## Global Constraints

- 仅 `E:/code/HiFiShifter/.worktrees/ara-plugin`、`codex/ara-plugin`，不 push、不 add -A、不动 SDK/registry。
- Windows x64/REAPER7.81、44100/48000Hz、mono/stereo、≤30秒/源；不新增倒放/stretch/content fades。
- 原内嵌 spec 的有界队列/512MiB共享预算/实时零IO推理等待/独立 app 兼容约束全部保留。
- 不改宿主顶层/父窗口，不强杀、不热覆盖；Computer Use 检测到用户输入时先停输入并重新观察。
- 中文源头/关键 doc；MSVC后重设TEMP/TMP，cargo offline/jobs1；只在测试副本采集。
- 先完成旧 plan Task31 的主线程重复合成调查及修正门，不把 release 构建成功当性能通过。
- 原 Task26 未达门仍 open；Task30 的双轨 GUI 验收移到本计划 Task35，不花额外轮次验旧窗口交互。

## Task32：文档工作区授权投影

**Files:** 新增 `backend/hifishifter-plugin/src/editor/workspace.rs`；修改 `editor/mod.rs`、
`render/{document,extension}.rs`。测试放 workspace 模块 `#[cfg(test)]`，复用真实 owner/model fixture。

**Interfaces（本任务定义，后续复用）:**
```rust
pub(crate) struct WorkspaceScope { pub regions: std::collections::BTreeSet<u64> }
impl DocumentSession {
    pub(crate) fn workspace_scope(&self) -> Result<WorkspaceScope, String>;
    pub(crate) fn workspace_timeline(&self) -> Result<TimelineState, String>;
}
impl ExtensionOwner {
    pub(crate) fn editor_document(&self) -> Result<Arc<DocumentSession>, String>;
}
```

- [ ] **Step1:** 用 extension 现有同源双轨 fixture 编写真实 scope 回归：两 owner 分配的 keys
  并集包含四段且保留两轨；同 region 多次分配仅出现一次；空 scope 为零轨而非全文档。
  增加独立 model B 的相同 sourceID，A 工作区不能出现 B 的 clip。
  核心断言：`assert_eq!(document.workspace_timeline().unwrap().tracks.len(), 2);`。
- [ ] **Step2:** 从 `renderer_owners()` 收集活 owner、调用已存在 `assigned_regions()`，用
  `region_owners().resolve()` 验证真实 doc id；用 `clip_ids` 过滤模型时间线，保留宿主排序。
  消失/跨文档 key 返回 Err；不从名字或源ID dedup轨道，不改变 renderer 的 assigned_timeline。
- [ ] **Step3:** 覆盖 assignment 增删、同一源复制的顺序变更，确定 scope 变化也推进工作区
  可观察版本，而不是只等 document model_revision。所有 snapshot/scope读在非实时事务边界。
- [ ] **Step4:** 批末运行 scoped 回归，期望两文档/同源/空投影全部通过；明确 stage 三个源模块。

## Task33：共享原编辑 actor、history 与视图事件

**Files:** 修改 `render/document.rs`、`render/extension.rs`、`editor/{session,commands,events,routing}.rs`。

**Interfaces:**
```rust
impl DocumentSession {
    pub(crate) fn editor_session(self: &Arc<Self>) -> Result<Arc<EditorSession>, String>;
    pub(crate) fn workspace_projection(&self) -> Result<String, String>;
    pub(crate) fn accept_workspace_edits(&self, base_edit:u64, base_model:u64,
        client:&TimelineState, previous:&str) -> Result<(u64,u64,String),String>;
}
impl EditorSession {
    pub(crate) fn new(document:&Arc<DocumentSession>) -> Result<Arc<Self>,String>;
}
```

- [ ] **Step1:** 回归两个真实组件 `editor_session()` 满足 `Arc::ptr_eq(&a,&b)`，不同文档不等。
  两个 UiSink 经现有 enqueue 路由操作两个不同 trackId；必须从同 actor 获取完整两轨 payload。
  路由租约撤销回归保留，不能绕过 EditorLink.owner() 来访问全文档。
- [ ] **Step2:** DocumentSession 用 OnceLock 持 actor；session.owner 改 weak document。
  原 namespace/参数/分析/波形命令不重写，ensure_loaded 取工作区 snapshot，transport用文档时钟。
  `ExtensionOwner::editor_session` 只解真实文档并转交；外部一期 IPC Snapshot/Commit仍按实例范围。
- [ ] **Step3:** 提取 workspace_projection/accept_workspace_edits，验证完整当前scope/host几何，
  单事务合并参数；禁止请求通过伪造 scope 纳入未授权轨道。合成依旧调用每 owner 的区域隔离准备。
- [ ] **Step4:** 为 A60/B67/undo/redo 写现有 command 原语义回归：撤销B恢复64，A60不变；
  新view初始化不重置history/选轨。history/payload变化广播其它view并触发刷新；未知/过期view拒绝。
  不为多轨工作区另造简化 UI；原多轨选择行为仍来自 DockRoot。
- [ ] **Step5:** 批末运行 plugin lib 与 frontend 相关宿主/选择/历史回归、tsc。
  若尚未 native 验收，只记录源码回归，不能勾 Task35。

## Task34：组件保存、入口销毁及旧 state 兼容

**Files:** 修改 `render/{document,extension}.rs`、`editor/session.rs`、`vst3.rs`；
必要前端通知修改 `services/pluginHost.ts`，不改独立app设备入口。

**Interfaces:** 文档级 editor_session/flush 使用 Task33 定义；现有 encode_state/restore_state/v2保持形状。

- [ ] **Step1:** 写 getState 从任一 owner flush 同一共享actor，随后过滤该 owner 范围的回归；
  两组件state分别仅含对应params。用旧归档的v2 byte fixture恢复，限定同源身份仍不串。
- [ ] **Step2:** 将组件 `stop_editor` 与文档 shutdown 分开：component terminate撤销自身route/
  输出/view许可，不关闭共享actor。`DocumentSession::close` 先撤销任务与view，再安全joinworker，
  然后销毁模型存储；weak document消除Arc循环。没有renderer时不无限排重试。
- [ ] **Step3:** 覆盖入口view关闭、入口processor释放而其它owner存活、旧view继续请求被拒绝、
  文档关闭/迟到请求、两文档关闭一方不影响另一方；用真实lease和Weak计数验证，不只查字符串。
- [ ] **Step4:** 回归保存期间最新尾块、共享undo后保存、重开尚未开GUI时恢复PCM；
  期望每轨独立频率且对应PCM不变，模型冲突拒绝/保存曲线不伪报成功。
- [ ] **Step5:** 单次集中审查共享寿命/锁顺序/线程边界，修重要项后只定向重测。

## Task35：一个原 GUI 双轨实用验收及原二期余项

**Files:** 更新 `probe/ara/{EMBEDDED-EDITOR-RUN.md,EXECUTION-LEDGER.md}`、
`captures/embedded-editor-FINDINGS.md`；明确命名归档 `gui-workspace-*` WAV/JSON/RPP/截图。

- [ ] **Step1:** 正常退出宿主后构建新规范bundle，测试副本只打开一个FX；Computer Use先点击
  该窗口，再观察完整原GUI两轨四段。没有 HiFiShifter.exe，没有手工Submit。
- [ ] **Step2:** 同一窗口原选择轨道与音高动作设A60/B67，逐轨宿主Solo导出。
  复用 `verify_forward_gui_output.ps1 -FirstClipStartSec 1` 的严格布局/差异验证，追加各轨目标音高
  与裁切段频率，不以任何不同的WAV/hash当不串轨证据。
- [ ] **Step3:** GUI撤销B→64、重做B→67，两次导出且A保持60；可开第二FX检查同步视图，
  随后关全部FX、导出、保存正常退出、冷重开未开GUI时各轨PCM相同。
- [ ] **Step4:** 实测宿主seek/改源/几何及pending冲突，44100/48000 mono/stereo与30秒边界；
  保留原scope之外unsupported，不把BPM数字同步当完整TempoMap/隐式stretch通过。
- [ ] **Step5:** 独立app真实导入/原GUI编辑验收，最终一次完整源码门；汇总当前计划与原Task26
  每项证据。只有用户所需原GUI完整链路及全部明确门都有实测后关闭goal；缺项仍open。

## 当前状态

本文已核对现有 Doc/Owner/Editor 路由与范围实现，是下一批源改动计划，Task32-35尚未实现。
Task30仅第一轨60的独立导出已通过，第二轨仍64；不是统一工作区或双轨完整恢复证据。
