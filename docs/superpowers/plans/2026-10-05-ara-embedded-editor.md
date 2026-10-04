# ARA Embedded Editor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. 用户已选本会话自主分批执行，不重复询问；测试集中最后，不逐步骤跑红绿/重复review。

**Goal:** REAPER的FX窗口内使用完整原HiFiShifter编辑工作区并自动应用，保留独立app。

**Architecture:** 原React前端增加插件通信/事件适配；原生IPlugView/WebView2承载UI。
无Tauri共享编辑会话复用内核和现有命令语义；实例后台合并自动渲染，process只读快照。

**Tech Stack:** Rust、VST3/ARA2、Win32、WebView2 COM、React19/TypeScript、既有WORLD内核。

**Spec:** `../specs/2026-10-05-ara-embedded-editor-design.md`

## Global Constraints

- 仅ara-plugin worktree/codex/ara-plugin；不push、不add-A、不改SDK/registry源。
- Windows x64/REAPER7.81、44100/48000Hz、mono/stereo、≤30秒/源，正向普通/裁切；不做倒放。
- 独立app保留；插件禁止tauri/wry/app crate/cpal，允许webview2-com；不跑自有主循环。
- process/setProcessing禁止IO/推理/锁等待；已有512MiB共享PCM/退役快照预算不放宽。
- 中文文件头/关键函数doc。MSVC后重设TEMP/TMP和SDK路径，cargo --offline --jobs 1。
- 用户要求最后集中测试；先写回归用例，必要编译只为排错，未测行为不标完成。
- 不覆盖用户GUI工程/REAPER工程，不-nonewinst；launch拒绝已有REAPER主实例。

## Task 20: 一期证据归档与插件前端通道

**Files:** 修改一期FINDINGS/handoff/run/ledger/progress；新增
`frontend/src/services/pluginHost.ts`、`pluginHost.test.ts`、`hostEvents.ts`；修改
`services/invoke.ts`、`frontendErrorLog.ts`、`main.tsx`及原事件订阅调用点。

**Interfaces:**
```ts
type HostEvent<T> = { event: string; id: number; payload: T };
type PluginBootstrap = { version: 1; viewId: string };
interface PluginHostBridge {
  readonly kind: "plugin";
  invoke<T>(command: string, args?: Record<string, unknown>): Promise<T>;
  listen<T>(event: string, listener: (e: HostEvent<T>) => void): Promise<() => void>;
  dispose(): void;
}
```

- [ ] 写回归：同view命令response解Promise；错view/错version忽略；timeout、native
  error、dispose reject；event解绑；invoke复用原命名映射；无bootstrap不改变Tauri。
- [ ] 实现WebView postMessage adapter，单调id、有界pending、30秒timeout、事件隔离。
  `invoke`优先插件bridge但不伪造Tauri；`listen`插件本地/Tauri懒加载。
- [ ] 原动态event imports迁移hostEvents，window API暂保留显式插件禁用能力待Task23。
- [ ] 归档一期JSON/WAV和进程局部PATH修正，标明同路径改源等边界未完成。

## Task 21: 真IPlugView与原生WebView2宿主

**Files:** 新增`backend/hifishifter-plugin/src/editor/{mod.rs,view.rs,webview.rs}`；修改
`vst3.rs`、`lib.rs`、`Cargo.toml`、`tests/no_tauri_in_dependency_tree.rs`。

**Interfaces:**
```rust
pub(crate) fn create_view() -> *mut std::ffi::c_void;
#[repr(C)] struct ViewRect { left:i32, top:i32, right:i32, bottom:i32 }
// NativeEditor仅UI线程使用，不实现Send/Sync。
impl NativeEditor {
    fn attach(parent:*mut std::ffi::c_void, width:i32, height:i32) -> Result<Self,String>;
    fn resize(&self, width:i32, height:i32) -> Result<(),String>;
    fn close(&mut self);
}
```

- [ ] 写真实vtbl回归：createView("editor")非null；其它name/null拒绝；QI/refcount、
  rectangle布局、unsupported平台、bad/null rect、重复removed、setFrame引用平衡。
- [ ] 按锁定iplugview.h的12个view方法（加3个FUnknown槽）顺序实现，HWND child，不触碰parent，只onSize调整。
  初始1100×720、最小640×400，checked算术拒绝非法size；未处理键返回false。
- [ ] 使用缓存webview2-com0.38/windows0.61，异步COM callbacks持有取消state而非裸view；
  attach立即返回，removed标closed并close controller，不使用wait_with_pump或App runtime。
- [ ] 注册固定origin资产与bootstrap；只接受原生固定origin消息；初始化失败记录到
  自有child窗口。更新A5为禁Tauri/wry/app/cpal、允许原生WebView2。

## Task 22: 原编辑命令抽取及实例路由

**Files:** 新增`backend/hifishifter-kernel/src/editor/{mod.rs,params.rs,history.rs}`；修改
kernel lib与app `commands/{params,core,common}.rs`；新增plugin `editor/session.rs`及
`editor/commands.rs`；修改vst3 IConnectionPoint、render/document和extension。

**Interfaces:**
```rust
pub trait ParamHost {
    fn timeline(&self) -> &std::sync::Mutex<TimelineState>;
    fn checkpoint_timeline(&self, timeline:&TimelineState, operation:HistoryOp);
    fn mark_dirty(&self);
    fn publish_timeline(&self, timeline:TimelineState);
}
// editor::params为原命令函数，State<AppState>替换为&impl ParamHost，其它实参/返回值不变。
// EditorSession持有实例timeline/history、对应owner弱引用与UI通知出口，实现ParamHost。
pub(crate) struct EditorSession;
impl EditorSession {
    pub fn dispatch(&self, command:&str, args:serde_json::Value) -> Result<serde_json::Value,String>;
}
```

- [ ] 为pitch/dyn/tension/静态参数、分块checkpoint=false、平滑和undo/redo写原语义回归；
  同参数输入独立app和plugin得到同projection。未知/几何/设备命令明确Err。
- [ ] 抽取现有函数body为无tauri状态入口，app wrappers只解State并调用；保持命令JSON
  shapes及dirty/undo差异，不重写简化功能。波形/分析/processor描述符复用内核。
- [ ] 通过宿主IConnectionPoint真实processor/controller建立Arc会话，view绑定对应会话；
  测双实例不串、解绑销毁迟到请求失败、processor先销毁仍可安全close。
- [ ] WebMessage callback只排有界后台任务，返回JSON响应经UI消息队列，不在UI线程推理。

## Task 23: 复用完整原App并适配宿主能力

**Files:** 修改`frontend/src/App.tsx`、`features/ara/AraConnectionPanel.tsx`、
`features/dock/{detachBridge,detachedWindow}.ts`、timeline拖放/window监听、
PianoRollPanel、notebook、AboutDialog；新增`services/hostCapabilities.ts`及回归。

**Interfaces:**
```ts
type HostMode = "standalone" | "plugin";
function hostMode(): HostMode;
// 插件session提供get_timeline_state/波形/参数/历史与分析事件，JSON与原API一致。
```

- [ ] 插件bootstrap进入原App/DockRoot，不新增替代钢琴卷帘；隐藏连接/手动提交栏，
  显示自动应用generation/错误。插件不加载Tauri window/webviewWindow/openUrl接口。
- [ ] DAW几何只读、独立文件/设备/录音/分离窗口禁用并说明原因；原app相应入口不变。
- [ ] 原参数编辑器真实画曲线、选择/平滑、undo/redo、waveform/pitch事件路由到Task22；
  测无外部HiFiShifter进程也能工作，不能用mock数据作完成证据。

## Task 24: 自动渲染、最后一笔与工程state

**Files:** 新增plugin `editor/auto_apply.rs`；修改render/document/extension、state_channel
与state_stream；前端自动应用状态订阅。

**Interfaces:**
```rust
impl EditorSession {
    pub fn schedule_apply(&self, generation:u64);
    pub fn flush_edits(&self) -> Result<(),String>;
}
// 自动worker: latest generation + model_revision校验后才发布快照。
```

- [ ] 写rapid A→B→C只发布C、慢A不覆盖B、失败不假同步、host变更拒绝、单纯授权开关
  不冲突、undo/redo触发、false-checkpoint最终尾块、关闭UI/保存前最后一笔等回归。
- [ ] 150ms合并，实例worker非实时WORLD调用；参数权威立即更新供getState，音频晚发布。
  失败保持旧快照并显式错误；销毁取消任务并安全join，避免DLL卸载后执行代码。
- [ ] v2真实身份恢复、重开自动重新渲染；保留旧非空v1拒绝，不以丢编辑消除冲突。

## Task 25: 规范bundle及一期边界收尾

**Files:** 新增`probe/ara/build_embedded_editor.ps1`、`start_embedded_editor.ps1`、
`build_embedded_editor_probe.lua`、`EMBEDDED-EDITOR-RUN.md`；更新依赖加载代码/ledger。

- [ ] 构建frontend后将资源复制到`HiFiShifter.vst3/Contents/Resources/frontend`，
  plugin到`Contents/x86_64-win/HiFiShifter.vst3`，部署邻接DirectML/SoundTouch/ORT。
- [ ] 模块自身定位资源/依赖，固定虚拟origin、防导航和隔离profile；不依赖cwd或系统PATH。
- [ ] 隔离profile与一次性RPP脚本覆盖同路径改源重开、seek/冷启及44100/48000
  mono/stereo、30秒边界；保留一期失败证据，不修四条既有Windows/tmp失败。

## Task 26: 最后集中验收

- [ ] 一次集中运行frontend、共享editor、app相关、plugin/IPC和既有输出oracle测试；
  frontend/app/plugin build；依赖图无Tauri/wry/app/cpal，native ABI对锁定SDK核对。
- [ ] REAPER内原GUI真实pitch自动应用并导出：频率变化、gap/crop正确；关闭FX不影响
  已渲染供音；FX/RPP重开恢复；双实例、快速最后一笔、undo/redo、保存中渲染不丢编辑。
- [ ] 独立app真实导入和编辑仍能使用；没有启动它的情况下插件可完整编辑。
- [ ] 写FINDINGS、全部命令/退出码/音频hash/PCM/screenshot/日志，逐条审计本spec；
  未达项不能勾完成；明确路径stage/local commit，绝不push。只有全部要求实测后关闭目标。

## 进度

Task20一期归档/插件通信/事件适配已写，新增回归未执行；Task21真实IPlugView和异步
WebView源码已写，尚未宿主验收；Task22原params逻辑及既有互转用例已机械迁入kernel，
AppState实现ParamHost，独立app仍走原设备/dirty/undo副作用，cargo check app exit0。
插件native cargo check --tests exit0（含CSP入口/窗口token/重入修正及参数迁移），
前端通信迁移tsc -b exit0；最新模块pin增量尚未重新编译，新增回归仍未执行。全功能测试按用户要求
留最后。Task22实例路由/完整命令会话、Task23-26仍未完成。用户已选择本会话自主分批，
无需重新选择执行方式。最终验收前不以某一小批绿测重定义整个二期为完成。

本批Task22源实现：正式IConnectionPoint + 宿主IMessage/UTF16属性握手，PID/租约令牌
定位对应processor，拒绝未知/过期/跨进程路线；class_flags不再宣称distributable。
原waveform/mipmap、参数描述符、history checkpoint、授权PCM分析副本已共享，app仍
用原入口薄适配。原生UI消息有界排给实例actor；响应与有界事件分邮箱，主线程timer
回传COM，不在窗口回调推理。私有track/clip ID前缀分流内核异步事件。

Task24首个源实现：150ms合并、最新submitted ticket阻止旧作业发布、参数接受立即写入
组件权威；getState先flush已收到尾块，关闭FX不取消已排曲线写入；后台失败保留旧音频
并发状态事件。实际宿主播放时钟由process原子发布，UI不运行cpal设备。
编译检查：插件cargo check --tests与独立app cargo check均正常exit0；新增actor/路由
回归只编译未运行。当前尚未构建部署和真实REAPER验收，不能勾完成Task22/24。

下一批优先：Task23前端宿主能力/隐藏手动提交栏/自动应用状态，Task25规范bundle。
随后补齐Task24压力/冲突/双实例/持久化回归并集中Task26。已识别待收敛：retired快照
仍保留到owner释放，自动编辑下需确认/改进安全回收；actor渲染持doc transaction期间
宿主UI模型回调可能等待；PCM私有分析文件的重复几何刷新与回收需资源边界审计。
现阶段不宣称所有原GUI功能已接线，未支持的命令明确Err而不是伪报成功。

本批Task23/25：原GUI插件模式/自动状态栏、Tauri窗口前置拒绝、文件/录音/轨道几何
入口保护、timeline kernel geometryReadOnly桥接已写；UiSettings补丁合并保持其它配置。
frontend tsc/build通过，Rust引擎build通过，Win32薄入口utf8/engine_exit修正后编译通过。
隔离REAPER46132真实加载规范bundle（项目CWD，无PATH添加），IConnectionPoint消息
三组result0，FX对话框accessibility里出现原GUI/宿主轨道/WORLD/自动应用0/0，无外部app。
baseline已导出；新FINDINGS与EMBEDDED-EDITOR-RUN记录事实，不能勾最终修音/恢复验收。

Task24增加非实时退役快照回收，实时读区SeqCst计数保护指针生命周期，分配前回收，
失败保留旧音频。最新cargo check --tests exit0，仅编译新增1000次/活跃读者回归，
尚未运行/部署。当前REAPER加载的是回收改动前引擎，不可热替换。原始日志还暴露
参数系统剪贴板/转写未接线，时间轴剪贴板轮询源码已按只读模式关闭（未部署）。
评估/About窗口和Sky截图/主窗口bounds问题需当前状态重选，禁止保存前景Codex截图
当GUI证据。下一批处理剩余原GUI参数动作/clipboard及最终自动pitch导出/重开。
