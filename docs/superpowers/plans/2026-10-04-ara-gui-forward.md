# ARA 正向 GUI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. 用户授权本会话自主完成；减少review，整合后集中审查。

**Goal:** 在REAPER供源的原HiFiShifter GUI里编辑并提交，保存/重开仍可输出编辑结果。
**Architecture:** 现有内核增加PCM注入口；插件命名管道服务权威持有编辑状态；原app下载宿主PCM
到临时WAV用于既有分析/编辑，并显式提交曲线；process只读取原子快照。
**Tech Stack:** Rust、Windows命名管道、Tauri2、React19、已有WORLD/HiFiGAN内核。
**Spec:** `../specs/2026-10-04-ara-gui-forward-design.md`

## Global Constraints

- 仅worktree `E:\code\HiFiShifter\.worktrees\ara-plugin`，`codex/ara-plugin`；不push，不git add -A。
- 不改SDK/registry源，不修四条Windows/tmp失败；中文文件头/关键函数doc。
- Windows x64，44100/48000，mono/stereo，每源<=30秒；倒放不支持，stretch/fades暂不承诺。
- process/setProcessing无锁等待、无分配/释放大缓冲、无IO/reader/推理。
- cargo前加载MSVC，然后重设TEMP/TMP与锁定SDK变量，--offline --jobs 1。
- REAPER只隔离profile/一次性工程，不-nonewinst；不覆盖用户未保存GUI工程。

## Task 16: 内核宿主 PCM 注入口

**Files:** Modify `backend/hifishifter-kernel/src/mixdown.rs`，添加同模块/集成测试。
**Interfaces:** `pub struct MixdownPcm { pub sample_rate:u32, pub channels:u16, pub samples:Arc<Vec<f32>> }`；
`pub fn render_mixdown_with_pcm(timeline:&TimelineState,opts:MixdownOptions,sources:&HashMap<String,MixdownPcm>) -> Result<(u32,u16,f64,Vec<f32>),String>`。

- [ ] RED：不存在路径"ara://source"由注入PCM输出；缺源必须Err，不能静音或文件回退；
  两次同identity不同PCM得到不同输出；现有文件入口仍读实际文件。
- [ ] 实现统一内部函数，文件入口传None，注入入口传Some(sources)。注入数据校验采样率/
  声道/长度/有限值；禁用文件域整clip/formant缓存，DSP不得依赖源路径补读文件。
  核心选择形状：`let decoded = match sources { Some(s) => s.get(source_path).ok_or("missing host PCM")?, None => ... };`。
- [ ] GREEN：上述oracle和真实WORLD音高变化测输出，运行内核mixdown既有测试。报告已测
  命令/数字；只本地逐路径提交。

## Task 17: IPC、插件编辑权威与组件state

**Files:** Create `backend/hifishifter-ara-ipc`、插件`state_channel.rs`/`state_stream.rs`；Modify
workspace/plugin manifests、ara/model、render/document/extension、vst3/lib。
**Interfaces:** 共享`InstanceRecord`、`HostPcm`、`Request::{Snapshot,Commit{base_revision,model_revision,timeline}}`，
`Response{ok,error,revision,model_revision,timeline,sources}`；`discover()->Result<Vec<InstanceRecord>,String>`，
`exchange(&InstanceRecord,&Request)->Result<Response,String>`。

- [ ] RED：长度帧拒绝>64MiB/截断；真实管道snapshot/commit往返；stale revision冲突。
- [ ] 实现token/协议检查、发现心跳、可停止后台服务；客户端读取超时，不连接任意未知endpoint。
  插件服务绑定实际文档。快照按已授权源返回；commit只合并参数不接客户端几何。
- [ ] RED：实际renderer只播放分配clip，修改音量/音高输出变化；state短读写往返，坏state拒绝。
- [ ] GREEN：非实时调用Task16内核再原子发布，getState/setState持久化编辑。修音算法
  WORLD默认可用，允许既有HiFiGAN配置，未支持算法明确错误。只本地提交。

## Task 18: 原 GUI 客户端

**Files:** Create `backend/src-tauri/src/ara_bridge.rs`、`frontend/src/features/ara/AraConnectionPanel.tsx`及
测试；Modify commands.rs/lib.rs、App.tsx。
**Interfaces:** `ara_list_instances`、`ara_connect(instance_id,force)`、`ara_submit`、`ara_refresh`、
`ara_disconnect`。app使用Task17共享discover/exchange协议；返回普通timeline载荷供fetchTimeline。

- [ ] RED：下载PCM生成临时WAV，clip/take路径一起替换；没有PCM/非法输入/脏工程不覆盖。
- [ ] 实现app会话保存实例及两revision，连接/刷新调用snapshot并更新AppState；提交发送当前
  timeline，Conflict不覆盖本地。命令注册遵守现有门面，阻塞IPC使用spawn_blocking。
- [ ] RED/GREEN：GUI面板刷新实例、连接、提交、刷新、断开及失败状态；使用原参数编辑器。
  可用copy为"ARA / REAPER"、"连接"、"提交到REAPER"、"刷新"，显示不支持倒放。
- [ ] build frontend/app，相关测试；只本地逐路径提交。

## Task 19: 宿主全流程验收和可运行包

**Files:** `probe/ara`下启动/部署脚本、GUI验收记录和隔离证据，更新ledger/handoff。

- [ ] 最终插件/IPC/相关内核与app测试，frontend build；集中review并修正影响完整链路问题。
- [ ] 将构建插件和依赖部署到隔离vst目录，启动隔离REAPER与本worktree的HiFiShifter。
- [ ] GUI连接真实宿主源，原参数界面改曲线并提交；REAPER导出与原声逐样本不同，日志
  明确实际内核修音/ready，不接受只改gain来冒充pitch验收。
- [ ] 保存一次性RPP，关闭GUI/REAPER再打开，导出比对；正常/crop/gap无回归，seek/冷启
  供音记录。保持同路径改源再重开不能输出旧缓存。
- [ ] 确认无Tauri/WebView进入插件，写运行入口、限制与证据；本地commit，不push。

## 自查

Task16提供注入函数给17；17共享协议给18；19消费所有构建产物。共享manifest由主agent
编辑避免冲突。倒放唯一范围变更由用户明确授权；不能删除历史失败或宣称完成所有v1能力。
