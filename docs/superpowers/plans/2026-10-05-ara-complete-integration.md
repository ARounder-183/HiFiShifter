# ARA Complete Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans。
> 用户已指定本会话自主执行、不要逐步询问，测试按批末/最终集中；后续只在最终一次review。

**Goal:** 保留独立App，交付REAPER内同工程多轨原GUI、逐轨输出、完整clip变换与HiFiGAN缓存。

**Architecture:** 真实文档级原actor与逐renderer音频职责分离；ARA音频权威加经验证的宿主
元数据适配；复用原变换/模型管线，分层有界缓存与异步最新代次发布。

**Tech Stack:** Rust / ARA2 / VST3 / REAPER原生API（只有核对后使用）/ WebView2 / 原React与kernel。

**Spec:** `../specs/2026-10-05-ara-complete-integration-design.md`

## Global Constraints

- 仅ara-plugin worktree/codex/ara-plugin；不push、不add-A、不改SDK/registry、不动主develop。
- 宿主资产不覆盖、不-nonewinst、不强杀/热替换；测试只在独立profile与一次性副本。
- 不新增简化GUI/独立编辑App替代原工作区，不脚本提交曲线冒充GUI，不从路径/RPP冒充ARA源。
- process/setProcessing不做IO/推理/锁等待；原队列/512MiB先导保护保持到有证据的资源方案。
- 拉伸/渐变旧unsupported不再是最终完成边界；此前暂不做倒放仍保留。30秒不是最终素材上限。
- MSVC后重设TEMP/TMP和SDK，cargo offline/jobs1；中文头注释与关键doc。
- 原Task20-26余门仍在；新工程工作区Task32-35继续，但最终验收以本spec扩展后的门为准。
- 用户最后已明确四项目标，不重复询问执行方式/逐节审批；中间只报事实与风险。

## 执行顺序与当前前置

Task31后台准备源码在ffa8266a：首轮lib70/收尾26，review修正后定向11通过；最终native门
未过。当前先Task36 renderer/时钟→Task32-34共享工作区→Task38宿主完整变换→Task39缓存
→Task40长源/资源→Task41集中验收。旧Task37仅是调查，不足以完成新增“完整支持”。

## Task36：renderer透传与实际播放时钟

**Files:** `src/audio_abi.rs`、`vst3.rs`、`render/{extension,transport}.rs`、`editor/{session,commands}.rs`；
probe启动/诊断说明。路径均在backend/hifishifter-plugin（probe文件除外）。

**Interfaces:**
```rust
pub(crate) unsafe fn validate(data:*mut ProcessData)->Result<(),BufferError>;
pub(crate) unsafe fn pass_through(data:*mut ProcessData)->Result<(),BufferError>;
impl ExtensionOwner { pub(crate) fn renders_playback(&self)->bool; }
```

- [ ] 原生vtable回归：纯editor输入fade/gain形状在实时/停播/离线完全保留，即使有旧快照；
  对应in-place、同bus、inactive/null plane、silence flags、尾哨兵、非法输入无写入、零分配。
  核心断言用字面波形：`assert_eq!(out,[0.0,0.1,0.2,0.3]);`，不由产品算法算expected。
- [ ] 分离validate与clear，透传不先清零in-place输入；playback角色仍快照替换，停播静音/
  offline供音门保持。后台只为真实playback owner准备输出，GUI读取实际播放准备状态。
- [ ] 有界原子时钟来源/mode/回跳计数，在非实时actor诊断日志读出；根据真实REAPER连续
  播放证据收敛权威。覆盖prefetch、其它renderer停处理、seek/loop；不把最大值当正常时钟。
- [ ] 批末只运行相关plugin回归；新目录规范bundle冷开，普通fade最终PCM/游标实际验证。

## Task32-35：同工程工作区与逐组件恢复

按 `2026-10-05-ara-project-workspace.md` 的具体scope/actor/state步骤继续；该文末Task36-37
由本计划替代。增加跨轨solo/mute在原GUI作用于正确独立输出的门，不能因每renderer只保留
自己track而忽略其它轨的solo。共享actor不绕过原processor路由，宿主多个view是同步视图。
Task35不再作为整goal的最终门，执行完后仍继续Task38-41。

## Task38：完整宿主clip数据、变换与参数重投影

**Files:** 新增 `backend/hifishifter-plugin/src/host/{mod,reaper,geometry}.rs`；修改
`ara/{model,mapping}.rs`、`render/input.rs`、editor workspace/state；原kernel `state/model.rs`、
`mixdown.rs` 的共有变换入口仅在必要且保留App行为时修改。

**Interfaces（此任务定义）:**
```rust
struct HostClipGeometry {
    item_id:String, take_id:String, start_sec:f64, source_start_sec:f64,
    duration_sec:f64, playback_rate:f64,
    markers:Vec<hifishifter_kernel::state::ClipStretchMarker>,
    fade_in_sec:f64, fade_out_sec:f64, fade_in_shape:f64, fade_out_shape:f64,
    fade_in_dir:f64, fade_out_dir:f64, auto_fade_in_sec:f64, auto_fade_out_sec:f64,
}
```

- [ ] 先核对锁定ARA数据与REAPER官方扩展头/实际Host query返回，再固定typed ABI与
  当前插件所属track/item/take身份。未取得直接稳定绑定就保持缺口，不按位置/名字匹配。
- [ ] 读取真实宿主geometry快照，区分普通fade（宿主负责音频，仅投影GUI）与协商的
  content-based fade（插件负责一次）；把字段投影到原Clip/take，不修改宿主原工程。
- [ ] 字面映射/音频回归：source0..2秒映到项目1..5秒，rate0.5，裁切与markers分段映射
  不漂移；手工/自动fade逐端长度、shape/dir、overlap只应用一次。真实ordinary fade
  导出对照有插件/无插件；不能只比较非零hash。
- [ ] 稳定区域身份保存原编辑的局部/源坐标，移动/裁切/拆分/拉伸后重新投影到项目参数
  时间。旧v2状态兼容回归与pending冲突保留；没有对应曲线迁移前不能只删除Unsupported。
- [ ] 逐项真实REAPER验证倍率、保调、tempo/timebase、非线性markers与fade；事实录入
  captures，无法通过标准ARA提供的项由明确REAPER适配完成，不宣称原生跨宿主全支持。

## Task39：内容权威的HiFiGAN分层缓存

**Files:** 新增 `backend/hifishifter-plugin/src/render/cache.rs`；修改 `render/input.rs`；
共享kernel `mixdown.rs`、`renderer/` 与 `synth_clip_cache.rs`/`render_cache/` 仅复用实际契约。

**Interfaces（此任务定义）:**
```rust
struct HostRenderKey([u8;32]);
struct HostRenderCounters { source_hits:u64, intermediate_hits:u64, pcm_hits:u64, neural_runs:u64 }
// key输入来自冻结PCM、模型digest与有效局部参数；不能用viewId/地址/ara URI替代内容。
```

- [ ] 核对现有processor/mel/HNSEP/cache键与结果形态；明确宿主PCM不读file-only整clip
  cache。写真实源改变/参数改变/算法模型改变失效、同内容同参数跨owner命中测试。
- [ ] Source/分析、中间结果、目标PCM三层内容键与single-flight；命中与未命中返回同一
  形态，播放/导出共用；模型采样率结果派生44100/48000，不重复相同神经推理。
- [ ] 单独移动clip或普通fade/gain变更的真实counter断言：`assert_eq!(after.neural_runs,
  before.neural_runs);`；pitch/tension/source/model改变必须改变相关键并运行必要推理。
- [ ] 接入byte-budget LRU/受限磁盘配额、原子写入、格式版本/checksum、损坏拒绝/重建、
  取消/过期结果不污染成功缓存；收集实际内存/GPU峰值，不把缓存元数据命中当音频命中。
- [ ] HiFiGAN实际模型冷/暖、快速最后一笔/关GUI/冷重开/双轨不同参数实测，归档次数/
  耗时/PCM。WORLD通过不能代替此门；App原缓存回归必须仍通过。

## Task40：正常长源与资源约束

**Files:** `render/{source,input,snapshot,budget,cache}.rs`、kernel host PCM provider与analysis
适配、原私有waveform materialize；probe资源矩阵脚本。先保留旧安全cap，再按测试开放。

- [ ] 原30秒限制不作为最终完成门；增加超过30秒/正常完整人声/多轨用例，记录预算拒绝
  的真实来源（源、工作缓冲、GPU、中间层、播放快照），不是简单把常量提高。
- [ ] 有界source窗口/神经块准备与共享不可变chunk、源时间/参数对齐；在worker处理磁盘
  缓存/预热，实时seek读不到就明确pending/静音而非错误旧数据，最终就绪时输出精确。
- [ ] Offline导出必须完整供音，不能以慢worker尚未就绪造成静音；全范围预备/可验证的
  非实时准备契约与实时读取分离。任何磁盘映射缺页都不能被当作“零IO实时缓存”。
- [ ] peak RAM/GPU/缓存配额、退役快照/读者、长时间快速编辑、关闭取消与恢复资源验收；
  全部预算方案有证据，不能静默放宽原保护或缩小真实用户素材需求。

## Task41：四项目标最终集中门

- [ ] 一次集中运行App/kernel/plugin/frontend及native ABI/依赖/IPC测试与实际构建。
- [ ] 独立App真实原GUI导入编辑/播放保存；插件不启动它也能同窗口多轨完整编辑。
- [ ] Host位置/裁切/拆分/线性和非线性拉伸/tempo相关映射/普通与自动fade、seek/loop、
  44100/48000 mono/stereo/长人声、多轨/跨工程隔离/持续编辑与冲突取消，GUI与最终PCM一致。
- [ ] HiFiGAN冷/暖缓存计数与音频、关闭GUI供音、保存冷重开结果与缓存失效/损坏验证。
- [ ] 全部证据逐项审计后仅一次最终review；修重要项只定向重测。精确路径stage、本地
  commit，绝不push。只有四项目标及所有明确门都有当前证据才关闭goal。

## 当前状态

新范围尚未全部完成。Task36 renderer透传/诊断/typed REAPER时钟与Task38线性保调基础
已在ddbb5a97（lib80通过、host-stretch-01构建exit0，native尚未验收）；Task32在dfe8f4e
有授权scope并集，两定向测试通过；Task33共享actor在2cdf48ff，lib89/相关前端74/tsc
及document apply定向回归exit0；Task34在a5dfa654有12定向源码门，旧v2/真实stream/Weak/
尾笔/undo/无GUIWORLD双轨冷恢复通过，native多轨尚未验收。Task38a typed几何/能力契约
正在实施，完整marker/tempo/曲线迁移未完成。
完整geometry/曲线重投影/marker与tempo/HiFiGAN缓存/长源/native最终门仍open。
接口事实见probe/ara/HOST-GEOMETRY-FINDINGS.md。只源码，不自动恢复Esc停止的CU；
最终一次review。旧探针、线性音频测试、构建成功均不能用来勾完整宿主门。
