# ARA Complete Integration Implementation Plan

**最新收尾（2026-10-05）：** 用户已两次确认release-delivery-02验收通过，并明确要求
分支改名为feature/ara-plugin后推送origin。按用户新授权覆盖本计划旧“不push”约束，
不强推、不改主develop、不推送ignored工程/构建产物、不再重复测试或新增review。
交付与验收证据/残余资源边界见docs/ara-release-acceptance.md；后续open状态段保留历史，
不把历史待办描述冒充本轮仍未获用户验收。

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans。
> 用户已指定本会话自主执行、不要逐步询问，测试按批末/最终集中；后续只在最终一次review。

**Goal:** 保留独立App，交付REAPER内同工程多轨原GUI、逐轨输出、完整clip变换与HiFiGAN缓存。

**Architecture:** 真实文档级原actor与逐renderer音频职责分离；ARA音频权威加经验证的宿主
元数据适配；复用原变换/模型管线，分层有界缓存与异步最新代次发布。

**Tech Stack:** Rust / ARA2 / VST3 / REAPER原生API（只有核对后使用）/ WebView2 / 原React与kernel。

**Spec:** `../specs/2026-10-05-ara-complete-integration-design.md`

**最新用户范围：** 只做线性拉伸，删除非线性marker/坡度/段内tempo warp实现与最终门。
整段倍率/tempo-timebase线性变化、参数迁移、保调PCM、持久化、fade等原其它要求保留。

**最新推理范围：** HNSEP 整段处理是用户确认的预期行为，不实现/验收其分块；
Task39 保留 HNSEP 参数、正确缓存和资源保护，Task40 的神经分块仅指 HiFiGAN。
HiFiGAN 四个专有参数（breath_enabled、breath_gain、hifigan_tension、formant_shift_cents）
以及共通参数按实际 descriptor/原管线验证；不能用 WORLD 结果代替真实模型证据。

## 最新用户BUG与交付方式（2026-10-05）

最新执行约束：用户反对反复测试消耗token，已停止Verify流水线；不再主动全量测试或
新增review。仅构建的All Release release-delivery-02已exit0，App/插件同源产物完成，
manifest Verify/native保持false。源码/构建不是实际GUI验收，完整目标仍未关闭。

最新用户追加：“渐变不一定完全还原，走hifishifter自己的也可以”。渐变显示门改为
真实宿主manual/auto长度及参数自动刷新，曲线/波形用HFS同源示意包络且明确标注；
不再要求精确REAPER新轴公式，不改变普通fade由宿主应用一次的音频责任。其余最终门
保留；下面“精确校准仍open”的段落是历史状态，不继续将它作为本轮完成阻塞。

以下四项新增必达，不缩小前述五项目标：

- [ ] 非当前轨道的曲线编辑自动生效，不需要手动重新载入宿主。
- [ ] 宿主手动/自动fade在原GUI显示正确；普通fade音频仍只由宿主应用一次。
- [ ] 气声开关/音高等参数连续编辑自动生效，不依赖手工重载或GUI读取原线。
- [ ] 原GUI播放头与所属REAPER项目实际播放位置同步，不被prefetch/其它renderer回跳。

用户明确“先一次性都做完，再叫我测”。停止Computer Use逐项试操作；完成源码/统一
批末回归/构建后再自动启动隔离REAPER，交用户集中测试。不在中途请求验收，不热
替换任何仍加载的DLL，不强杀/覆盖用户项目，最终仍只一次集中review。

当前actor发现两条可解释自动应用卡住的源码路径：source投影清项目原线key后，
完整clip分析cache命中不会重发ClipPitchReady；apply又把key为None当永远pending。
另一个是apply只在recv_timeout超时执行，连续只读轮询可使到期任务饥饿。现正在修
actor主动组装所有缺key根的已有cache、循环顶部执行到期任务，并等已入队写入处理完
才取最新票据。2026-10-05本批actor首轮28/29，修正新增张力夹具的分离开关后定向及
host合同14项exit0；前端28项/tsc exit0。另修渲染错误自动恢复、播放查询不排到合成后面、
宿主位置不加请求RTT及100ms插值上限。细节见probe/ara/EDITOR-SYNC-FINDINGS.md。
仍无本批native结论，不勾四BUG门。fade新轴只显示真实长度/原始轴；任意新曲线精确
绘制尚缺oracle，不把旧shape公式或数值显示当完整曲线已同步。

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
- [ ] 字面映射/音频回归：source0..2秒映到项目1..5秒，rate0.5，裁切/拆分/固定倍率映射
  不漂移；手工/自动fade逐端长度、shape/dir、overlap只应用一次。真实ordinary fade
  导出对照有插件/无插件；不能只比较非零hash。
- [ ] 稳定区域身份保存原编辑的局部/源坐标，移动/裁切/拆分/拉伸后重新投影到项目参数
  时间。旧v2状态兼容回归与pending冲突保留；没有对应曲线迁移前不能只删除Unsupported。
- [ ] 逐项真实REAPER验证倍率、保调、tempo/timebase整段倍率与fade；事实录入
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
- [ ] 有界source窗口/HiFiGAN神经块及批量准备与共享不可变chunk、源时间/参数对齐；HNSEP
  保留整段推理、核对其资源成本，不要求分块。在worker处理磁盘
  缓存/预热，实时seek读不到就明确pending/静音而非错误旧数据，最终就绪时输出精确。
- [ ] Offline导出必须完整供音，不能以慢worker尚未就绪造成静音；全范围预备/可验证的
  非实时准备契约与实时读取分离。任何磁盘映射缺页都不能被当作“零IO实时缓存”。
- [ ] peak RAM/GPU/缓存配额、退役快照/读者、长时间快速编辑、关闭取消与恢复资源验收；
  全部预算方案有证据，不能静默放宽原保护或缩小真实用户素材需求。

## Task41：四项目标最终集中门

- [ ] 一次集中运行App/kernel/plugin/frontend及native ABI/依赖/IPC测试与实际构建。
- [ ] 独立App真实原GUI导入编辑/播放保存；插件不启动它也能同窗口多轨完整编辑。
- [ ] Host位置/裁切/拆分/线性拉伸/tempo-timebase整段倍率/普通与自动fade、seek/loop、
  44100/48000 mono/stereo/长人声、多轨/跨工程隔离/持续编辑与冲突取消，GUI与最终PCM一致。
- [ ] HiFiGAN冷/暖缓存计数与音频、关闭GUI供音、保存冷重开结果与缓存失效/损坏验证。
- [ ] 全部证据逐项审计后仅一次最终review；修重要项只定向重测。精确路径stage、本地
  commit，绝不push。只有四项目标及所有明确门都有当前证据才关闭goal。

## 当前状态

## 新增最终交付Task45-47

用户最新目标追加以下产物，不缩小五项目标或四BUG；当前一次性probe流程需要正式收敛。
随后用户明确跨平台插件“本次不作要求”；Task47列为后续事项，不参与本轮完成审计。

- [ ] Task45：正式统一构建入口与构建文档，覆盖共享frontend/kernel、独立App、插件
  engine/原生loader/bundle、模型/运行DLL、debug/release、隔离启动、明确输出和错误。
  不热替换、不push，复用既有锁定SDK检出，不用git add -A。
  用户最新追加单独简短中文VST使用说明，不作多语言；初稿位于docs/hifishifter-vst-使用说明.md，
  最后集中验收后更新真实能力/限制，不把当前未校准项写成支持完成。
- [ ] Task46：App/插件功能同步说明及可执行护栏，唯一共享DSP/业务/GUI源码、宿主能力
  适配边界与修改位置、同批构建/测试矩阵；不能交付两份手工复制产品代码。
- 后续Task47（本次不要求）：Linux/macOS可行性与实现矩阵。核对平台VST3模块入口/UID/SDK工具链，
  评估原GUI嵌入和资源路径；区分共享内核/独立App可用、插件可编译、真实宿主可用。
  未知/平台实测缺失明确保留，不凭Windows外推跨平台；独立App原跨平台代码仍须保留。

## 当前状态（续）

本批集中回归frontend2724、kernel App配置587/插件配置585、App245均通过且正常退出。
Windows快照夹具不再写死/tmp，模型is_available改只读（App显式预热/实际按需加载保留），
五个快照与三个能力轮询护栏、真实短HiFiGAN/HNSEP均exit0。plugin首次全量154过/2失败，
四样本scope夹具显式旁路与v3 state断言修正后bound22/state1定向exit0；mapping/assignment/
exports/依赖及IPC共24项通过。未再重跑最后完整plugin；完整Verify/native/Release仍open。
同源All Debug regression-fixed-build-03已exit0，45文件摘要0差异；本次打包未再请求Verify，
manifest Verify/native保持false。详见probe/ara/REGRESSION-FINDINGS.md；不为这批测试启动
REAPER或中途叫用户验收。

旧v2首次恢复现已在真实assigned区域限定的完整图ready后捕获source basis，不等用户
再落笔。初载后移动+裁切+线性拉伸、局部音频投影、升级v3冷rebind合同与原真实归档v2
隔离合同共2项exit0。v2首次加载前已改变但未保存的原几何无法回推，不猜basis。
详见SOURCE-AUTHORITY-FINDINGS.md；不外推为完整native迁移/拆分矩阵已通过。
连续拆分的历史完整父范围与最近活子父范围现已通过live标记区分，保留历史且不选择
最小范围猜父；嵌套拆分、真活重叠歧义及原atlas/v2共7合同exit0，native仍open。
渐变资料核对：SWS旧GetMediaItemFadeBezParms/AskJF明确是旧视觉Bezier近似，不是
7.81双轴精确音频函数；未把它接到新曲线冒充完成。网络公式仍缺，需后续统一宿主oracle。

本批HNSEP算子profile确认末级97通道Concat占用随完整谱帧线性增长：10秒输出343MB，
三分钟其输入+输出存活下界约12.3GB。新增整段资源预检，成功cache hit不受其限制，
Windows物理/提交空间、Linux MemAvailable与失败信息合同通过；短真实模型/cache链
仍通过。没有改变HNSEP分块范围或模型。峰值降到普通机器可用、macOS资源查询与GPU
VRAM并未验证，不勾Task40。详细原始节点与保护边界见MULTITRACK-RESOURCE-FINDINGS。

Task45-46新增tools/build-hifishifter.ps1与docs/ara-build-and-sync.md：同源frontend、
App与插件分开的Cargo调用、全新交付名、MSVC后TEMP、锁定SDK、模型/DLL打包及可选
集中Verify。语法/PlanOnly exit0，不勾完整交付门。
随后真实All Debug paired-build-01 exit0；构建入口修复MSVC dot-source变量$name与
校验参数碰撞，使用BuildName内部变量保留-Name别名。带完整source/model摘要及45文件
manifest的paired-build-manifest-02 exit0，摘要全复核0差异；App有vslib、插件无vslib
导入，frontend一致。仅构建，不启动REAPER/不运行GUI；Release/Verify/native仍open。

2026-10-05最新资源批：双轨180秒原actor修改其它轨、两率完整尾部通过，旧ready与
新结果共存下显式额度峰值497,088,000字节，512MiB未增大。单region省mixed副本、
逐bit相同平面归并、跨算法轨原生率隔离与准备队列历史错误修复已实现。
真实180秒HiFiGAN/HNSEP完整模型输入及暖缓存通过，但旧CPUarena峰值18.3GB。
关Separator CPU arena/pattern后短mask逐bit一致，结束驻留降至1.3GB，瞬时峰值仍13.3GB。
详见probe/ara/MULTITRACK-RESOURCE-FINDINGS.md；需要继续产品级NN资源预检/峰值优化，
不勾Task40，不把显式PCM额度当系统RAM。HNSEP不分块；REAPER仍未启动。

2026-10-05最新源码批：内嵌workspace共享SourcePcm Arc，取消源/分析的30秒门，
两率PCM与准备临时域在固定512MiB额度内整批预检。单轨180秒GUI/两率完整尾部/seek
合同通过，显式额度峰值265,248,000字节，不等于模型/GPU/RSS峰值。实际源ID不变换
220→440Hz的原线自动失效/目标保留、无gen宿主移动自动准备也通过；私有内容路径保持
mtime，坏WAV原子重建。详情probe/ara/SOURCE-AUTHORITY-FINDINGS.md。
多轨长源新旧快照共存/稀疏区间、三分钟真实HiFiGAN宿主与资源峰值仍open，不勾Task40。
单独kernel无vslib测试的历史测试import编译门已修，仅测试作用域；3定向exit0，不等于
完整cargo test基线或独立App验收。REAPER未启动，最终统一构建/用户集中测仍待源码收尾。

2026-10-05补充：Task38重叠GUI投影/选择在e15a0792，10定向exit0；Task39/40已实现
HNSEP完整内容/model缓存与worker单飞、两类128MiB模型cache、HiFiGAN每批4块和
原生率合成派生48k。真实CPU四项模型诊断与相关定向回归正常exit0，精确结果见
`probe/ara/NEURAL-CACHE-FINDINGS.md`。仅kernel35秒长源，不代表宿主30秒cap已开放；
长源资源、HiFiGAN完整分层身份/移动命中/磁盘、native与独立App最终门仍open。
下方状态段保留旧批次背景；非线性marker/tempo warp已由用户明确排除。

新范围尚未全部完成。Task36 renderer透传/诊断/typed REAPER时钟与Task38线性保调基础
已在ddbb5a97（lib80通过、host-stretch-01构建exit0，native尚未验收）；Task32在dfe8f4e
有授权scope并集，两定向测试通过；Task33共享actor在2cdf48ff，lib89/相关前端74/tsc
及document apply定向回归exit0；Task34在a5dfa654有12定向源码门，旧v2/真实stream/Weak/
尾笔/undo/无GUIWORLD双轨冷恢复通过，native多轨尚未验收。Task38a在6b8fb67c/b5572a4e
有typed几何/能力/初始化重入合同（19新+6旧相关exit0）；Task38b有秒域映射与单GUI驱动
隐藏实例元数据/稳定版本缓存（11相关exit0），完整marker/tempo/曲线源权威与持久化未完成。
完整geometry/曲线重投影/marker与tempo/HiFiGAN缓存/长源/native最终门仍open。
接口事实见probe/ara/HOST-GEOMETRY-FINDINGS.md。只源码，不自动恢复Esc停止的CU；
最终一次review。旧探针、线性音频测试、构建成功均不能用来勾完整宿主门。
