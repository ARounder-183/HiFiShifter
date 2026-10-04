# HiFiShifter 接入 ARA2 · 探针计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用最小成本回答"HiFiShifter 能否作为 ARA2 插件接入 DAW"这一可行性问题，产出①宿主侧 ARA 模型的真实样本 ②Rust 接触 ARA 的可行路径 ③`ARA region → TimelineState` 转换的原型与逐样本比对结果。

**Architecture:** 三步串行，按风险降序。第一步完全不碰 Rust（用已装的 REAPER + Melodyne 与官方 ARADemoPlugin 采集真实 ARA 模型）；第二步解决唯一没有一手证据的风险（Rust 如何接触 ARA）；第三步把采集到的样本映射成 `TimelineState` 并交给现有 `render_mixdown_interleaved` 验证输出。

**Tech Stack:** REAPER 7.81 · Melodyne 5 (VST3, ARA) · Celemony ARA SDK (Apache-2.0) · CMake 4.4 + VS 2022 Community (MSVC 14.44) · Rust stable 1.97 · 现有 `backend/src-tauri` 的 mixdown/renderer 内核

**Spec:** `docs/superpowers/specs/2026-10-04-ara-bridge-design.md`

## Global Constraints

- **探针代码一律是可丢弃的**。所有实验产物放 `probe/ara/`，在计划结束时**整体删除或明确标注为一次性**。不得混入 `backend/`、`frontend/`。
- **绝不 push**。只做本地提交。
- **每个文件最上方必须有中文头注释**；每个关键函数必须有中文 doc 注释（仓库硬约定）。
- 分支：`feature/ara-bridge-probe`。开工前 `git status --short` 必须为空。
- **宿主版本**：REAPER 7.81，配置目录 `C:\Users\ARounder\AppData\Roaming\REAPER`，其中 `ara=2` 已启用。
- **ARA SDK 许可**：Apache-2.0（[Celemony/ARA_SDK](https://github.com/Celemony/ARA_SDK)）。
- **`vslib` 不参与任何探针步骤**：闭源、仅 Windows、文件 IO 型，本计划不触碰。

## 环境前提（开工前必须成立）

本分支的构建环境有两个已确认的坑。**不做这一步，后面所有失败都无法归因。**

1. **MSVC 环境**：`cc` / `cmake` 两个 crate 直接调用 `cl.exe`，从不初始化 MSVC 环境。缺变量时的症状具有误导性（`cl.exe` 明明存在，却报 `D8050: cannot execute 'c1xx.dll'`）。修复已随分支提供：

   ```powershell
   cd E:\code\HiFiShifter\.worktrees\ara-bridge-probe
   . .\tools\msvc-env.ps1
   ```

   若报"未对文件进行数字签名"，先在允许的会话里 `Set-ExecutionPolicy -Scope Process Bypass`。

2. **前端产物**：`build.rs` 在缺少 `frontend/dist` 时会执行 `npm run build` 并可能 panic。本分支已构建 `frontend/dist`，勿删除；若已删除，先 `cd frontend; npm install --ignore-scripts; npm run build`（`--ignore-scripts` 用于避开 npm 的 postinstall 子进程）。

3. **基线**：`cargo test` 基线数字由人工在普通终端取得后填入下表。**在基线为空的情况下开始改代码，后续任何失败都无法区分"本来就坏"与"改坏了"。**

   | 项目 | 基线值 |
   | --- | --- |
   | `cargo test` | **已取得（2026-10-04，`--no-fail-fast --jobs 1`）**：`backend_lib` 单测 777 个 → 772 passed / 4 failed / 1 ignored；`main.rs` 0 个；集成测试 17 个全通过（`loop_semantics` 10、`track_duplicate` 5、`smoke` 1、`reaper_export_rates` 1）；doc-tests 0 个。合计 **789 passed / 4 failed / 1 ignored**。 |
   | 既有失败项 | **不要修**。4 个全部在 `audio_engine::snapshot::tests`，同一原因：测试硬编码 POSIX 路径 `/tmp/hifishifter-*.aiff`，在 Windows 上解析为 `E:\tmp\…`，而该目录不存在，于是 `std::fs::write` 报 `Os { code: 3, kind: NotFound }`。与探针无关，改动前后一致。 |

   > **开发分支 `codex/ara-plugin` 上的基线**（内核抽取开始后）：
   > `backend/src-tauri` 的 lib 单测当前是 **762 passed / 4 failed / 1 ignored**，
   > `backend/hifishifter-kernel` 是 **10 passed**（`fade_curves` 7 + `byte_budget_cache` 3）。
   > 合计仍是 **772 passed / 4 failed / 1 ignored**（加集成测试 17 个通过）。
   > 后续每搬一个模块，都要按"app 侧减少、内核侧增加、合计不变"核对。

**已知限制（记录在案，不影响本计划）**：在本会话的沙箱内，`cargo test` 的原生 C/C++ 构建步骤会被间歇性阻断（症状在 `D8050` / `MSB6003: Failed to create a temporary file` 之间游走，失败点随 cargo 重试而游走）。已排除：`TEMP` 有效且可写、`cl.exe` 单独调用成功、无陈旧 MSBuild 临时文件。判定为沙箱对 cargo 孙进程编译器的干扰，**非仓库问题** —— 主树同名测试二进制可正常构建。故基线须在普通终端取得。

---

## Task 1: 取得宿主侧 ARA 模型的真实样本

**为什么第一步是它**：R2（映射无损）是全案最可能致命的假设，而验证它需要**真实的 ARA 数据**。本项目已有 REAPER 往返经验，但那是**文本格式（.rpp / 剪贴板）**，不是 ARA 的运行时对象模型。两者是否同构，目前只有推断。

**成本**：半天。**不碰 Rust。**

**Files:**
- Create: `probe/ara/README.md`（记录环境与结论）
- Create: `probe/ara/captures/`（样本落盘目录）
- Create: `probe/ara/ARADemoPlugin/`（SDK 示例，可丢弃）

**Interfaces:**
- Produces: `probe/ara/captures/ara-model.json` —— 一份带注释的真实 ARA 文档样本（`audioSources` / `audioModifications` / `playbackRegions` + 宿主 tempo map），供 Task 3 作为夹具使用。
- Produces: `probe/ara/captures/FINDINGS.md` —— 宿主实际给出的字段清单，以及"与 REAPER item 模型的同构程度"的实测结论（这是 spec §2.3 那张映射表的第一个实证）。

- [ ] **Step 1: 确认 REAPER 的 ARA 已生效（最低成本的一致性检查）**

用已装的 Melodyne 5 做一次"ARA 是否真的在工作"的判定，避免后面把 ARA 问题误判成构建问题。

```powershell
# 启动 REAPER，插入一条轨道，加载素材，在该轨道 FX 链加入：
#   C:\Program Files\Common Files\VST3\Celemony\Melodyne\Melodyne.vst3
# 判据：Melodyne 窗口内能看到该轨道音频的 blob/音符内容（而非"无音频"）
& "D:\Softwares\REAPER (x64)\reaper.exe"
```

Expected: Melodyne 显示该轨道音频内容。若显示"无音频"，说明 ARA 未生效 —— 先查 `REAPER.ini` 的 `ara=2` 与 `vstpath64` 是否包含 `C:\Program Files\Common Files\VST3`（当前已包含）。

- [ ] **Step 2: 克隆并构建官方 ARADemoPlugin**

Melodyne 只能证明"ARA 在工作"，不能给出可 dump 的对象模型与字段名，故需要官方示例。

```powershell
cd E:\code\HiFiShifter\.worktrees\ara-bridge-probe
New-Item -ItemType Directory -Force probe\ara | Out-Null
cd probe\ara
git clone --recurse-submodules https://github.com/Celemony/ARA_SDK.git
```

Expected: `ARA_SDK/` 下出现 `ARA_SDK` / `ARA_Examples` / `ARA_Library` 等子模块（伞形仓库，见 SDK README）。

随后按仓库内 `ARA_Examples` 与 `ARA_Library` 的 README 构建 Demo。**注意**：本计划不预设构建命令 —— 该 SDK 的 CMake 入口与 VST3 SDK 依赖方式需读了 README 才写，不得凭猜测执行。

- [ ] **Step 3: 在 Demo 的文档控制器里加一段 dump**

在 `ARADocumentControllerSpecialisation` 的文档变更回调处，把以下内容序列化为 JSON 落盘到 `probe/ara/captures/ara-model.json`：

- 每个 `ARAAudioSource`：稳定 id、名称、采样率、声道数、样本数
- 每个 `ARAAudioModification`：稳定 id、所属 source id、`startSample` / `endSample`、名称
- 每个 `ARAPlaybackRegion`：稳定 id、所属 modification id、`startSample`（时间线位置）、`duration`、`startInModificationSample`、`contentTransform`（`timeStretch` / `sampleRate` / `reversed`）、`name`、`color`
- 宿主 tempo map：`tempo` / `barSignature` / `beatSignature` 的变更点列表，以及 ARA 文档的 `musicalContextSampleRate`

判据：文件非空，且**字段名与 SDK 头文件一致**（不臆造字段）。

- [ ] **Step 4: 造一份带"难点"的素材，重采一次**

上面那份太干净 —— 它不会暴露问题。故意构造以下情形再采一次：

1. 同一素材**复制多次**放在不同位置（考察 source 复用与 region 多对一）
2. 对某个 region 做**拉伸**（考察 `contentTransform.timeStretch`）
3. 对某个 region 做**倒放**
4. 对某个 region 设**淡化**
5. 在**非 44.1kHz** 的工程采样率下重复一次（现有模型域固定 44.1kHz）

判据：`ara-model.json` 里能同时看到 1–4 的痕迹；第 5 项记录工程采样率与 ARA 的 `musicalContextSampleRate` 是否不同。

- [ ] **Step 5: 写 FINDINGS.md 并停手**

记录三件事，**不要在这一步就开始写转换器**：

1. ARA 实际给出的字段清单（与上面预期字段的差异）
2. **与 REAPER item 模型的同构程度**：哪些字段是 REAPER 语义的直接对应、哪些是 ARA 独有的、哪些是 REAPER 有而 ARA 没有的（后者才是真风险）
3. 明确列出**丢失的字段**：任何 HiFiShifter 渲染需要、但 ARA 不提供的输入

**杀死判据**：若官方 ARADemoPlugin 在 REAPER 里**无法加载**，且 30 分钟内无法定位到"是构建问题还是宿主兼容问题"，则停 —— 此时"Rust 自写 ARA 绑定"的预期成本要大幅上调（§Task 2 的 R1 直接受影响）。

```powershell
git add probe/ara
git commit -m "probe(ara): capture the real ARA document model from REAPER"
```

---

## Task 2: 确定 Rust 接触 ARA 的路径

**为什么第二步是它**：这是全案**唯一没有一手证据**的假设（spec §6 的 R1），而且它是**前置否决项** —— 映射再完美，碰到 ARA 的路不通就是零。

**成本**：1–2 天。

**Files:**
- Create: `probe/ara/rust-path/FINDINGS.md`
- Create: `probe/ara/rust-path/`（选中的路径的最小验证工程，可丢弃）

**Interfaces:**
- Consumes: Task 1 的 `captures/ara-model.json`（用于确认绑定能表达出这些字段）
- Produces: `probe/ara/rust-path/FINDINGS.md` —— 选定路径 + 证据 + 剩余风险，供 spec 的 R1 结论落定

- [ ] **Step 1: 评估路径 A —— `ara2-bridge` crate**

```powershell
cd E:\code\HiFiShifter\.worktrees\ara-bridge-probe\probe\ara\rust-path
cargo search ara2-bridge
cargo add ara2-bridge --dry-run
```

评估维度（逐条记录，不要只看 README 的宣称）：

1. 版本与最近提交时间（活跃度）
2. 是否覆盖 Task 1 实际需要的对象（`ARAAudioSource` / `ARAAudioModification` / `ARAPlaybackRegion` / `contentTransform` / tempo map）
3. **能否加载进 REAPER** —— 这是唯一有意义的判据，文档写得再好也不算
4. 它自带 VST3 外壳，还是需要外接 `vst3-sys` / nih-plug

- [ ] **Step 2: 评估路径 B —— 自写最小 ARA 绑定**

判断规模，不要直接开工。要回答的是"这一层有多大"：

1. ARA 是 COM 风格的 C API：需要手写的是接口表（vtable）声明 + `IUnknown` 引用计数语义。
2. 数一下为了**跑通 Task 1 那份样本所需的最小接口集**（文档控制器、音频源、修改、播放区间、内容读取器）—— 给出接口数量级。
3. 与仓库里既有的 FFI 先例对照 —— **代价主要在 vtable 声明，不在加载机制**：
   - 动态加载：`vocoder/gpu_info.rs` 用 `libloading::Library::new("nvml.dll")` 拿函数指针（nvml 不可用时优雅降级）
   - 静态 `extern "C"`：`vocoder/world.rs`、`audio/soundtouch.rs`、`audio/sstretch.rs`

   两种加载方式仓库里都已有可参照的实现；ARA 的额外成本是 COM 风格的接口表与引用计数语义，不在"怎么把库加载进来"。

判据：产出一个**接口清单 + 数量级估算**，而不是一句"可以做"。

- [ ] **Step 3: 选定路径并只做"能加载"的最小验证**

选定后，目标**仅有一个**：一个能装进 REAPER、被识别为 ARA 插件、并能在日志里打印出"我看到了 N 个 audioSource / M 个 playbackRegion"的最小工程。

```powershell
# 判据（必须全部满足）：
#   1. REAPER 的 FX 浏览器里能看到该插件
#   2. 挂到轨道后，插件被以 ARA 方式激活（而非普通 VST 插入）
#   3. 日志里打印出的 source/region 数量与 REAPER 里实际摆放的一致
```

Expected: 三项全过 → R1 成立，进 Task 3。任一项不过 → 记录失败点，**回到 Step 1/2 换路径重试一次**；两条路径都失败则触发杀死判据。

**杀死判据**：路径 A 与路径 B 在**合计 2 天**内都拿不到"能加载进 REAPER 并读到 ARA 对象"，则停 —— 结论是"Rust 侧需要先建一层 ARA 绑定，这是独立的、规模已知但不可忽略的前置工程"，并把该结论回报给 spec §6 的 R1。**此时不要继续 Task 3** —— 在没有可用绑定的情况下写的转换器无法验证。

```powershell
git add probe/ara/rust-path
git commit -m "probe(ara): establish the Rust ARA integration path"
```

---

## Task 3: `ARA region → TimelineState` 转换与逐样本比对

**为什么第三步是它**：R1 通过后，这是 R2 的直接检验；且转换器本身**就是插件里那个转换器的原型**，不是白做的探针。

**成本**：2 天。

**Files:**
- Create: `probe/ara/mapping/`（转换器原型 + 测试，可丢弃）
- Create: `probe/ara/captures/roundtrip/FINDINGS.md`
- Test: `probe/ara/mapping/tests/`

**Interfaces:**
- Consumes: Task 1 的 `captures/ara-model.json`
- Consumes: 现有 `crate::audio::mixdown::render_mixdown_interleaved(timeline: &TimelineState, opts: MixdownOptions) -> Result<(u32, u16, f64, Vec<f32>), String>`
- Consumes: 现有 `crate::state::{TimelineState, Clip, Track, TrackParamsState}`
- Produces: `fn ara_document_to_timeline(doc: &AraDocument) -> Result<TimelineState, MappingError>` —— 转换器原型（签名在此固定，供后续实现计划引用）。**`AraDocument` 是本探针内定义的反序列化类型**，结构对齐 Task 1 落盘的 `ara-model.json`；它不是 ARA SDK 的类型 —— 探针刻意让转换器只依赖样本文件，从而在 Task 2 的绑定尚未稳定时也能独立验证映射。

- [ ] **Step 1: 写失败测试 —— 转换器必须无损**

```rust
// probe/ara/mapping/tests/mapping.rs
// 判据：Task 1 采集的样本转成 TimelineState 后，渲染所需的每个字段都能在
//        TimelineState 里找到对应，且值一致。
#[test]
fn ara_regions_map_to_clips_without_losing_render_inputs() {
    let doc = load_fixture("captures/ara-model.json");
    let tl = ara_document_to_timeline(&doc).expect("mapping must succeed");

    // 每个 playbackRegion 对应一个 Clip
    assert_eq!(tl.clips.len(), doc.playback_regions.len());

    // 位置与长度（时间线秒）
    for (clip, region) in tl.clips.iter().zip(doc.playback_regions.iter()) {
        assert!((clip.start_sec - region.start_sec).abs() < 1e-9);
        assert!((clip.length_sec - region.duration_sec).abs() < 1e-9);
    }

    // 拉伸必须落到 playback_rate（否则渲染输出长度错）
    // 倒放必须有表达（否则渲染方向错）
    // 淡化必须落到 fade_in_shape / fade_in_dir（否则包络错）
}
```

- [ ] **Step 2: 运行测试，确认失败**

Run: `cargo test --manifest-path probe/ara/mapping/Cargo.toml ara_regions_map_to_clips_without_losing_render_inputs`
Expected: FAIL —— `ara_document_to_timeline` 未定义。

- [ ] **Step 3: 实现转换器**

按 spec §5.2 的映射表实现。**复用既有 REAPER 语义**（`import/reaper_import.rs`、`fade_curves.rs`、`state.rs` 的 take/拉伸标记/loop），不要为 ARA 另立一套。

**源的身份映射是这一步最危险的子问题**：spec §4.2 已定"宿主是权威源"，故 `Clip.source_path` 不能指向本地文件 —— 需要一个"源由 ARA 提供 PCM"的表示。这一步先把该表示定下来（哪怕只是占位 enum），**因为渲染缓存键依赖它**。

- [ ] **Step 4: 运行测试，确认通过**

Run: `cargo test --manifest-path probe/ara/mapping/Cargo.toml`
Expected: PASS。

- [ ] **Step 5: 写"丢失字段"测试 —— 这一步比上面的通过更重要**

把 Task 1 的 `FINDINGS.md` 里列出的"ARA 不提供、但渲染需要"的字段，逐个写成显式的失败/降级测试。

这是本计划**最有价值的产出**：它把"映射不全"从一句担心变成一份可核对的清单。若某个字段确实无法获得，正确的做法是让转换器**显式降级并在测试里钉住降级行为**，而不是悄悄填默认值。

- [ ] **Step 6: 逐样本比对**

把转换出的 `TimelineState` 喂给现有 `render_mixdown_interleaved`，与"同一素材在 HiFiShifter 本体里手工拼出来的结果"逐样本比较。

判据（先定死，避免事后找理由）：

- 采样率与总线数一致
- 有效区间的逐样本最大绝对差低于一个**事先约定**的阈值（建议 1e-6；若因拉伸算法实现差异导致更大，必须在 FINDINGS 里解释来源，而不是直接放宽阈值）
- 长度一致

**杀死判据**：若"丢失字段"清单里包含**渲染无法降级**的输入（即缺了它就无法产生正确音频），则 R2 不成立 —— **整个进程内 ARA 方案的 v1 范围必须重写**，并回报 spec §6 的 R2。这是本计划希望尽早买到的那个结论。

```powershell
git add probe/ara
git commit -m "probe(ara): prototype the ARA-to-TimelineState mapping with sample-level comparison"
```

---

## 收尾

- [ ] 把三步的实际结论回写 spec 的 §6 风险表（R1 / R2 从"假设"变成"已验证"或"已否决"）
- [ ] 明确处置 `probe/`：整体删除，或保留并在 README 标注为一次性
- [ ] 把基线数字填入本计划的"环境前提"

---

## Review Focus

以下是 spec 未写、但最可能咬到使用者的输入类别。各自由对应任务的测试钉住：

| 输入 / 情形 | 合理预期 | 归属任务 |
| --- | --- | --- |
| 同一 source 被多个 region 复用 | 每个 region 独立渲染，互不串台 | Task 3 Step 1 |
| 工程采样率 ≠ 44.1kHz（模型域） | 内容不错位、不压缩 8.1% | Task 1 Step 4 / Task 3 Step 6 |
| region 被拉伸 / 倒放 | 输出长度与方向正确 | Task 3 Step 1 |
| 宿主在播放中改源内容（同路径） | 不静默复用过期渲染缓存 | Task 3 Step 3 |
| region 长度为 0 或极短 | 不 panic、不产生非有限样本 | Task 3 Step 6 |
| 淡化长度为 0 / 曲率为极端值 | 包络与本体一致 | Task 3 Step 1 |
| 空文档（无 region） | 转换成功且产生空时间线，不报错 | Task 3 Step 1 |
