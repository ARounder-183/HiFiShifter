# Native material consolidation — implementation status

This is an implementation log, not a completed-release claim. Public version remains **0.25**. No GitHub publication is authorized until the user confirms the complete work.

## 当前集成状态（2026-10-22，非发布说明）

公开版本 **0.25**、WCSNDM **0.0803** 不变。用户当前要求是完成所有非自动注音功能后停止，交给用户验收；本轮不新增 ASR／TIFA 自动注音，也不把现有工程摆放或分析帧冒称词法恢复。下面早期检查记录保留历史语境，已修复的失败不能再视为当前失败，已通过的旧构建也不能替代最新改动的验证。

### 当前入口与已验证范围

- 三种显式轨道类型为 Hachi、UTAU、DiffSinger；新建／UST 导入默认 Hachi，渲染算法和轨道类型不是同一开关。类型模型、家族拒绝策略、不同编辑界面及既有工具栏路径已有编译测试；切换界面前通过 `cancelPendingEditingForTrackChange` 放弃旧轨未提交输入，不产生音乐撤销。源码导航见 [逐文件索引](AI_SOURCE_INDEX.md)。
- 同一物理录音可在一个 OTTO 中登记多套稳定 `audioId`；加载、编辑、拖放必须显式携带选中的 ID，歧义路径不能退化为“取第一套”。变体服务、模型及界面交接专项已通过，具体格式见 [OTTO 契约](otto-format.md)。
- 当前音符局部编辑复用会话和模型的局部应用／恢复接口，不发布源文件。已验证局部模型与会话服务；完整两个生产 GUI 入口及后续适配仍待核验，不据此宣布局部编辑全部完成。
- 社区数据集适配针对 openvpi 的 `transcriptions.csv`，保留输入明确提供的音素时间，使用显式字典或用户分组；没有独立 P/C 时必须确认将元音核起点同时作为 P/C。社区的 V 起始词划分不等于 C+V 音节。服务／格式测试及合成数据集的实际渲染链通过；新导入向导的宿主模拟 GUI 测试正在运行、尚未报告通过，也未完成生产 Main 接线验收，不属于已完成的生产入口。
- 工程素材登记服务只规划和复制现有物理录音，按独立标注变体写 OTTO；项目收纳只改引用路径与绑定，一次撤销，不重分析、不重写源时钟或调音。服务与模型测试通过；最新 28b 的真实 teio 登记前／登记收纳后／冷重开实际音频硬门槛也已通过，证据见下文。这不是由引用指纹相等推导“声音相同”；Main 可选登记控制器及其实际加载界面仍待接线验收。

### 声音、界面与证据边界

- 当前编译回归记录见 [resume-build27 验证输出](../../validation/resume-build27/smoke-project-material-registration/stdout.txt)、[模型收纳](../../validation/resume-build27/smoke-project-material-registration-model/stdout.txt)、[局部模型](../../validation/resume-build27/smoke-note-material-local-override/stdout.txt)、[三轨道界面](../../validation/resume-build27/smoke-track-editing-interfaces-ui/stdout.txt)。这一轮核心／模型／变体／家族／UI 回归通过，不是整体验收或全部后端通过。
- 父任务已复核恢复后的真实 Tokyo 生产默认渲染：与冻结接受结果的频谱／包络一致度为 100%，RMS 比为 1；teio 的 splice、process、process-merge 三模式也与各自旧执行程序基准保持 100%。这是特定工程、模型、输入与模式的回归结论，不推广成任意资源的承诺；既有原始参考与接受参考须分别保留。
- 全轨实际音高手势、人工连接图节点、显式作者曲线尾部支持的严格接缝测试已通过；旧 PBW 数据缺省不推断新作者意图。真实神经辅音保护合成夹具已通过固定 C／仅伸缩 V、25ms 对齐及冷重开 PCM 比较；这是明确生成的测试声音，不是原始录音词法对齐证明。
- 可选重叠提示默认关闭，只在现有 View 菜单控制音符内部的黄色横线，无整高竖带。单例素材选择器仍只能由右键显式打开，与显示开关独立。浅／深主题、真实时间宽度的放大截图和菜单动作已有专项检查；图片必须实际查看，不能以 PNG 存在代替视觉验收。
- USTX 使用严格 YAML 边界和公共谱面映射，处理 0.8 的 480 ticks、声部绝对位置和多轨道；不支持字段有原始信息／警告保留，不宣称完整 OpenUTAU 音色、表达式或渲染兼容。DiffSinger 家族错误请求有拒绝验证，但没有可用真实模型资源的正向神经渲染通过证据。

最新 28b 真实登记音频硬门槛的 [23 项实际输出](<G:/hjs-tc/tmp/teio-reg28/stdout.txt>) 已逐项回读：31 个完整物理 WAV 只复制一次，81 套携带注音变体对应 81 个绑定音符；收纳后与冷重开均从隔离目录明确按变体 ID 读回。前／后／冷重开的原生请求、F0 数组、源时钟和完整音乐快照一致，实际后端均为 `nsf-hifigan-onnx[cpu]+variable-mel-hop-splice-first`，无警告或回退；三次离线导出非静音，解码 PCM 逐字节相同。原工程／原始录音／模型字节 SHA 未变化。该证据验证的是实际核心登记链，不冒充尚未接线的 Main 登记 GUI 验收。

### 最新待验证改动与停止条件

1. 用户已明确确认“携带的摆放注音变体”：保留原片段边界、P 和 `-` 标签；已有 5513 个时钟结点、209 个原始分区及调音不应被格式化重建。它不是自动推断真正词组，也不把分析帧变为手工 V。
2. [ProjectMaterialRegistrationFacts](../juce/src/ProjectMaterialRegistrationFacts.h) 默认 `unresolved` 必须拒绝；仅显式 `ConfirmedCarriedPlacements` 允许上述解释。局部素材的基线 `-1` 哨兵必须原样保留，避免移动 P；作者／来源指纹新增基线覆盖。这些修正在 28b 的模型／登记事实专项及真实 teio 前后／冷重开音频中通过；不可因此跳过尚待整合的局部 GUI。
3. 登记租约在项目成功收纳后必须立刻 `retainAfterAdoption()`；即使后续素材根偏好写入失败，也只能提示重试，不能删除已被项目引用的新目录。一个物理 WAV 只复制一次，多个变体使用不同稳定注音 ID；不虚构一个跨文件／偏好／项目的全局原子提交。
4. 新社区向导的宿主模拟 GUI 测试仍在运行，未报告通过；Main 登记控制器／真实加载 GUI、局部第三方字段适配／两个生产入口和最终三种界面的人工图片复核仍有工作。完成这些非自动注音项目后停止等待用户验收，不进入下一阶段自动注音。
5. 未发布、未推送；不改公开版本，不增加尚未完成的正式更新编号。公开接口索引待头文件稳定后统一重新生成。

## Approved sequence（历史计划）

1. Compact folder-level `otto.ini`: audio sections, one pronunciation per line, stable identities, multiple aliases, explicit continuous links, multiple vowel segments and transition flags. Legacy annotation reads and explicit migration; lossy standard OTO export.
2. Shared material editing transactions and immediate propagation to current project references, without overwriting musical edits or local material overrides.
3. Folder / audio (virtual folder) / pronunciation tree; multi-select contiguous drag/drop and source-disambiguating pronunciation input.
4. Quick-edit wrench and material manager use the same multi-pronunciation editing model.
5. Restore omitted features and remove duplicate controls/implementations, preserving genuinely different operations.
6. Light-theme palette and compatibility, rendering and UI regression passes.

## Implemented foundation

- `OttoMaterialIO`: strict versioned UTF-8 codec; per-folder audio records and per-audio pronunciation IDs; compact defaults, optional exact legacy metadata; aliases escaped independently; multiple-vowel segments and transition/next fields.
- Atomic same-directory replacement, cooperative cross-process lock and explicit per-draft revision tokens. Unknown versions/fields and malformed documents cause an error rather than silent replacement.
- `SampleSettings::save` writes only OTTO. Explicit folder registration uses native OTTO first, classic OTO second, and creates a valid **empty** OTTO only when neither exists. Registration/import never invents filename pronunciations or rewrites source PCM. An existing OTTO also blocks legacy fallback for absent audio sections. Legacy CSV migration is explicit; annotation copies merge only the copied recording.
- `OttoMaterialMigration`: explicit folder transaction; preserve existing OTTO records and original audio/legacy annotations; standard OTO export merges by recording path, preserves unrelated rows/encoding, refuses native-manifest/audio targets and reports format losses.
- `MaterialAnnotationSession`: reusable source-time draft, batch split/alias assignment, adjacent merge, deletion, shared boundary movement, undo/redo, independent stale-editor detection and publication preflight. No source PCM is cut or rewritten.
- `OtoWaveformEditorComponent::forMaterial` extends the existing UTAU waveform editor. Native overview is the default; double-click enters the same precise timing implementation. The internal `MaterialAnnotationEditor` is only the overview child, not another independently launchable editor. Both the wrench and manager share the draft/publication service and existing audition host. Multi-pronunciation detail locks offset/cutoff typing **and** mouse handles; native segment selectors only appear in detail.
- Stable material identity and source-baseline snapshots in `NoteData`, saved with the project; source-update preflight/report and one transactional apply. Repeated placements are matched by identity, not row index. Native target pitch/envelopes/vibrato/gain remain intact. Local overrides and unresolved snapshots are retained.
- Asset manager has one precise-edit action, not separate native/compatibility commands. Format views switch within that editor and all native publication saves OTTO. Source preflight runs before disk publication. Folder/audio/pronunciation hierarchy, adjacent Ctrl/Shift selection, source identities in drag payloads and a source-disambiguating insertion candidate box are implemented.
- Main View owns display options; duplicate piano display buttons, root zoom buttons/sliders and duplicate file-action icons were removed. Source binding lives in the material manager. Label edits no longer switch a native track to UTAU or silently bind a default bank.
- Same-source merge configuration, directional connection-graph mutation, generic envelope presets, explicit engine-independent pitch/note drawing and scoped MIDI export use their existing authoritative services. Native NSF formant/tension/breath controls now reach the real ONNX path using one shared expression DSP; neutral controls skip processing.

## Historical foundation checkpoint (before the latest UI corrections)

Native Windows Clang-CL/xwin build on the relocated G-drive toolchain succeeded in a separate integrated build directory (old baseline build/cache not overwritten).

- `--smoke-otto-material`: **48/48** checks passed, including aggregate migration/export, subdirectory paths, aliases, exact metadata, repeated syllables, conflict rejection, legacy preservation and no implicit read-time migration.
- `--smoke-material-annotation`: **22/22** checks passed, including multi-cut continuity, envelope preservation, shared boundary movement, atomic draft operations, stale-editor rejection and unchanged source PCM bytes.
- `--smoke-material-binding`: all **30** assertions passed, including reordered repeated placements, preflight without mutation, preserved musical controls, coherent source clocks, undo/save/load, override isolation and unresolved references.
- Existing `--smoke-integrated`, `--smoke-native-hjm` (legacy compatibility), `--smoke-render-order`, `--smoke-glide-seam`, `--smoke-timeline-pit`, `--smoke-asset-manager`: exit **0** after migration-aware expectations.

These are historical targeted checks, not proof that the plan is finished or that old audio-similarity numbers apply to a new build. Later compiled checks include 99/99 unified-editor assertions with actual light/dark overview/detail captures, 18/18 material-tree checks with 220/320/400-width captures, real ONNX native-expression checks, decoded MIDI scope tests, directional graph tests and exact topology automation tests. Native-envelope, explicit drawing and asset-manager regressions also exit0 after matching the revised single-entry contracts. Individual failures and their repairs remain in the external validation logs; none is a substitute for the real-project gate below.

## 历史验收门槛与失败定位（保留至 build13b 的记录；当前结论见上文）

Each round must compile, run the compiled UI, capture PNGs and actually read them. Report picture and audio results separately. Static inventories and synthetic fixtures never substitute for the previously accepted real projects. Keep original project/media/model/engine hashes, solo scope and render order fixed; compare both original references and frozen accepted outputs. Do not weaken thresholds, count silence as singing similarity, claim human listening when none occurred, or call the native MLD5 substitute official Melodyne.

Latest measured results through build13b, **not release acceptance**:

- All 31 original teio sources match the archive. The normal production build12 reproduces frozen-original-executable **splice, process and process-merge at 100.0% spectral / 100.0% envelope / RMS ratio 1**, without diagnostic arguments. Missing old-project noise-protection defaults and UV carrier/decoder compatibility were repaired. An obsolete intermediate process WAV is not used to justify changing gain; the same frozen old executable is the reproducible oracle.
- Tokyo normal production still regresses near 97.2% spectral against the original reference. Exact tracing proved matching fixed resampler arguments but changed PIT. Solo0/harmony-muted state is explicitly logged. The `host-pit-own-nominal-bridge` diagnostic in build13 restores **100.0% spectral/envelope against the frozen accepted output**, and **98.9% spectral against the original reference**. Verified cause includes automatic transition windows expanded from 40ms to approximately 306ms. Production repair must preserve explicit whole-track edits and forced bridges; the diagnostic alone is NOT acceptance.
- The combined compiled native GUI workflow now runs with real GAME+FCPE analysis and NSF ONNX export; measured source provenance and full contour survive alias publication, original stereo source bytes remain unchanged, and cold save/reopen renders have exact PCM equality. Source topology/repeated-placement fixtures also retain exact PCM. Synthetic workflows do not replace the real-project gate.
- Build13b named-vowel, shared timing, adjacency, protected-time math, unified-editor, single-entry toolbar, material insertion/replacement model and codec commands pass. Whole-track actual gestures pass except a strict outside-curve numerical comparison (3 float ULPs versus fixed 1e-5 MIDI limit), and force-model testing exposed a fixture using normal lyric `R` instead of the canonical rest `RR`. Fixes are being tested, not hidden by relaxed tolerances.
- The first real neural consonant-timing fixture FAILED. Forensics proved its periodic cosine bursts were incorrectly labelled unvoiced, causing neural reconstruction rather than raw-source UV restoration. A physically aperiodic, equal-energy fixture and native source preflight are being added; all original timing/RMS/correlation gates remain unchanged. Neither source-clock assertions nor unchanged cold-reopen PCM alone count as acoustic consonant-preservation acceptance.

## 历史剩余事项与保守边界（具体进展以上文当前状态为准）

- Complete explicit linked source splits now propagate atomically through intact owners and repeated placements, preserve the old source clock and use exact cubic subdivisions/shared-envelope slices/vibrato phase offsets. Derived contours remain read-only facts rather than hundreds of new pitch handles.
- Unsafe cropped/merged/multi-owner/overlapping/gapped/independent/unpitched/robust owners, mixed pitch authorities, unsupported advanced expression layers, ambiguous links and irreproducible envelope-base saturation still reject preflight. These are explicit supported-rollout boundaries, **not completion of all topology cases**. Deleted source IDs retain unresolved musical snapshots instead of deleting notes.
- Re-run the complete combined native GUI workflow after every later graph/scope/source change; an older passing build is not current acceptance.
- Real compiled screenshots confirm modeless precision-editor ownership, the corrected light-theme fields, semantic one-C/one-V default and explicitly named `an/ng` subdivisions, plus seven fully visible tool icons at width1000 with a320-wide asset panel. Graph bridge handles/curve rendering and global artificial-seam continuity still require full compiled UI/audio acceptance. Broader local-override editor scope and all existing-language labels need final review.
- Pronunciation candidates currently insert at the playhead; fully integrating source replacement into inline pronunciation editing and searching unexpanded classic-OTO aliases remains unfinished.
- Native NSF controls are not falsely translated into unrelated external UTAU flags; unsupported backend capabilities must be reported rather than deleting stored data.
- OTTO extension policy remains strict; no guessed lossy metadata migration or whole-manifest copy is allowed.
- No final release, GitHub push or full-backend pass has been authorized/claimed.

## Build

`build-windows-native.ps1` defaults to `G:\hjs-tc`, with an isolated `build\hachishifter-integrated-otto` directory. It rejects a cache from a different source tree and isolates JUCE patches/binary state. Use `-Jobs 3`; no public version increment.
