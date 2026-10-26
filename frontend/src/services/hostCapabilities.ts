// 原GUI的运行模式与能力边界；独立app保持原窗口/文件/设备行为，插件由DAW供源。
import { translateOutsideReact } from "../i18n/I18nProvider";
import type { MessageKey } from "../i18n/messages";
import { hostMode as detectHostMode, pluginBootstrap } from "./hostDetect";
import type { PluginBootstrap } from "./pluginHost";

export type HostMode = "standalone" | "plugin";
/** 必须使用显式bootstrap，而不是把所有WebView2都误认为插件。判据底层在 hostDetect。 */
export function hostMode(): HostMode {
    return detectHostMode();
}
export function isPluginMode(): boolean {
    return hostMode() === "plugin";
}
/**
 * 宿主写能力的**唯一**读取入口：后端 `hostCapabilities` 快照。
 *
 * 【为什么只有一个来源】此前 `clipEditing` / `audioImport` 等顶层布尔与
 * `hostCapabilities` 并存，后端两处各自计算 —— 必然漂移。现在统一从这里读。
 */
function caps(): PluginBootstrap["hostCapabilities"] {
    return pluginBootstrap()?.hostCapabilities;
}
/** 仅宿主原生明确提供ARA播放请求能力时开放，不把插件模式等同支持播放控制。 */
export function canControlHostTransport(): boolean {
    return isPluginMode() && pluginBootstrap()?.transportControl === true;
}
/**
 * 播放光标是不是**宿主**的，且插件改不动它。
 *
 * 【为什么单独一个谓词】"能不能控制传输"（播放/暂停）与"能不能移动光标"是两件事，
 * 只是当前恰好都由同一个 ARA 播放请求提供。将来若出现只给播放、不给定位的宿主，
 * 分叉点在这里，而调用点不用动。
 *
 * 【为什么它必须存在】非 ARA 宿主里 `set_transport` 必然失败，而前端每一次"跳转"
 * 都是"乐观写 store + 打后端"两步：光标先跳过去，随后 30 Hz 的播放轮询用宿主真实
 * 位置把它拽回来。宿主拥有光标时，镜像不该写源。
 */
export function playheadIsHostOwned(): boolean {
    return isPluginMode() && !canControlHostTransport();
}

/**
 * 只有本次原生入口明确具备写 API **且**有宿主撤销栈，才打开片段拖拽/裁切/线性拉伸。
 *
 * 【为什么还要 `undoStack`】后端 `effective_host_capabilities` 把 `clipWrite`
 * （`can_edit_clips`）与 `undo_stack`（`has_project_history`）分开报；两者同时成立
 * 才等于"有 project history 的宿主"，也就是此前的 `clipEditing` 语义。
 */
export function canEditHostClips(): boolean {
    const c = caps();
    return isPluginMode() && c?.clipWrite === true && c.undoStack === true;
}
/** 分割有独立原生能力门，不因允许拖拽就放开未实现的片段命令。 */
export function canSplitHostClips(): boolean {
    return isPluginMode() && caps()?.split === true;
}
/** 原生完整item剪贴板及创建/删除接口已接通时开放，旧插件不会误发App私有媒体命令。 */
export function canClipboardHostClips(): boolean {
    return isPluginMode() && caps()?.itemClipboard === true;
}
/** 只允许插件私有参数分组，不代表能重排或创建REAPER的轨道/folder。 */
export function canGroupPluginTracks(): boolean {
    return isPluginMode() && pluginBootstrap()?.trackGrouping === true;
}
/** 独立App保留原几何编辑；旧插件或其它宿主缺写接口时保持只读。 */
export function isHostGeometryReadOnly(): boolean {
    return isPluginMode() && !canEditHostClips();
}
/**
 * 宿主的淡化轴可编辑。
 *
 * 【为什么不能沿用 `isPluginMode()`】淡变形状**不是**宿主的独占领域：写的是宿主
 * 自己的轴（≤7.80 的 `C_FADE*SHAPE`/`D_FADE*DIR`，≥7.81 的 `D_FADE*DIR_NEW`/
 * `DIR2_NEW`），声音也由宿主渲染。此前用模式一刀切，于是"长度能改、形状不能改"
 * 这个不一致一直留着 —— 而形状本来就能改。
 *
 * 前提是**片段几何可写**（走的是同一条 `host_edit` 路径）且宿主版本能判别轴语义；
 * 后者读不出来时保持只读，不猜。
 */
export function canEditHostFadeAxes(): boolean {
    return isPluginMode() && canEditHostClips() && hostFadeAxes() !== null;
}

/**
 * 淡变**长度**能否编辑。
 *
 * 【为什么与形状分开】长度落在 `D_FADE*LEN` / `D_FADE*LEN_AUTO` 四个键上，与宿主的
 * 轴版本**无关**（`editor/host_edit.rs` 的写口把它们和轴写入分开）。所以即使轴版本
 * 读不出来（`fadeAxes === null`，形状/曲率不可写），长度拖拽照常可用 —— 此前用一个
 * `fadeShapeReadOnly` 布尔把两者一起关掉，用户连长度都调不了。
 */
export function canEditFadeLength(): boolean {
    return !isPluginMode() || canEditHostClips();
}

/**
 * 淡变**形状**能否编辑。
 *
 * - 独立 App：用 HiFiShifter 自己的形状轴，永远可写；
 * - legacy 宿主（≤7.80）：写 `C_FADE*SHAPE`；
 * - continuous 宿主（≥7.81）：写 `(curvature, S)` 预设对；
 * - 版本读不出来：不可写，也不猜。
 */
export function canEditFadeShape(): boolean {
    return !isPluginMode() || (canEditHostClips() && hostFadeAxes() !== null);
}

/**
 * 淡变**曲率**能否编辑。
 *
 * 与形状同门槛：两者都要宿主版本分得清轴语义（`fade_axes_v2`），否则不知道曲率该写
 * 哪个键（legacy `D_FADE*DIR` / continuous `D_FADE*DIR_NEW`）。
 */
export function canEditFadeCurvature(): boolean {
    return canEditFadeShape();
}

/**
 * 淡变的 **S 参数轴**能否编辑。
 *
 * 只有 REAPER ≥7.81 有这根轴（`D_FADE*DIR2_NEW`）。legacy 宿主写它会被后端拒绝
 * （`validate_fade_axes`："fade S parameter requires REAPER 7.81 or later"）。
 */
export function canEditFadeS(): boolean {
    return !isPluginMode() || (canEditHostClips() && hostFadeAxes() === "continuous");
}

/**
 * 宿主用哪一套淡化轴；`null` = 版本读不出来。
 *
 * 独立 App 没有这个概念（它用自己的曲率轴），所以非插件模式返回 `null`；
 * 调用方应先判 [`canEditHostFadeAxes`]。
 */
export function hostFadeAxes(): "legacy" | "continuous" | null {
    return pluginBootstrap()?.fadeAxes ?? null;
}

/**
 * 撤销历史由谁承载：`"host"` = 宿主权威栈，`"local"` = 插件自管栈。
 *
 * 【为什么需要】「操作记录」面板里 `label` 的含义取决于它：本地栈是 op key（要按
 * `history_op_<key>` 本地化），宿主栈是宿主自己的可读字符串（原样显示）。缺省
 * `"local"`，与后端 `BootstrapFlags.history_backend` 的缺省一致。
 */
export function historyBackend(): "host" | "local" {
    return pluginBootstrap()?.historyBackend === "host" ? "host" : "local";
}

/**
 * 当前编辑后端身份：`"reaper"`（宿主给了原生扩展）或 `"ara_only"`（只有 ARA 2.0）。
 *
 * 【为什么需要】ARA-only 后端上片段几何只读。缺省按 `"ara_only"`（非插件模式 / 旧后端
 * 没有这个字段时），与后端 [`HostCapabilities::ara_only`] 的保守缺省一致。
 */
export function hostKind(): "reaper" | "ara_only" | "capture" {
    // 【为什么三值原样透传】此前非 reaper 一律折成 `ara_only`，于是后端发的 `capture`
    // 在前端消失 —— 任何按 `hostKind` 分流的地方（几何只读、引导页、探活）都区分不出
    // Capture。缺省仍按最保守的 `ara_only`（旧后端没有这个字段时）。
    const kind = pluginBootstrap()?.hostKind;
    return kind === "reaper" || kind === "capture" ? kind : "ara_only";
}

/**
 * 预设形状按钮是否可用。
 *
 * 【为什么新轴宿主上也能用】REAPER ≥7.81 由 curvature/S 两个连续轴决定形状，
 * 而"预设 → (curvature, S)"的映射是**实测**出来的（`timeline/hostFadeAxes.ts`，
 * 证据 `probe/ara/FADE-AXIS-FINDINGS.md`）：七个预设各自对应一组确定的坐标，两轴
 * 正交，没有表达不到的预设。所以新轴宿主上照常摆七个按钮，点了写两个分量。
 *
 * 此前这里只放行 `legacy`，理由是"映射尚未校准" —— 那条理由已经随实测消失。
 * 仍然只放行"版本读得出来"的宿主：`null` 时轴语义未知，不猜。
 */
export function canSelectHostFadeShape(): boolean {
    return !isPluginMode() || hostFadeAxes() !== null;
}

/**
 * 速度映射的 **BPM / 拍号轴**能否编辑。
 *
 * 【为什么插件里不成立】两者都是**宿主权威**：BPM 与拍号经 VST3 进程上下文读入
 * （`render::transport` 的 `tempo()` / `time_signature()`），插件只读不写。
 * 界面必须把这两个字段显示成只读并说明原因，而不是让用户改完才发现被覆盖。
 *
 * 【为什么音阶轴不需要对应的函数】Tempo Map 不是"一组宿主参数的编辑器"，它是
 * **随时间变化的音阶**的存储：`TempoPointData.scale` 就是音阶覆盖，
 * `TimelineState::scale_segments()` 由它产出逐段的生效音阶，渲染缓存键
 * （`scale-signature`）与子轨级数渲染都锚定它。音阶是 HiFiShifter 自有的
 * （REAPER 没有工程调号概念），因此在插件里**永远**可编辑 —— 一个恒真的闸门只是噪声，
 * 所以速度映射的菜单组与按钮在插件里照常渲染，只有 BPM/拍号字段是只读的。
 */
export function canEditTempoMapTempo(): boolean {
    return !isPluginMode();
}
/** 文件菜单只开放明确具备宿主媒体创建能力的音频导入，不放开项目文件/设备命令。 */
export function canImportHostAudio(): boolean {
    return isPluginMode() && caps()?.mediaImport === true;
}

/**
 * 插件里能不能**导出处理结果到文件**。
 *
 * 【为什么插件里成立】导出只读已授权的 PCM（`ARAAudioAccessControllerInterface`
 * 对读取**没有角色限制**）再写用户选定的文件 —— 不需要任何宿主写接口。插件侧的
 * `export_audio_advanced` / `quick_export_selected_clips` / `save_synthesized` 全部
 * 共用播放那条渲染数学（`RenderInput::mix_interleaved`），因此导出与听到的一致。
 *
 * 【为什么必须单独一个谓词】此前 `pluginAllowsAction` 把 `project.*` 整段拒绝，
 * 于是**快捷键**（`Ctrl+E`）是死的 —— 菜单能点、键盘没反应，正是本仓明令禁止的
 * 静默失效。导出是唯一一条不属于"宿主独占"的 `project.*` 动作。
 */
export function canExportAudio(): boolean {
    return !isPluginMode() || caps()?.exportAudio === true;
}

/**
 * 插件里能不能导出**纯数据文件**（布局 / 主题 / 颤音预设）。
 *
 * 【为什么插件里成立】三者都是"把前端已经算好的 JSON 存成文件"，不碰宿主、不碰
 * 音频、不碰时间线。
 */
export function canExportDataFiles(): boolean {
    return !isPluginMode() || caps()?.exportData === true;
}

/**
 * 插件里能不能**新建一条宿主轨道**。
 *
 * 【为什么不复用 `canImportHostAudio`】导入音频会顺带建轨，所以导入蕴含建轨；但
 * 反过来不成立 —— 原生侧的门槛也不同（建轨还需要 `InsertTrackInProject` 与 FX
 * 接口齐备）。用导入能力门住"添加轨道"，会让一个不需要媒体导入的功能随它一起消失。
 *
 * 注意：新建的轨道**还没有音频**。要让这个 HiFiShifter 实例有内容，得再往这条轨道
 * 导入音频（见 [`canImportHostAudio`]）—— UI 必须把这件事说清楚。
 */
export function canCreateHostTracks(): boolean {
    return isPluginMode() && caps()?.trackCreate === true;
}

/**
 * 插件里能不能由插件**建立轨道组**（REAPER 的 folder 父子）。
 *
 * 【为什么是"不能"而不是"尽力而为"】"为每个文件夹创建轨道组"要么建出真正的
 * folder 层级，要么什么都不建。宿主的 folder 由每条轨道的 `I_FOLDERDEPTH` 编码，
 * 而插件对轨道结构是**只读**的（`host/folder.rs` 的类型里根本没有写入口）；退化成
 * 一串彼此无关的空轨道，用户看到的是"分组标题"，实际什么也没分 —— 比不做更坏。
 *
 * 【替代路径是完整的】不带轨道组的目录导入在插件里完全可用：文件按所选排布方式
 * 落到宿主轨道上（`importHostBatch`），只是不建立 folder 层级。
 */
export function canCreateHostTrackGroups(): boolean {
    return !isPluginMode();
}

/**
 * 插件里能不能**一次把多个文件作为 Take 导入**。
 *
 * 【为什么不能】`importMultipleAudioAtPosition` 的 `as-takes` 分支在插件模式下
 * 直接拒绝：多 take 的**创建**没有宿主对应（ARA 侧只能读宿主已有的 take，不能往
 * 一个 item 里新建 take）。此前这个选项照常可选，点了只得到一条无人处理的拒绝。
 *
 * 注意与 Part 2 的多 take **枚举**是两件事：读多个 take 已经实现（每个 take 都是
 * 一个独立片段），这里说的是往同一个 item 里**新建** take。
 */
export function canImportAsTakes(): boolean {
    return !isPluginMode();
}

/**
 * 插件里能不能**把 MIDI 的音符导入到音高曲线**。
 *
 * 【为什么插件里能做】`import_midi_to_pitch` 写的是插件自己的音高曲线
 * （`params_by_root_track`，与参数编辑器同一份数据），既不碰宿主几何，也不需要任何
 * 宿主写接口 —— 它是纯数据变换。此前整条链路被挡掉，只是因为"还没搬进内核"，
 * 现在 App 与插件跑的是同一份实现（`kernel::editor::midi_import`）。
 *
 * 【为什么仍要这个函数】导入对话框要区分"导入到曲线"和"建成片段"两件事：插件里
 * 只有前者成立，UI 必须把选项摆对，而不是让用户选完才失败。
 */
export function canImportMidiToPitch(): boolean {
    return true;
}

/**
 * 插件里能不能**把 MIDI 建成片段**（新建 MIDI clip / 替换已有 MIDI clip 的数据）。
 *
 * 【为什么不能】`import_midi_as_clip` / `replace_midi_clip_data` 建的是**本地片段**，
 * 而插件的时间线是宿主清单的投影（`workspace_timeline_locked` 只保留已分配 region
 * 的 clip）：造出来的片段在下一次宿主同步时就会消失。那不是"没实现"，是在这里做不到。
 *
 * 【替代路径是完整的】"导入到音高曲线"在插件里可用 —— 用户要的"把这段 MIDI 的音符
 * 变成我的编辑内容"由此满足；只有"在时间线上多出一个 MIDI 片段"这一步没有。
 */
export function canImportMidiAsClip(): boolean {
    return !isPluginMode();
}

/**
 * 宿主接管范围的说明文案（被禁用项的 tooltip、以及"该窗口在插件里不可用"的异常）。
 *
 * 【为什么是函数而不是常量】常量在模块加载期取值，语言就冻在那一刻的
 * localStorage 上 —— 用户切换语言后这串文案不会跟着变。调用点分散在菜单、
 * 右键菜单与命令式异常里（都不是组件），因此走 `translateOutsideReact`：
 * 它在无浏览器环境下回落英文词典，不会因为取一条文案而抛错。
 */
export function dawControlledReason(): string {
    return translateOutsideReact("plugin_daw_controlled_reason");
}

// ── 能力矩阵 ────────────────────────────────────────────────────────────────

/**
 * 一条能力。**每一条都必须能回答「为什么不能」** —— 这正是此前缺的那一半。
 *
 * 【为什么要有这张表】在此之前，二十条谓词各自被二十处调用点直接使用，而「不能做」
 * 时的说明只有一句通用文案（dawControlledReason）。于是用户看到的是「宿主接管」，
 * 而真实原因可能是「宿主没给播放位置」「这条命令没实现」「只在独立 App 有意义」——
 * 三者对用户意味着完全不同的下一步。
 */
export type CapabilityId =
    | "transport.control"
    | "clip.edit"
    | "clip.split"
    | "clip.clipboard"
    | "clip.geometry"
    | "clip.normalize"
    | "track.create"
    | "track.group"
    | "track.edit"
    | "fade.length"
    | "fade.shape"
    | "fade.curvature"
    | "fade.s"
    | "tempo.bpm"
    | "media.import"
    | "midi.importPitch"
    | "midi.importClip"
    | "export.audio"
    | "export.data"
    | "take.import"
    | "project.file"
    | "recording.local"
    | "playback.metronome"
    | "surface.pianoRollOnly"
    | "host.keyboardHandoff";

/** 一条能力的判定结果。 */
export interface CapabilityDecision {
    allowed: boolean;
    /** 不可用时的理由 key；可用时为 null（不需要解释）。 */
    reason: MessageKey | null;
}

/**
 * 能力 → 判定。表里每一条要么给谓词，要么给常量 false（永远不成立的能力）。
 *
 * 【为什么把「永远不成立」也写进表】它们此前是散在 pluginAllowsAction 里的裸 false。
 * 裸 false 没有地方挂理由，于是界面上只能退化成通用文案。
 */
const CAPABILITIES: Record<CapabilityId, { allowed: () => boolean; reason: MessageKey }> = {
    "transport.control": {
        allowed: canControlHostTransport,
        reason: "cap_reason_unsupported_by_host",
    },
    "clip.edit": { allowed: canEditHostClips, reason: "cap_reason_unsupported_by_host" },
    "clip.split": { allowed: canSplitHostClips, reason: "cap_reason_unsupported_by_host" },
    "clip.clipboard": { allowed: canClipboardHostClips, reason: "cap_reason_unsupported_by_host" },
    "clip.geometry": { allowed: () => false, reason: "cap_reason_host_owns" },
    "clip.normalize": { allowed: () => true, reason: "cap_reason_host_owns" },
    "track.create": { allowed: canCreateHostTracks, reason: "cap_reason_unsupported_by_host" },
    "track.group": { allowed: canGroupPluginTracks, reason: "cap_reason_unsupported_by_host" },
    "track.edit": { allowed: () => false, reason: "cap_reason_host_owns" },
    "fade.length": { allowed: canEditFadeLength, reason: "cap_reason_unsupported_by_host" },
    "fade.shape": { allowed: canEditFadeShape, reason: "cap_reason_unsupported_by_host" },
    "fade.curvature": { allowed: canEditFadeCurvature, reason: "cap_reason_unsupported_by_host" },
    "fade.s": { allowed: canEditFadeS, reason: "cap_reason_unsupported_by_host" },
    "tempo.bpm": { allowed: canEditTempoMapTempo, reason: "cap_reason_host_owns" },
    "media.import": { allowed: canImportHostAudio, reason: "cap_reason_unsupported_by_host" },
    "midi.importPitch": { allowed: canImportMidiToPitch, reason: "cap_reason_unsupported_by_host" },
    "midi.importClip": { allowed: canImportMidiAsClip, reason: "cap_reason_unsupported_by_host" },
    "export.audio": { allowed: canExportAudio, reason: "cap_reason_unsupported_by_host" },
    "export.data": { allowed: canExportDataFiles, reason: "cap_reason_unsupported_by_host" },
    "take.import": { allowed: canImportAsTakes, reason: "cap_reason_unsupported_by_host" },
    "project.file": { allowed: () => false, reason: "cap_reason_app_only" },
    "recording.local": { allowed: () => false, reason: "cap_reason_app_only" },
    "playback.metronome": { allowed: () => false, reason: "cap_reason_app_only" },
    "surface.pianoRollOnly": { allowed: () => false, reason: "cap_reason_piano_roll_only" },
    // 交还键盘是"本窗口与宿主的关系"：独立 App 里没有宿主，这条动作没有意义。
    "host.keyboardHandoff": { allowed: isPluginMode, reason: "cap_reason_plugin_only" },
};

/**
 * 查询一条能力。独立 App 形态下绝大多数能力恒真（谓词内部已经处理了这一点）。
 *
 * 【为什么返回对象而不是布尔】布尔能回答「能不能」，回答不了「为什么不能」。
 * 界面上每一个 aria-disabled 的键都该有一条说得清的理由，理由只能从这里来。
 */
export function capability(id: CapabilityId): CapabilityDecision {
    const entry = CAPABILITIES[id];
    const allowed = entry.allowed();
    return { allowed, reason: allowed ? null : entry.reason };
}

/**
 * 这条动作**为什么不能做**的成句文案；能做的或不受宿主限制的返回 null。
 *
 * 【为什么用 pluginAllowsAction 作闸门】这样它与真正的判据逐字节同源 —— 若哪天两者
 * 分叉，这里会跟着一起错，而不会出现「提示说不能、实际能做」这种更坏的情形。
 *
 * 【为什么需要它】快捷键被拒此前是**静默 no-op**：按了没反应、也没有任何说明。
 * tooltip 能解释菜单项，但键盘没有可以挂 tooltip 的表面。
 */
export function capabilityDeniedText(action: string, surface: string | null): string | null {
    if (pluginAllowsAction(action, surface)) return null;
    const id = capabilityForAction(action, surface);
    if (id === null) return null;
    const decision = capability(id);
    return decision.reason === null ? null : translateOutsideReact(decision.reason);
}

/**
 * 一条能力不可用时的**文案**（已按当前语言翻译）。
 *
 * 【为什么走 translateOutsideReact】调用点大多不在组件里（右键菜单项的 tooltip、
 * 命令式异常），与 dawControlledReason 同一取舍：无浏览器环境时回落英文词典。
 */
export function capabilityReasonText(id: CapabilityId): string {
    const decision = capability(id);
    if (decision.allowed || decision.reason === null) return "";
    return translateOutsideReact(decision.reason);
}

/**
 * 某个动作在某个界面上归哪条能力。
 *
 * 【为什么动作名不直接等于能力 id】动作名是快捷键 / 菜单的标识，能力是产品语义；
 * 一个能力可能对应多个动作（clip.copy / cut / paste 都是剪贴板）。把映射写在一处，
 * 才不会出现「菜单按能力判、快捷键按动作名判」这种两套判据。
 *
 * 返回 null 表示「这条动作不受宿主能力限制」。
 */
export function capabilityForAction(action: string, surface: string | null): CapabilityId | null {
    if (action === "host.handKeyboardToHost") return "host.keyboardHandoff";
    if (action === "playback.toggle" || action === "playback.stop") return "transport.control";
    if (action === "playback.metronome") return "playback.metronome";
    if (action === "project.importMedia") return "media.import";
    if (action === "project.importMidi") return "midi.importPitch";
    if (action === "project.export") return "export.audio";
    if (action === "clip.split") return "clip.split";
    if (action === "track.createHost") return "track.create";
    if (action === "clip.delete" || action === "edit.pasteTracks") return "clip.clipboard";
    if (action === "clip.normalize") return "clip.normalize";
    if (action === "clip.group" || action === "clip.ungroup") return "clip.edit";
    if (action.startsWith("transport.") || action.startsWith("recording.")) {
        return action.startsWith("recording.") ? "recording.local" : "transport.control";
    }
    if (action.startsWith("project.")) return "project.file";
    // 共享剪贴板键的别名不决定目标；换轨后 clip.paste 也可能粘贴参数线。
    // 必须先解析内容 / 选区，再按实际事件通道拒绝宿主几何操作。
    if (action === "clip.copy" || action === "clip.cut" || action === "clip.paste") return null;
    if (action.startsWith("track.")) {
        if (
            ["track.selectUp", "track.selectDown", "track.toggleMute", "track.toggleSolo"].includes(
                action,
            )
        )
            return null;
        return "track.edit";
    }
    if (action.startsWith("clip.") && surface !== "pianoRoll") return "clip.geometry";
    if (action.startsWith("edit.") && surface !== "pianoRoll") return "surface.pianoRollOnly";
    return null;
}

/**
 * 「这个动作为什么不能做」—— 给 tooltip / 提示用；可用或不受限时返回 null。
 *
 * 【为什么单独导出】UI 需要的往往只是这一句，而不是先问 allowed 再问 reason。
 */
export function capabilityDeniedReason(action: string, surface: string | null): MessageKey | null {
    if (!isPluginMode()) return null;
    const id = capabilityForAction(action, surface);
    if (id === null) return null;
    return capability(id).reason;
}

/**
 * 插件里这条动作是否放行。
 *
 * 【判据只有一个来源】此前这里是一张前缀表，与上面二十条谓词各写一遍，于是两者必然
 * 漂移 —— 代码里已经记过两次因此出的事故（MIDI 导入、导出：菜单能用、快捷键是死的）。
 * 现在动作名先经 capabilityForAction 映射到能力，再问能力矩阵；前缀表只剩「哪些动作
 * 归哪条能力」这一件事。
 *
 * 【为什么参数面（pianoRoll）的 edit.* 要单独一条白名单】那些是纯参数操作，只改插件
 * 自己持有的曲线（set_param_frames 一族），不写任何宿主几何 —— 用「edit.* 只在
 * pianoRoll 可用」一刀切会把它们一起拒掉，钢琴卷帘里的移调 / 平滑 / 平均 / 颤音 /
 * 量化就全灰了。
 */
const PIANO_ROLL_EDIT_ACTIONS = [
    "edit.undo",
    "edit.redo",
    "edit.selectAll",
    "edit.deselect",
    "edit.addClipsToParamSelection",
    "edit.removeClipsFromParamSelection",
    "edit.initialize",
    "edit.setPitch",
    "edit.transposeCents",
    "edit.transposeDegrees",
    "edit.average",
    "edit.smooth",
    "edit.addVibrato",
    "edit.quantize",
    "edit.meanQuantize",
];

export function pluginAllowsAction(action: string, surface: string | null): boolean {
    if (!isPluginMode()) return true;
    const id = capabilityForAction(action, surface);
    // 【为什么先查映射再看界面】edit.pasteTracks 虽然以 edit. 开头，但它是一条**宿主
    // 剪贴板**动作（映射到 clip.clipboard）。先按 edit. 前缀一刀切会把它当成「只在
    // 参数编辑器可用」而拒掉 —— 那正是这类两套判据最典型的失效方式。
    if (id === "surface.pianoRollOnly") return PIANO_ROLL_EDIT_ACTIONS.includes(action);
    if (id === null) return true;
    return capability(id).allowed;
}
/** 插件共享编辑快捷键只允许发给参数面板；不向宿主轨道派发几何写操作。 */
export function pluginAllowsEditChannel(channel: string | null, op?: string): channel is string {
    return (
        channel !== null &&
        (!isPluginMode() ||
            channel === "hifi:editOp" ||
            (channel === "hifi:timelineEditOp" && op === "split" && canSplitHostClips()) ||
            (channel === "hifi:timelineEditOp" &&
                ["copy", "cut", "paste", "pasteTracks", "delete"].includes(op ?? "") &&
                canClipboardHostClips()) ||
            (channel === "hifi:timelineEditOp" &&
                ["group", "ungroup"].includes(op ?? "") &&
                canEditHostClips()))
    );
}
