// 原GUI的运行模式与能力边界；独立app保持原窗口/文件/设备行为，插件由DAW供源。
import { translateOutsideReact } from "../i18n/I18nProvider";

export type HostMode = "standalone" | "plugin";
/** 必须使用显式bootstrap，而不是把所有WebView2都误认为插件。 */
export function hostMode(): HostMode {
    return typeof window !== "undefined" && window.__HFS_PLUGIN_BOOTSTRAP__?.version === 1
        ? "plugin"
        : "standalone";
}
export function isPluginMode(): boolean {
    return hostMode() === "plugin";
}
/** 仅宿主原生明确提供ARA播放请求能力时开放，不把插件模式等同支持播放控制。 */
export function canControlHostTransport(): boolean {
    return isPluginMode() && window.__HFS_PLUGIN_BOOTSTRAP__?.transportControl === true;
}
/** 只有本次原生入口明确具备REAPER写API，才打开片段拖拽/裁切/线性拉伸。 */
export function canEditHostClips(): boolean {
    return isPluginMode() && window.__HFS_PLUGIN_BOOTSTRAP__?.clipEditing === true;
}
/** 分割有独立原生能力门，不因允许拖拽就放开未实现的片段命令。 */
export function canSplitHostClips(): boolean {
    return isPluginMode() && window.__HFS_PLUGIN_BOOTSTRAP__?.clipSplitting === true;
}
/** 原生完整item剪贴板及创建/删除接口已接通时开放，旧插件不会误发App私有媒体命令。 */
export function canClipboardHostClips(): boolean {
    return isPluginMode() && window.__HFS_PLUGIN_BOOTSTRAP__?.clipClipboard === true;
}
/** 只允许插件私有参数分组，不代表能重排或创建REAPER的轨道/folder。 */
export function canGroupPluginTracks(): boolean {
    return isPluginMode() && window.__HFS_PLUGIN_BOOTSTRAP__?.trackGrouping === true;
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
 * 与形状同门槛：两者都要宿主版本分得清轴语义（`fade_axes_new`），否则不知道曲率该写
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
    return window.__HFS_PLUGIN_BOOTSTRAP__?.fadeAxes ?? null;
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
    return isPluginMode() && window.__HFS_PLUGIN_BOOTSTRAP__?.audioImport === true;
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
    return isPluginMode() && window.__HFS_PLUGIN_BOOTSTRAP__?.trackCreation === true;
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

/** 原编辑工具与查看操作保留；不将DAW几何操作发到独立app命令路径。 */
export function pluginAllowsAction(action: string, surface: string | null): boolean {
    if (!isPluginMode()) return true;
    if (["playback.toggle", "playback.stop"].includes(action)) return canControlHostTransport();
    if (action === "project.importMedia") return canImportHostAudio();
    // MIDI 导入的**快捷键**必须与菜单同一条判据：菜单项已经可用（导入到音高曲线），
    // 而下面那条 `project.` 前缀规则会把它一并拒掉 —— 结果就是"菜单能用、快捷键
    // 按了没反应"，正是本仓明令禁止的静默失效。
    if (action === "project.importMidi") return canImportMidiToPitch();
    if (action === "clip.split") return canSplitHostClips();
    // 新建**宿主**轨道：与 `track.*` 的其余动作不同，它明确要写宿主工程，且已有
    // 原生实现（`create_host_track`）。用独立动作名，免得被下面那条 `track.` 规则
    // 一刀切掉。
    if (action === "track.createHost") return canCreateHostTracks();
    if (action === "clip.delete" || action === "edit.pasteTracks") return canClipboardHostClips();
    // 归一化只改HFS音频处理参数，不改REAPER item几何，可在插件中继续使用。
    if (action === "clip.normalize") return true;
    if (["clip.group", "clip.ungroup"].includes(action)) return canEditHostClips();
    if (
        action.startsWith("project.") ||
        action.startsWith("transport.") ||
        action.startsWith("recording.") ||
        ["playback.toggle", "playback.stop", "playback.metronome"].includes(action)
    )
        return false;
    if (
        action.startsWith("track.") &&
        !["track.selectUp", "track.selectDown", "track.toggleMute", "track.toggleSolo"].includes(
            action,
        )
    )
        return false;
    // 共享剪贴板键的别名不决定目标；换轨后clip.paste也可能粘贴参数线。
    // 必须先解析内容/选区，再按实际事件通道拒绝宿主几何操作。
    // 共享快捷键先按剪贴板内容路由；参数剪贴板在没有clipClipboard时仍必须可用。
    if (["clip.copy", "clip.cut", "clip.paste"].includes(action)) return true;
    if (action.startsWith("clip.") && surface !== "pianoRoll") return false;
    if (action.startsWith("edit.") && surface !== "pianoRoll") {
        return [
            "edit.undo",
            "edit.redo",
            "edit.selectAll",
            "edit.deselect",
            "edit.addClipsToParamSelection",
            "edit.removeClipsFromParamSelection",
        ].includes(action);
    }
    return true;
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
