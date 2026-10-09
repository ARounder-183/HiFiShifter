// 插件共享粘贴按最终目标授权：换轨不禁参数粘贴，但仍禁止宿主片段写操作。
// @vitest-environment jsdom
import { afterEach, expect, test } from "vitest";
import {
    pluginAllowsAction,
    pluginAllowsEditChannel,
    isHostGeometryReadOnly,
    canEditHostClips,
    canImportHostAudio,
    canCreateHostTracks,
    canGroupPluginTracks,
    canImportMidiToPitch,
    canImportMidiAsClip,
    canEditHostFadeAxes,
    canEditFadeLength,
    canEditFadeShape,
    canEditFadeCurvature,
    canEditFadeS,
    canSelectHostFadeShape,
    canEditTempoMapTempo,
    hostFadeAxes,
    dawControlledReason,
} from "./hostCapabilities";
import { resolveActionByFocus, resolvePasteRoute } from "../features/keybindings/focusRouting";
import { DEFAULT_KEYBINDINGS } from "../features/keybindings/defaultKeybindings";

afterEach(() => {
    delete window.__HFS_PLUGIN_BOOTSTRAP__;
});

test("private track grouping needs its own capability and never opens host track creation", () => {
    window.__HFS_PLUGIN_BOOTSTRAP__ = { version: 1, viewId: "groups", clipEditing: true };
    expect(canGroupPluginTracks()).toBe(false);
    window.__HFS_PLUGIN_BOOTSTRAP__.trackGrouping = true;
    expect(canGroupPluginTracks()).toBe(true);
    expect(pluginAllowsAction("track.add", "trackHeader")).toBe(false);
});

test("Ctrl+V after a host track switch reaches parameter paste before channel authorization", () => {
    window.__HFS_PLUGIN_BOOTSTRAP__ = { version: 1, viewId: "paste" };
    for (const surface of ["pianoRoll", "trackHeader", "timeline", null] as const) {
        const action = resolveActionByFocus(
            new KeyboardEvent("keydown", { key: "v", ctrlKey: true }),
            DEFAULT_KEYBINDINGS,
            surface,
            "select",
        );
        expect(action).not.toBeNull();
        expect(pluginAllowsAction(action!, surface)).toBe(true);
        expect(pluginAllowsEditChannel(resolvePasteRoute("param", surface))).toBe(true);
        expect(pluginAllowsEditChannel(resolvePasteRoute("clips", surface))).toBe(false);
    }
});

test("standalone clipboard routes stay allowed; unrelated plugin geometry shortcuts stay blocked", () => {
    expect(pluginAllowsEditChannel("hifi:timelineEditOp")).toBe(true);
    expect(pluginAllowsEditChannel(null)).toBe(false);
    window.__HFS_PLUGIN_BOOTSTRAP__ = { version: 1, viewId: "paste" };
    expect(pluginAllowsAction("clip.delete", "timeline")).toBe(false);
    expect(pluginAllowsAction("track.add", "pianoRoll")).toBe(false);
    expect(pluginAllowsEditChannel("hifi:timelineEditOp")).toBe(false);
});

test("geometry editing requires explicit native write capability and standalone stays editable", () => {
    expect(isHostGeometryReadOnly()).toBe(false);
    window.__HFS_PLUGIN_BOOTSTRAP__ = { version: 1, viewId: "host-geometry" };
    expect(isHostGeometryReadOnly()).toBe(true);
    expect(canEditHostClips()).toBe(false);
    window.__HFS_PLUGIN_BOOTSTRAP__.clipEditing = true;
    expect(isHostGeometryReadOnly()).toBe(false);
    expect(canEditHostClips()).toBe(true);
});
test("native audio import opens only its media action, not project file/device commands", () => {
    window.__HFS_PLUGIN_BOOTSTRAP__ = { version: 1, viewId: "media" };
    expect(canImportHostAudio()).toBe(false);
    expect(pluginAllowsAction("project.importMedia", "timeline")).toBe(false);
    window.__HFS_PLUGIN_BOOTSTRAP__.audioImport = true;
    expect(canImportHostAudio()).toBe(true);
    expect(pluginAllowsAction("project.importMedia", "timeline")).toBe(true);
    expect(pluginAllowsAction("project.open", "timeline")).toBe(false);
    expect(pluginAllowsAction("project.new", "timeline")).toBe(false);
});

/**
 * 宿主建轨是一条**独立**能力，且只放行它自己那个动作名。
 *
 * 【为什么必须与导入分开】导入音频会顺带建轨，所以导入蕴含建轨；反过来不成立。
 * 合成一个标志会让"添加轨道"随"能不能导入音频"一起开关 —— 而原生侧门槛不同。
 */
test("host track creation is its own capability and does not open the rest of the track menu", () => {
    window.__HFS_PLUGIN_BOOTSTRAP__ = { version: 1, viewId: "tracks", audioImport: true };
    expect(canCreateHostTracks()).toBe(false);
    expect(pluginAllowsAction("track.createHost", "trackHeader")).toBe(false);
    window.__HFS_PLUGIN_BOOTSTRAP__.trackCreation = true;
    expect(canCreateHostTracks()).toBe(true);
    expect(pluginAllowsAction("track.createHost", "trackHeader")).toBe(true);
    // 其余 track.* 动作仍被拦下：建轨不等于能重排/删除宿主轨道。
    for (const action of ["track.add", "track.remove", "track.duplicate", "track.move"])
        expect(pluginAllowsAction(action, "trackHeader")).toBe(false);
});

test("native split opens only split action and its timeline channel", () => {
    window.__HFS_PLUGIN_BOOTSTRAP__ = { version: 1, viewId: "split", clipEditing: true };
    expect(pluginAllowsAction("clip.split", "timeline")).toBe(false);
    window.__HFS_PLUGIN_BOOTSTRAP__.clipSplitting = true;
    expect(pluginAllowsAction("clip.split", "timeline")).toBe(true);
    expect(pluginAllowsEditChannel("hifi:timelineEditOp", "split")).toBe(true);
    for (const op of ["delete", "cut", "paste", "glue", undefined])
        expect(pluginAllowsEditChannel("hifi:timelineEditOp", op)).toBe(false);
    expect(pluginAllowsAction("clip.delete", "timeline")).toBe(false);
});

test("native clipboard opens copy/cut/paste/delete only with its own host capability", () => {
    window.__HFS_PLUGIN_BOOTSTRAP__ = { version: 1, viewId: "clipboard", clipEditing: true };
    for (const op of ["copy", "cut", "paste", "pasteTracks", "delete"])
        expect(pluginAllowsEditChannel("hifi:timelineEditOp", op)).toBe(false);
    window.__HFS_PLUGIN_BOOTSTRAP__.clipClipboard = true;
    for (const op of ["copy", "cut", "paste", "pasteTracks", "delete"])
        expect(pluginAllowsEditChannel("hifi:timelineEditOp", op)).toBe(true);
    expect(pluginAllowsAction("clip.delete", "timeline")).toBe(true);
    expect(pluginAllowsAction("edit.pasteTracks", "timeline")).toBe(true);
    expect(pluginAllowsAction("clip.copy", "timeline")).toBe(true);
    expect(pluginAllowsAction("clip.cut", "timeline")).toBe(true);
    for (const op of ["glue", "cycleTake"])
        expect(pluginAllowsEditChannel("hifi:timelineEditOp", op)).toBe(false);
    expect(pluginAllowsAction("clip.group", "timeline")).toBe(true);
    expect(pluginAllowsAction("clip.ungroup", "timeline")).toBe(true);
});

test("MIDI import reaches the pitch curve in the plugin but never builds a local clip", () => {
    // 独立 App：两条路都在。
    expect(canImportMidiToPitch()).toBe(true);
    expect(canImportMidiAsClip()).toBe(true);
    expect(pluginAllowsAction("project.importMidi", "timeline")).toBe(true);
    // 插件：曲线是插件自己的权威（与参数编辑器同一份数据），片段归宿主 ——
    // 本地片段会在下一次宿主同步时消失，所以那一档必须关掉。
    window.__HFS_PLUGIN_BOOTSTRAP__ = { version: 1, viewId: "midi" };
    expect(canImportMidiToPitch()).toBe(true);
    expect(canImportMidiAsClip()).toBe(false);
    // 菜单项与快捷键必须走同一条判据，否则会出现"菜单能用、快捷键按了没反应"。
    expect(pluginAllowsAction("project.importMidi", "timeline")).toBe(true);
});

/**
 * 淡化轴能力分三档：版本读不出（`null`）→ 整块只读；旧轴 → 形状与曲率都写；
 * 新轴（≥7.81）→ 曲率与 S 直接写，形状预设由 Rust 侧翻成 `(curvature, S)` 一对分量。
 *
 * 【为什么新轴也放行预设】映射是实测的（`timeline/hostFadeAxes.ts`，证据
 * `probe/ara/FADE-AXIS-FINDINGS.md`），七个预设各自对应一组确定坐标。此前这里只放行
 * `legacy`，那条"映射尚未校准"的理由已经随实测消失。
 */
test("fade shape editing follows the host axis generation instead of one blanket mode", () => {
    window.__HFS_PLUGIN_BOOTSTRAP__ = { version: 1, viewId: "fade", clipEditing: true };
    expect(hostFadeAxes()).toBeNull();
    expect(canEditHostFadeAxes()).toBe(false);
    expect(canSelectHostFadeShape()).toBe(false);

    for (const axes of ["legacy", "continuous"] as const) {
        window.__HFS_PLUGIN_BOOTSTRAP__.fadeAxes = axes;
        expect(canEditHostFadeAxes()).toBe(true);
        expect(canSelectHostFadeShape()).toBe(true);
    }

    // 几何不可写时，轴语义再清楚也不能开闸 —— 走的是同一条 host_edit 路径。
    window.__HFS_PLUGIN_BOOTSTRAP__.clipEditing = false;
    expect(canEditHostFadeAxes()).toBe(false);

    // 独立 App 用自己的曲率轴，没有宿主版本这个概念。
    delete window.__HFS_PLUGIN_BOOTSTRAP__;
    expect(hostFadeAxes()).toBeNull();
    expect(canSelectHostFadeShape()).toBe(true);
});

/**
 * 淡变能力**按轴**拆开，而不是一个 `fadeShapeReadOnly` 布尔把四件事一起关掉。
 *
 * 【为什么长度必须与形状分开】长度落在 `D_FADE*LEN` / `*_AUTO`，与宿主轴版本无关。
 * 此前用一个布尔门住全部淡变编辑，于是宿主版本读不出来时用户**连长度都调不了** ——
 * 而那本来是能做的。
 *
 * 【为什么 S 轴只有新轴有】`D_FADE*DIR2_NEW` 是 REAPER ≥7.81 才有的键；legacy 宿主
 * 写它会被后端 `validate_fade_axes` 拒绝。
 */
test("fade capabilities split by axis: length always writable, S only on 7.81+", () => {
    window.__HFS_PLUGIN_BOOTSTRAP__ = { version: 1, viewId: "fade-split", clipEditing: true };
    // 轴版本读不出来：长度仍可写，形状/曲率/S 不可写。
    expect(hostFadeAxes()).toBeNull();
    expect(canEditFadeLength()).toBe(true);
    expect(canEditFadeShape()).toBe(false);
    expect(canEditFadeCurvature()).toBe(false);
    expect(canEditFadeS()).toBe(false);

    // legacy：形状与曲率可写，S 不可写。
    window.__HFS_PLUGIN_BOOTSTRAP__.fadeAxes = "legacy";
    expect(canEditFadeLength()).toBe(true);
    expect(canEditFadeShape()).toBe(true);
    expect(canEditFadeCurvature()).toBe(true);
    expect(canEditFadeS()).toBe(false);

    // continuous：三样都可写。
    window.__HFS_PLUGIN_BOOTSTRAP__.fadeAxes = "continuous";
    expect(canEditFadeShape()).toBe(true);
    expect(canEditFadeCurvature()).toBe(true);
    expect(canEditFadeS()).toBe(true);

    // 几何不可写时，长度也一起关掉（同一条 host_edit 路径）。
    window.__HFS_PLUGIN_BOOTSTRAP__.clipEditing = false;
    expect(canEditFadeLength()).toBe(false);
    expect(canEditFadeShape()).toBe(false);

    // 独立 App：自己的形状轴，全部可写。
    delete window.__HFS_PLUGIN_BOOTSTRAP__;
    expect(canEditFadeLength()).toBe(true);
    expect(canEditFadeShape()).toBe(true);
    expect(canEditFadeCurvature()).toBe(true);
    expect(canEditFadeS()).toBe(true);
});

/**
 * 速度映射**按轴**切分，不按功能整组切。
 *
 * 【为什么音阶轴在插件里必须可用】Tempo Map 是"随时间变化的音阶"的存储
 * （`TempoPointData.scale`），音阶又是 HiFiShifter 自有的（REAPER 没有工程调号概念）。
 * 整组按模式隐藏 = 砍掉音阶功能的一半。
 *
 * 【为什么 BPM/拍号轴仍不可编辑】两者是宿主权威（VST3 进程上下文只读）。
 */
test("tempo map splits by axis: BPM/time signature stay host-owned in the plugin", () => {
    window.__HFS_PLUGIN_BOOTSTRAP__ = { version: 1, viewId: "tempo", clipEditing: true };
    expect(canEditTempoMapTempo()).toBe(false);
    // 与淡化轴能力正交：宿主轴再清楚也不放行 BPM/拍号。
    window.__HFS_PLUGIN_BOOTSTRAP__.fadeAxes = "legacy";
    expect(canEditHostFadeAxes()).toBe(true);
    expect(canEditTempoMapTempo()).toBe(false);

    delete window.__HFS_PLUGIN_BOOTSTRAP__;
    expect(canEditTempoMapTempo()).toBe(true);
});

test("dawControlledReason follows the current locale instead of freezing at module load", () => {
    const stored = localStorage.getItem("hifishifter.locale");
    try {
        localStorage.setItem("hifishifter.locale", "en-US");
        const en = dawControlledReason();
        localStorage.setItem("hifishifter.locale", "zh-CN");
        const zh = dawControlledReason();
        // 查不到键时 translateOutsideReact 原样返回键名 —— 那种静默失败必须被拦下。
        expect(en).not.toBe("plugin_daw_controlled_reason");
        expect(zh).not.toBe("plugin_daw_controlled_reason");
        expect(zh).not.toBe(en);
    } finally {
        if (stored === null) localStorage.removeItem("hifishifter.locale");
        else localStorage.setItem("hifishifter.locale", stored);
    }
});
