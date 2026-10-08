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
