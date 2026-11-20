/*
 * 片段右键菜单：插件形态下**没有第二行说明**，做不到的项**直接不出现**。
 *
 * 【要钉死什么】此前插件走的是「一棵树 + 逐项禁用 + 第二行原因」：Take 结构编辑、
 * 胶合、静音检测、假立体声扫描在 REAPER 里永远做不到，却各占两行菜单，把真正能用
 * 的项淹掉 —— 而那句原因对用户**不可操作**（换不了设置、换不了宿主）。
 *
 * 现在：结构性不可用 ⇒ 不渲染；情境性不可用 ⇒ 只 disabled。同时钉住两处**门禁写错**
 * 的修复：音高参考与快速导出在后端插件里早已可用，此前却被 !pluginMode 一并关掉。
 */
// @vitest-environment jsdom
import { act, type ComponentProps } from "react";
import { createRoot, type Root } from "react-dom/client";
import { Provider } from "react-redux";
import { combineReducers, configureStore } from "@reduxjs/toolkit";
import { afterEach, beforeEach, expect, test } from "vitest";

import sessionReducer from "../../../features/session/sessionSlice";
import dockReducer from "../../../features/dock/dockSlice";
import keybindingsReducer from "../../../features/keybindings/keybindingsSlice";
import { enUS } from "../../../i18n/en-US";
import { I18nProvider } from "../../../i18n/I18nProvider";
import { writableHostCaps, type HostCaps } from "../../../services/pluginBootstrapFixtures";
import { AppThemeProvider } from "../../../theme/AppThemeProvider";
import type { ClipInfo } from "../../../features/session/sessionTypes";
import { ClipContextMenu } from "./ClipContextMenu";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let container: HTMLDivElement;
let root: Root;

function buildStore() {
    return configureStore({
        reducer: combineReducers({
            session: sessionReducer,
            dock: dockReducer,
            keybindings: keybindingsReducer,
        }),
        middleware: (getDefault) => getDefault({ serializableCheck: false, immutableCheck: false }),
    });
}

/** 两个 take 的音频片段：take 子菜单与逐 take 行都需要它。 */
function clipWithTakes(): ClipInfo {
    return {
        id: "clip-1",
        trackId: "track-1",
        name: "vocal",
        startSec: 0,
        lengthSec: 4,
        color: "blue",
        takes: [
            {
                id: "t1",
                name: "Take A",
                gain: 1,
                reversed: false,
                channelMode: 0,
                loopEnabled: false,
                sourcePath: "C:\\audio\\vocal.wav",
            },
            {
                id: "t2",
                name: "Take B",
                gain: 1,
                reversed: false,
                channelMode: 0,
                loopEnabled: false,
                sourcePath: "C:\\audio\\vocal.wav",
            },
        ],
        activeTakeId: "t1",
        sourcePath: "C:\\audio\\vocal.wav",
        gain: 1,
        muted: false,
        sourceStartSec: 0,
        sourceEndSec: 4,
        playbackRate: 1,
        reversed: false,
        channelMode: 0,
        loopEnabled: false,
        snapOffsetSec: 0,
        fadeInSec: 0.1,
        fadeOutSec: 0.1,
        fadeInShape: 0,
        fadeOutShape: 0,
    } as ClipInfo;
}

type MenuProps = ComponentProps<typeof ClipContextMenu>;

/** 把菜单挂到一棵全新的根上（每个用例都要干净的 DOM）。 */
async function mountMenu(overrides: Partial<MenuProps> = {}) {
    const clip = clipWithTakes();
    await act(async () => {
        root.render(
            <Provider store={buildStore()}>
                <AppThemeProvider>
                    <I18nProvider>
                        <ClipContextMenu
                            x={10}
                            y={10}
                            clip={clip}
                            selectedClips={[clip]}
                            playheadInClip={true}
                            canSplitSelected={false}
                            onClose={() => undefined}
                            onDelete={() => undefined}
                            onMute={() => undefined}
                            onCopy={() => undefined}
                            onCut={() => undefined}
                            onReplace={() => undefined}
                            onQuickExport={() => undefined}
                            onSplit={() => undefined}
                            onGlue={() => undefined}
                            onNormalize={() => undefined}
                            onToggleReverse={() => undefined}
                            onToggleLoop={() => undefined}
                            onSetChannelMode={() => undefined}
                            onFadeShapeChange={() => undefined}
                            onConvertToPitchRef={() => undefined}
                            onUpdatePitchRef={() => undefined}
                            onImportHostPitch={() => undefined}
                            onExportHostPitch={() => undefined}
                            {...overrides}
                        />
                    </I18nProvider>
                </AppThemeProvider>
            </Provider>,
        );
    });
}

function pluginCaps(overrides: Partial<HostCaps> = {}): HostCaps {
    return writableHostCaps({ split: true, itemClipboard: true, ...overrides });
}

function setBootstrap(caps: HostCaps, hostKind: "reaper" | "ara_only" = "reaper") {
    window.__HFS_PLUGIN_BOOTSTRAP__ = {
        version: 1,
        viewId: "menu",
        hostKind,
        hostCapabilities: caps,
    };
}

/** 顶层菜单项文案（含子菜单触发项，不含未展开的子面板内容）。 */
function menuLabels(): string[] {
    return Array.from(document.body.querySelectorAll(".hs-menu__item")).map(
        (el) => el.textContent ?? "",
    );
}

function entryByLabel(label: string): HTMLButtonElement | undefined {
    return Array.from(
        document.body.querySelectorAll<HTMLButtonElement>("button.hs-menu__item"),
    ).find((el) => (el.textContent ?? "").includes(label));
}

beforeEach(() => {
    localStorage.setItem("hifishifter.locale", "en-US");
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
});

afterEach(() => {
    act(() => root.unmount());
    container.remove();
    document.body.innerHTML = "";
    localStorage.removeItem("hifishifter.locale");
    delete window.__HFS_PLUGIN_BOOTSTRAP__;
});

test("no menu entry ever renders a second line in the plug-in", async () => {
    setBootstrap(pluginCaps());
    await mountMenu();
    expect(document.body.querySelectorAll(".hs-menu__description").length).toBe(0);
});

test("structurally impossible entries are absent instead of disabled", async () => {
    setBootstrap(pluginCaps());
    await mountMenu();

    const labels = menuLabels().join("\n");
    // 胶合、静音检测、假立体声扫描在 REAPER 插件里都做不到 ⇒ 不出现。
    for (const label of [enUS.common_glue, enUS.ctx_silence_detection, enUS.ctx_scan_fake_stereo]) {
        expect(labels, label).not.toContain(label);
    }
    // Take 子菜单仍然存在（列表是有效信息），但**写操作**只在展开后才有内容。
    const trigger = entryByLabel(enUS.clip_takes);
    expect(trigger).toBeTruthy();
    await act(async () => {
        trigger!.click();
    });
    const openLabels = menuLabels().join("\n");
    for (const label of [
        enUS.clip_pack_into_takes,
        enUS.clip_take_add,
        enUS.clip_take_duplicate,
        enUS.clip_take_remove,
        enUS.clip_take_explode,
        enUS.ctx_replace,
    ]) {
        expect(openLabels, label).not.toContain(label);
    }
    // 能做的两件仍在：只读 take 名（说明行，不是可点项）与重命名。
    expect(openLabels).toContain(enUS.clip_take_rename);
    const panelText = Array.from(document.body.querySelectorAll(".hs-menu__label"))
        .map((el) => el.textContent ?? "")
        .join("\n");
    expect(panelText).toContain("Take A");
    expect(panelText).toContain("Take B");
});

test("reverse becomes usable once the host can write the source section", async () => {
    setBootstrap(pluginCaps({ clipReverse: true }));
    await mountMenu();
    const reverse = entryByLabel(enUS.ctx_reverse);
    expect(reverse, "倒放在宿主支持结构写回时必须出现").toBeTruthy();
    expect(reverse!.disabled).toBe(false);
});

test("reverse stays hidden (not disabled) when the host cannot write it", async () => {
    setBootstrap(pluginCaps({ clipReverse: false }));
    await mountMenu();
    expect(entryByLabel(enUS.ctx_reverse)).toBeUndefined();
});

test("pitch reference and quick export are no longer gated by plugin mode", async () => {
    setBootstrap(pluginCaps());
    await mountMenu();
    // 后端插件里这两条命令都已实现（convert_clips_to_pitch_reference /
    // quick_export_selected_clips），门禁此前按 !pluginMode 一刀切关掉。
    expect(entryByLabel(enUS.ctx_convert_to_pitch_ref)?.disabled).toBe(false);
    expect(entryByLabel(enUS.ctx_quick_export)?.disabled).toBe(false);
});

test("the REAPER pitch bridge only appears on REAPER with the envelope API", async () => {
    setBootstrap(pluginCaps({ takePitchBridge: true }), "reaper");
    await mountMenu();
    expect(entryByLabel(enUS.ctx_host_pitch)).toBeTruthy();

    act(() => root.unmount());
    document.body.innerHTML = "";
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
    setBootstrap(pluginCaps({ takePitchBridge: true }), "ara_only");
    await mountMenu();
    expect(entryByLabel(enUS.ctx_host_pitch)).toBeUndefined();
});

test("loop and channel mode stay usable: they are plain host attributes", async () => {
    setBootstrap(pluginCaps());
    await mountMenu();
    expect(entryByLabel(enUS.ctx_loop)?.disabled).toBe(false);
    expect(entryByLabel(enUS.ctx_channel_mode)).toBeTruthy();
});

test("split is contextually disabled and carries no explanation", async () => {
    setBootstrap(pluginCaps());
    await mountMenu({ playheadInClip: false });
    const split = entryByLabel(enUS.ctx_split_at_playhead);
    expect(split?.disabled).toBe(true);
    expect(split?.querySelector(".hs-menu__description")).toBeNull();
});

/*
 * 独立 App 必须看到**完整**的菜单。
 *
 * 【为什么钉这一条】能力位重构时，渲染端把 `canEditHostClips` 一族（插件形态的
 * "宿主写能力"谓词，独立 App 恒为 false）直接当成了菜单判据 —— 独立 App 的
 * 删除/复制/剪切/静音/声道模式/循环/分割/编组 整段消失，而后端命令全部健在。
 * 独立 App 无 bootstrap ⇒ 所有宿主能力谓词都 false ⇒ 只有"独立 App 恒真"的
 * 那些谓词（倒放 / 音高参考 / 快速导出 / 淡变形状）幸存。
 */
test("standalone keeps the full tree: delete/mute/channel mode/loop/split/group/copy", async () => {
    await mountMenu({ canSplitSelected: true });
    for (const label of [
        enUS.ctx_delete,
        enUS.clip_mute,
        enUS.ctx_channel_mode,
        enUS.ctx_loop,
        enUS.ctx_split_at_playhead,
        enUS.ctx_copy,
        enUS.ctx_cut,
        enUS.ctx_normalize,
    ]) {
        const entry = entryByLabel(label);
        expect(entry, label).toBeTruthy();
        expect(entry!.disabled, label).toBe(false);
    }
    // 声道模式子菜单展开后五档模式齐全。
    const channel = entryByLabel(enUS.ctx_channel_mode);
    await act(async () => {
        channel!.click();
    });
    expect(menuLabels().join("\n")).toContain(enUS.clip_channel_mode_normal);

    // take 子菜单在独立 App 里是可操作的 take 行（倒放 / 声道按钮在行尾）。
    act(() => root.unmount());
    document.body.innerHTML = "";
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
    await mountMenu();
    const takes = entryByLabel(enUS.clip_takes);
    await act(async () => {
        takes!.click();
    });
    const openLabels = menuLabels().join("\n");
    for (const label of [
        enUS.clip_take_add,
        enUS.clip_take_duplicate,
        enUS.clip_take_remove,
        enUS.clip_take_explode,
        enUS.ctx_replace,
    ]) {
        expect(openLabels, label).toContain(label);
    }
});
