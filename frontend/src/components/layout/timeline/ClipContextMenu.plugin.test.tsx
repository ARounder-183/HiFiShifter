/*
 * 片段右键菜单：插件里**一棵树**，做不到的项禁用并说明原因。
 *
 * 【要钉死什么】此前插件走的是另一棵手抄的 10 项子树，于是 Take 子菜单 / 声道模式 /
 * 循环 / 淡变形状行在插件里**根本不存在** —— 用户报障"右键菜单缺很多入口"。
 * 现在合并成一棵树：这些入口必须**出现**；插件里做不到的项带 `title` 说明原因，
 * 而不是消失。
 */
// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { Provider } from "react-redux";
import { combineReducers, configureStore } from "@reduxjs/toolkit";
import { afterEach, beforeEach, expect, test } from "vitest";

import sessionReducer from "../../../features/session/sessionSlice";
import dockReducer from "../../../features/dock/dockSlice";
import keybindingsReducer from "../../../features/keybindings/keybindingsSlice";
import { enUS } from "../../../i18n/en-US";
import { I18nProvider } from "../../../i18n/I18nProvider";
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
            { id: "t1", name: "Take A", gain: 1, reversed: false, channelMode: 0, loopEnabled: false },
            { id: "t2", name: "Take B", gain: 1, reversed: false, channelMode: 0, loopEnabled: false },
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

async function renderMenu(clip: ClipInfo) {
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
                        />
                    </I18nProvider>
                </AppThemeProvider>
            </Provider>,
        );
    });
}

function menuLabels(): string[] {
    return Array.from(document.body.querySelectorAll(".hs-menu__item")).map(
        (el) => el.textContent ?? "",
    );
}

function entryByLabel(label: string): HTMLButtonElement {
    const found = Array.from(document.body.querySelectorAll<HTMLButtonElement>("button.hs-menu__item")).find(
        (el) => (el.textContent ?? "").includes(label),
    );
    if (!found) throw new Error(`menu entry not found: ${label}`);
    return found;
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

test("plugin mode still shows the take, channel-mode and loop entries", async () => {
    window.__HFS_PLUGIN_BOOTSTRAP__ = { version: 1, viewId: "menu", clipEditing: true };
    await renderMenu(clipWithTakes());

    const labels = menuLabels().join("\n");
    expect(labels).toContain(enUS.clip_takes);
    expect(labels).toContain(enUS.ctx_channel_mode);
    expect(labels).toContain(enUS.ctx_loop);
    expect(labels).toContain(enUS.ctx_reverse);
});

test("entries the plugin cannot do are disabled with a reason, not hidden", async () => {
    window.__HFS_PLUGIN_BOOTSTRAP__ = { version: 1, viewId: "menu", clipEditing: true };
    await renderMenu(clipWithTakes());

    // 倒放：ARA 不给反向 PCM —— 有专有原因，不是"由 REAPER 控制"。
    const reverse = entryByLabel(enUS.ctx_reverse);
    expect(reverse.disabled).toBe(true);
    expect(reverse.dataset.tooltip).toBe(enUS.plugin_reverse_unavailable);

    // Take 子菜单的**内容**要悬停展开才渲染（`AppSubMenu` 懒渲染），所以这里只断言
    // 触发项存在 —— 它的存在本身正是本次修复的重点（此前插件里整块没有）。
    expect(entryByLabel(enUS.clip_takes)).not.toBeNull();
});

test("loop stays usable in the plugin: it is a plain host item attribute", async () => {
    window.__HFS_PLUGIN_BOOTSTRAP__ = { version: 1, viewId: "menu", clipEditing: true };
    await renderMenu(clipWithTakes());
    expect(entryByLabel(enUS.ctx_loop).disabled).toBe(false);
});
