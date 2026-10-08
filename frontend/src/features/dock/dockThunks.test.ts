/**
 * 停靠布局的**分形态存储**。
 *
 * 【要钉死什么】ARA 窗口下限 640×400，独立 App 是整屏。两者此前共用
 * `UiSettings.dock` 一个字段，于是互相覆盖对方的排布：在插件里折叠面板、把分隔条
 * 拖到极端比例之后，下次在 2560×1440 上开 App 恢复的就是那份小视口布局。
 *
 * 这里钉住的是"读哪个字段、写哪个字段"—— 它是这个 bug 的唯一防线，
 * 因为两个字段同形，写错名字不会有任何编译或类型错误。
 */
// @vitest-environment jsdom
import { configureStore } from "@reduxjs/toolkit";
import { afterEach, beforeEach, expect, test, vi } from "vitest";

const api = vi.hoisted(() => ({
    getUiSettings: vi.fn(),
    saveUiSettings: vi.fn(),
}));

vi.mock("../../services/api/settings", () => ({ settingsApi: api }));

import dockReducer, { hydrateDock } from "./dockSlice";
import { PLUGIN_DEFAULT_TRACK_HEADER_PX } from "./dockSchema";
import { loadDockSettings, persistDockSettings } from "./dockThunks";

function buildStore() {
    return configureStore({
        reducer: { dock: dockReducer },
        middleware: (getDefault) => getDefault({ serializableCheck: false, immutableCheck: false }),
    });
}

beforeEach(() => {
    api.getUiSettings.mockReset();
    api.saveUiSettings.mockReset();
    api.saveUiSettings.mockResolvedValue({ ok: true });
});

afterEach(() => {
    delete window.__HFS_PLUGIN_BOOTSTRAP__;
});

/** 插件形态：只有显式 bootstrap 才算（与 `isPluginMode()` 同判据）。 */
function enterPluginMode() {
    window.__HFS_PLUGIN_BOOTSTRAP__ = { version: 1, viewId: "dock" };
}

test("standalone reads and writes the shared dock field", async () => {
    const store = buildStore();
    api.getUiSettings.mockResolvedValue({ dock: { layout: { schema: 2 }, startupLayout: "last" } });

    await store.dispatch(loadDockSettings());
    expect(store.getState().dock.layout.schema).toBe(2);

    await store.dispatch(persistDockSettings());
    const patch = api.saveUiSettings.mock.calls[0][0];
    expect(Object.keys(patch)).toEqual(["dock"]);
    expect(patch.dock.layout).toBeDefined();
});

test("plugin reads and writes its own dock field", async () => {
    enterPluginMode();
    const store = buildStore();
    // 两个字段都在，值不同：必须读到插件那一份。
    api.getUiSettings.mockResolvedValue({
        dock: { layout: { schema: 2, tabPosition: "top" } },
        dockPlugin: { layout: { schema: 2, tabPosition: "bottom" } },
    });

    await store.dispatch(loadDockSettings());
    expect(store.getState().dock.layout.tabPosition).toBe("bottom");

    await store.dispatch(persistDockSettings());
    const patch = api.saveUiSettings.mock.calls[0][0];
    expect(Object.keys(patch)).toEqual(["dockPlugin"]);
});

test("a plugin with no stored layout yet falls back to the defaults", async () => {
    enterPluginMode();
    const store = buildStore();
    // 后端只有 App 那份：插件这一份还没建立（迁移前的老配置）。
    api.getUiSettings.mockResolvedValue({ dock: { layout: { schema: 2 } } });

    await store.dispatch(loadDockSettings());
    // 不借用 App 的布局 —— 它可能是为大窗口排的，在 640×400 里不能用。
    // 归一化会补出默认布局（`normalizeDockLayout` 的职责），这里只断言没有套用。
    expect(store.getState().dock.layout.tabPosition).not.toBe("__from_app__");
});

test("the two modes never write into each other's field", async () => {
    const store = buildStore();

    await store.dispatch(persistDockSettings());
    expect(Object.keys(api.saveUiSettings.mock.calls[0][0])).toEqual(["dock"]);

    enterPluginMode();
    await store.dispatch(persistDockSettings());
    expect(Object.keys(api.saveUiSettings.mock.calls[1][0])).toEqual(["dockPlugin"]);
});

/**
 * 插件首次打开时不能套用 App 的默认轨道头宽度。
 *
 * 【为什么这条值得单独测】256px 占 640px 窗口的 40%，时间轴几乎看不见 ——
 * 用户第一件事必然是把它拖小。这是"第一次打开就不好用"，而不是"挤一点"。
 */
test("a plugin with no stored layout starts with a narrower track header", async () => {
    enterPluginMode();
    const store = buildStore();
    api.getUiSettings.mockResolvedValue({});

    const payload = await store.dispatch(loadDockSettings()).unwrap();
    expect(payload.pluginFirstRun).toBe(true);
    store.dispatch(hydrateDock(payload));
    expect(store.getState().dock.layout.gutters.timelineTrackHeaderPx).toBe(
        PLUGIN_DEFAULT_TRACK_HEADER_PX,
    );
});

test("a plugin that already has a layout keeps the stored gutter", async () => {
    enterPluginMode();
    const store = buildStore();
    api.getUiSettings.mockResolvedValue({
        dockPlugin: { layout: { schema: 2, gutters: { timelineTrackHeaderPx: 300 } } },
    });

    const payload = await store.dispatch(loadDockSettings()).unwrap();
    expect(payload.pluginFirstRun).toBe(false);
    store.dispatch(hydrateDock(payload));
    expect(store.getState().dock.layout.gutters.timelineTrackHeaderPx).toBe(300);
});

test("the standalone default is untouched", async () => {
    const store = buildStore();
    api.getUiSettings.mockResolvedValue({});
    const payload = await store.dispatch(loadDockSettings()).unwrap();
    expect(payload.pluginFirstRun).toBe(false);
    store.dispatch(hydrateDock(payload));
    expect(store.getState().dock.layout.gutters.timelineTrackHeaderPx).toBe(256);
});
