/*
 * 面板配置通道（`form.props`）与组件装配的契约测试。
 *
 * 【为什么必须有】`DockPanelProps.props` 一直被注释宣称为「未来 API 面板可直接
 * 使用」、`normalizeDockLayout` 也早已持久化它、宿主也早已传给面板组件 ——
 * 但在这之前**没有任何 action 能写入**。也就是说这条存储通道只有读的一半，
 * 面板拿到的永远是空对象。这个测试锁定写的一半，以及"装配组件不破坏元数据"。
 */
import { configureStore } from "@reduxjs/toolkit";
import { describe, expect, test, vi } from "vitest";

import dockReducer from "./dockSlice";
import { setPanelProps, setFormPropsById } from "./dockApi";
import type { GetState } from "./dockApi";
import {
    getPanel,
    registerPanel,
    resetPanelRegistryForTests,
    setPanelComponent,
} from "./panelRegistry";

function makeStore() {
    return configureStore({
        reducer: { dock: dockReducer },
        middleware: (getDefault) => getDefault({ serializableCheck: false }),
    });
}

describe("面板配置通道", () => {
    test("setFormProps 浅合并，未提及的键保持不变", () => {
        const store = makeStore();
        const formId = Object.keys(store.getState().dock.layout.forms)[0];
        expect(formId).toBeTruthy();

        store.dispatch({
            type: "dock/setFormProps",
            payload: { formId, props: { columns: 3, filter: "audio" } },
        });
        store.dispatch({
            type: "dock/setFormProps",
            payload: { formId, props: { columns: 5 } },
        });

        const form = store.getState().dock.layout.forms[formId];
        expect(form.props).toEqual({ columns: 5, filter: "audio" });
    });

    test("显式传 undefined 会删除该键（而不是留一个 undefined）", () => {
        const store = makeStore();
        const formId = Object.keys(store.getState().dock.layout.forms)[0];

        store.dispatch({
            type: "dock/setFormProps",
            payload: { formId, props: { columns: 3, filter: "audio" } },
        });
        store.dispatch({
            type: "dock/setFormProps",
            payload: { formId, props: { filter: undefined } },
        });

        const form = store.getState().dock.layout.forms[formId];
        expect(form.props).toEqual({ columns: 3 });
        expect("filter" in (form.props ?? {})).toBe(false);
    });

    test("formId 不存在时是空操作，不抛错", () => {
        const store = makeStore();
        expect(() =>
            store.dispatch({
                type: "dock/setFormProps",
                payload: { formId: "does-not-exist", props: { a: 1 } },
            }),
        ).not.toThrow();
    });

    test("dockApi 的 setPanelProps 按面板 id 定位窗体", () => {
        const store = makeStore();
        const state = store.getState();
        const form = Object.values(state.dock.layout.forms)[0];
        const panelId = form.panelId;

        // 测试 store 只挂了 dock 分片；facade 只读 state.dock，收窄是安全的。
        setPanelProps(store.dispatch, store.getState as unknown as GetState, panelId, {
            open: true,
        });

        const updated = Object.values(store.getState().dock.layout.forms).find(
            (candidate) => candidate.panelId === panelId,
        );
        expect(updated?.props).toEqual({ open: true });
    });

    test("setFormPropsById 直接按窗体 id 写入", () => {
        const store = makeStore();
        const formId = Object.keys(store.getState().dock.layout.forms)[0];
        setFormPropsById(store.dispatch, formId, { x: 1 });
        expect(store.getState().dock.layout.forms[formId].props).toEqual({ x: 1 });
    });
});

describe("面板组件装配", () => {
    test("setPanelComponent 只补 component，元数据保持不变", () => {
        resetPanelRegistryForTests();
        registerPanel({
            id: "test.panel",
            titleKey: "panel_timeline",
            defaultWidth: 300,
            defaultHeight: 200,
            detachable: true,
            order: 7,
        });

        const before = getPanel("test.panel")!;
        const Dummy = () => null;
        setPanelComponent("test.panel", Dummy);
        const after = getPanel("test.panel")!;

        expect(after.component).toBe(Dummy);
        // 元数据逐字段不变
        expect(after.id).toBe(before.id);
        expect(after.titleKey).toBe(before.titleKey);
        expect(after.defaultWidth).toBe(before.defaultWidth);
        expect(after.defaultHeight).toBe(before.defaultHeight);
        expect(after.detachable).toBe(before.detachable);
        expect(after.order).toBe(before.order);
    });

    test("对未注册的面板装配组件会告警但不抛错", () => {
        resetPanelRegistryForTests();
        const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
        expect(() => setPanelComponent("nope", () => null)).not.toThrow();
        expect(warn).toHaveBeenCalled();
        warn.mockRestore();
    });

    test("装配会推进注册表版本号（订阅者据此重渲染）", async () => {
        resetPanelRegistryForTests();
        registerPanel({
            id: "test.panel",
            titleKey: "panel_timeline",
            defaultWidth: 300,
            defaultHeight: 200,
        });
        const { getPanelRegistryVersion } = await import("./panelRegistry");
        const before = getPanelRegistryVersion();
        setPanelComponent("test.panel", () => null);
        expect(getPanelRegistryVersion()).toBeGreaterThan(before);
    });
});
