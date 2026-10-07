// App与插件共用折叠按钮：真实DOM点击改变布局，不改参数会话，也不丢面板挂载记录。
// @vitest-environment jsdom
import { configureStore } from "@reduxjs/toolkit";
import { act } from "react";
import { createRoot } from "react-dom/client";
import { Provider } from "react-redux";
import { afterEach, expect, test } from "vitest";
import dockReducer, { markFormsMounted, syncRegisteredPanels, openPanel, setDockLayout, setGutterSize } from "../../features/dock/dockSlice";
import sessionReducer from "../../features/session/sessionSlice";
import { isPanelVisible } from "../../features/dock/dockApi";
import { registerBuiltinPanels, PANEL_TIMELINE, PANEL_PARAM_EDITOR, PANEL_FILE_BROWSER } from "../dock/registerBuiltinPanels";
import { TimelinePanelToggle } from "./TimelinePanelToggle";

(globalThis as {IS_REACT_ACT_ENVIRONMENT?:boolean}).IS_REACT_ACT_ENVIRONMENT=true;
afterEach(()=>{delete window.__HFS_PLUGIN_BOOTSTRAP__;});
test("App and plugin collapse timeline, preserve session and mount records, then reopen",async()=>{
    registerBuiltinPanels();
    for (const plugin of [false,true]) {
        if (plugin) window.__HFS_PLUGIN_BOOTSTRAP__={version:1,viewId:"collapse"};
        const store=configureStore({reducer:{dock:dockReducer,session:sessionReducer}});
        store.dispatch(syncRegisteredPanels());store.dispatch(markFormsMounted(store.getState().dock.layout.order));
        if (plugin) {
            const layout=store.getState().dock.layout;const tree=layout.roots.main;
            if (tree.t==="split") store.dispatch(setDockLayout({...layout,roots:{...layout.roots,main:{...tree,ratio:0.37}}}));
            store.dispatch(openPanel({panelId:PANEL_FILE_BROWSER}));
        }
        const session=store.getState().session;const mounted=store.getState().dock.mountedFormIds;
        const originalRoots=store.getState().dock.layout.roots;
        const container=document.createElement("div");document.body.append(container);const root=createRoot(container);
        await act(async()=>{root.render(<Provider store={store}><TimelinePanelToggle /></Provider>);});
        expect(container.querySelector("button")?.getAttribute("aria-expanded")).toBe("true");
        await act(async()=>{container.querySelector("button")!.click();});
        expect(isPanelVisible(store.getState as never,PANEL_TIMELINE)).toBe(false);
        expect(isPanelVisible(store.getState as never,PANEL_PARAM_EDITOR)).toBe(true);
        expect(store.getState().session).toBe(session);expect(store.getState().dock.mountedFormIds).toEqual(mounted);
        expect(container.textContent).toContain("展开轨道面板");
        await act(async()=>{store.dispatch(setGutterSize({key:"timelineTrackHeaderPx",px:300}));});
        await act(async()=>{container.querySelector("button")!.click();});
        expect(isPanelVisible(store.getState as never,PANEL_TIMELINE)).toBe(true);expect(store.getState().session).toBe(session);
        expect(store.getState().dock.layout.roots).toEqual(originalRoots);
        expect(store.getState().dock.layout.gutters.timelineTrackHeaderPx).toBe(300);
        await act(async()=>{root.unmount();});container.remove();
    }
});
