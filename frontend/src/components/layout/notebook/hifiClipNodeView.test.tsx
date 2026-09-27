// @vitest-environment jsdom
/*
 * 剪贴板暂存块的**真实渲染**回归测试。
 *
 * 【为什么必须有】这个节点曾经"解析正确、序列化正确、样式也写好了"，唯独没有
 * 注册 NodeView —— 于是 ProseMirror 退回 `renderHTML`，把正文里那块载荷渲染成
 * 一个没有子节点的空 `<div data-hifi-clip>`。文档里明明有东西，用户看到的却是
 * 一片空白。往返测试（`markdownRoundTrip`）抓不到它：**节点存在**与**节点可见**
 * 是两件事，前者一路绿灯。
 *
 * 【为什么必须真的挂 React】`ReactNodeViewRenderer` 在编辑器没有
 * `editor.contentComponent` 时（即没有经过 `EditorContent` 挂载，例如直接
 * `new Editor({ element })`）返回的是一个空 `<span>` 兜底视图，组件根本不会
 * 被渲染。任何"不挂 React"的测试都会毫无意义地通过，因此这里走
 * `createRoot` + `EditorContent` 的真实挂载路径。
 *
 * 【依赖 Redux 与 i18n 的原因】卡片要读 `notebook` 切片的设置与附件索引才能
 * 判断"载荷还在不在"，文案走 `useI18n`，所以必须同时提供 `Provider` 与
 * `I18nProvider`（后者在非 Tauri 环境下不会调用后端）。
 */

import { configureStore } from "@reduxjs/toolkit";
import { Editor } from "@tiptap/core";
import { EditorContent } from "@tiptap/react";
import { act } from "react";
import { createRoot } from "react-dom/client";
import { Provider } from "react-redux";
import { expect, test } from "vitest";

import notebookReducer, {
    type NotebookAssetSummary,
} from "../../../features/notebook/notebookSlice";
import { I18nProvider } from "../../../i18n/I18nProvider";
import { buildNotebookExtensions } from "./notebookExtensions";

// React 19 要求显式声明这是 act() 环境，否则每次 act 都会打印一条警告。
(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const CLIP_ID = "7c1e9a4b2d";

/** 正文里的一块暂存载荷（与 `hifiClipBlock.ts` 的写入端同构）。 */
const CLIP_FENCE = [
    "```hifi-clip",
    `id: ${CLIP_ID}`,
    "kind: clips",
    "title: 副歌 A",
    "source: 我的歌",
    "encoding: fragment",
    "clips: 3",
    "tracks: 2",
    "duration: 4.820",
    "captured: 2026-09-23T10:04:11Z",
    "```",
].join("\n");

/** 附件索引里"载荷确实在"的那一条。 */
const HEALTHY_ASSET: NotebookAssetSummary = {
    id: CLIP_ID,
    kind: "clip_payload",
    ext: "hsf",
    mime: "application/octet-stream",
    byteLen: 1024,
    createdAtMs: 0,
    meta: { clipKind: "clips", title: "副歌 A" },
    hasData: true,
};

const PARAM_ID = "a1b2c3d4e5";

/** 参数线载荷：与时间轴载荷的差别不只是数据，动作文案也必须不同。 */
const PARAM_FENCE = [
    "```hifi-clip",
    `id: ${PARAM_ID}`,
    "kind: param",
    "title: 共振峰",
    "encoding: param",
    "param: formant",
    "frames: 240",
    "captured: 2026-09-23T10:04:11Z",
    "```",
].join("\n");

const PARAM_ASSET: NotebookAssetSummary = {
    id: PARAM_ID,
    kind: "clip_payload",
    ext: "hsp",
    mime: "application/json",
    byteLen: 512,
    createdAtMs: 0,
    meta: { clipKind: "param", title: "共振峰" },
    hasData: true,
};

interface Mounted {
    host: HTMLElement;
    editor: Editor;
    /** 卸载 React 树并销毁编辑器（各用例用 try/finally 调用）。 */
    unmount: () => Promise<void>;
}

/** 用真实挂载路径把一块暂存载荷渲染出来。 */
async function mountCard(options: {
    editable?: boolean;
    /** 正文里的暂存块；缺省是一块时间轴片段载荷。 */
    content?: string;
    /** 附件索引内容；缺省为空 = 载荷已丢失。 */
    assets?: NotebookAssetSummary[];
}): Promise<Mounted> {
    const initial = notebookReducer(undefined, { type: "@@init" });
    const store = configureStore({
        reducer: { notebook: notebookReducer },
        preloadedState: {
            notebook: {
                ...initial,
                assetIndex: Object.fromEntries(
                    (options.assets ?? []).map((entry) => [entry.id, entry]),
                ),
            },
        },
    });

    const editor = new Editor({
        element: document.createElement("div"),
        extensions: buildNotebookExtensions({ markdownShortcuts: true, slashCommands: false }),
        content: options.content ?? CLIP_FENCE,
        editable: options.editable ?? true,
    });

    const host = document.createElement("div");
    document.body.append(host);
    const root = createRoot(host);
    await act(async () => {
        root.render(
            <Provider store={store}>
                <I18nProvider>
                    <EditorContent editor={editor} />
                </I18nProvider>
            </Provider>,
        );
    });

    return {
        host,
        editor,
        unmount: async () => {
            await act(async () => root.unmount());
            editor.destroy();
            host.remove();
        },
    };
}

test("暂存块在富文本里渲染成卡片，而不是一片空白", async () => {
    const mounted = await mountCard({ assets: [HEALTHY_ASSET] });
    try {
        // 这一条就是本文件存在的理由：没有注册 NodeView 时整个选择器为空。
        const card = mounted.host.querySelector(".hs-notebook-clip");
        expect(card, "正文里的暂存块没有渲染成卡片（NodeView 未注册？）").not.toBeNull();
        expect(card?.getAttribute("data-missing")).toBe("false");

        // 围栏正文必须真的被读成卡片信息，而不是只把 `<div data-hifi-clip>` 塞进 DOM。
        expect(mounted.host.querySelector(".hs-notebook-clip-title")?.textContent).toContain(
            "副歌 A",
        );
        const meta = mounted.host.querySelector(".hs-notebook-clip-meta")?.textContent ?? "";
        expect(meta).toContain("3");
        expect(meta).toContain("0:04.820");

        // 载荷在时两个动作按钮可用，且时间轴载荷说的是"插入"。
        const actions = mounted.host.querySelectorAll(".hs-notebook-clip-actions button");
        expect(actions.length).toBe(2);
        for (const button of actions) expect(button.hasAttribute("disabled")).toBe(false);
        expect(buttonLabels(mounted.host)).toContain("Insert into timeline");
    } finally {
        await mounted.unmount();
    }
});

test("载荷缺失时卡片仍然可见，只是动作不可用", async () => {
    const mounted = await mountCard({ assets: [] });
    try {
        const card = mounted.host.querySelector(".hs-notebook-clip");
        expect(card).not.toBeNull();
        expect(card?.getAttribute("data-missing")).toBe("true");
        // "为什么不给点"必须说清楚，而不是静默留两个灰按钮。
        expect(mounted.host.querySelector(".hs-notebook-clip-missing")).not.toBeNull();
        for (const button of mounted.host.querySelectorAll(".hs-notebook-clip-actions button")) {
            expect(button.hasAttribute("disabled")).toBe(true);
        }
    } finally {
        await mounted.unmount();
    }
});

test("只读编辑器（分栏预览栏）里卡片可见但不提供操作", async () => {
    const mounted = await mountCard({ editable: false, assets: [HEALTHY_ASSET] });
    try {
        // 预览栏与左栏长得一样：卡片本体照常渲染。
        expect(mounted.host.querySelector(".hs-notebook-clip")).not.toBeNull();
        // 但操作入口必须消失 —— 只读栏里的按钮只会误导。
        expect(mounted.host.querySelector(".hs-notebook-clip-more")).toBeNull();
        expect(mounted.host.querySelector(".hs-notebook-clip-actions")).toBeNull();
    } finally {
        await mounted.unmount();
    }
});

test("参数线载荷的动作文案是「应用到参数编辑器」，且不提供轨道插入", async () => {
    const mounted = await mountCard({ content: PARAM_FENCE, assets: [PARAM_ASSET] });
    try {
        // 徽标如实标注种类：参数线载荷不是"片段"。
        expect(mounted.host.querySelector(".hs-notebook-clip-badge")?.textContent).toBe(
            "Parameter",
        );

        // 参数线载荷没有时间轴几何：主操作说的是"应用到参数编辑器"。
        const labels = buttonLabels(mounted.host);
        expect(labels).toContain("Apply to parameter editor");
        expect(labels).not.toContain("Insert into timeline");

        // ⋯ 菜单里同样不能出现轨道相关入口（它是时间轴载荷的语汇）。
        await act(async () => {
            mounted.host.querySelector<HTMLButtonElement>(".hs-notebook-clip-more")?.click();
        });
        const menu = mounted.host.querySelector('[role="menu"]');
        expect(menu, "⋯ 菜单没有打开").not.toBeNull();
        const menuText = menu?.textContent ?? "";
        expect(menuText).toContain("Apply to parameter editor");
        expect(menuText).not.toContain("Insert as new tracks");
        expect(menuText).not.toContain("Insert into timeline");
    } finally {
        await mounted.unmount();
    }
});

/** 卡片动作区两个按钮的文案。 */
function buttonLabels(host: HTMLElement): string[] {
    return Array.from(host.querySelectorAll(".hs-notebook-clip-actions button")).map(
        (button) => button.textContent ?? "",
    );
}
