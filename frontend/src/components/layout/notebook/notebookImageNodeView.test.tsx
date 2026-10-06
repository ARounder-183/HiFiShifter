// @vitest-environment jsdom
/*
 * 图片卡片右键菜单的**内容**回归测试。
 *
 * 【要钉死什么】这张菜单在并入共享 builder（`notebookMenu.ts`）之前是就地手写的
 * 五项，并入之后由"目标 + 动作"推导 —— 于是出现了一个新的失败模式：**动作忘了
 * 提供，菜单项就静默消失**（builder 的契约是"没有动作就没有这一项"）。手工点测
 * 很容易漏掉少一项，因此这里把"图片专属五项 + 通用复制组"逐条钉住。
 *
 * 【为什么真的挂 React】与 `hifiClipNodeView.test.tsx` 同理：`ReactNodeViewRenderer`
 * 在没有 `EditorContent` 的路径下渲染的是空 span，组件根本不会跑。
 */

import { configureStore } from "@reduxjs/toolkit";
import { Editor } from "@tiptap/core";
import { EditorContent } from "@tiptap/react";
import { act } from "react";
import { createRoot } from "react-dom/client";
import { Provider } from "react-redux";
import { afterEach, beforeEach, expect, test, vi } from "vitest";

const resolveImage = vi.fn();

vi.mock("./notebookImageCache", () => ({
    resolveImage: (...args: unknown[]) => resolveImage(...args),
    subscribeAssetInvalidation: () => () => {},
}));

import notebookReducer from "../../../features/notebook/notebookSlice";
import { I18nProvider } from "../../../i18n/I18nProvider";
import { buildNotebookExtensions } from "./notebookExtensions";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const SRC = "hifi-asset://abc123.png";

beforeEach(() => {
    resolveImage.mockReset();
    resolveImage.mockResolvedValue({ url: "blob:image", missing: false });
    // jsdom 没有 createObjectURL；NodeView 的复制路径会用到它。
    (URL as unknown as { createObjectURL: (blob: Blob) => string }).createObjectURL = () =>
        "blob:stub";
});

afterEach(() => {
    document.body.innerHTML = "";
});

/** 挂一张图片卡片，返回右键后菜单里的条目文案。 */
async function openImageMenu(): Promise<{ items: string[]; host: HTMLElement }> {
    // 图片卡片只从 session 读一个 `project.path`（决定相对路径的基准目录），
    // 因此这里给一个够用的桩，而不是把整个 session 切片拖进来。
    const sessionStub = (state = { project: { path: null } }) => state;
    const store = configureStore({
        reducer: { notebook: notebookReducer, session: sessionStub },
    });
    const editor = new Editor({
        element: document.createElement("div"),
        extensions: buildNotebookExtensions({ markdownShortcuts: true, slashCommands: false }),
        content: `![图](${SRC})`,
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
    // 让 `resolveImage` 的 promise 落地。
    await act(async () => {
        await Promise.resolve();
    });

    const image = host.querySelector("img");
    if (!image) throw new Error("图片卡片没有渲染出来");
    await act(async () => {
        image.dispatchEvent(
            new MouseEvent("contextmenu", { bubbles: true, clientX: 10, clientY: 10 }),
        );
    });

    const menu = document.querySelector('[role="menu"]');
    if (!menu) throw new Error("右键菜单没有打开");
    const items = Array.from(menu.querySelectorAll(".hs-menu__item")).map(
        (item) => item.textContent ?? "",
    );
    await act(async () => root.unmount());
    editor.destroy();
    host.remove();
    return { items, host };
}

test("图片菜单含图片专属五项", async () => {
    const { items } = await openImageMenu();
    const text = items.join("\n");
    expect(text).toContain("Copy image");
    expect(text).toContain("Reset display width");
    expect(text).toContain("Edit alt text");
    expect(text).toContain("Remove image");
});

test("图片菜单含通用复制组（并入 builder 后新增的能力）", async () => {
    const { items } = await openImageMenu();
    const text = items.join("\n");
    expect(text).toContain("Cut");
    expect(text).toContain("Select All");
    // "复制为…" 现在是子菜单：变体在展开后才出现（顶层只留一个触发项）。
    expect(text).toContain("Copy as");
    expect(text).not.toContain("Copy as Markdown");
});

test("图片菜单不含只对文本有意义的项", async () => {
    const { items } = await openImageMenu();
    const text = items.join("\n");
    // 这些项需要面板的 insertContext / 编辑器历史，图片卡片都没有 ——
    // builder 的契约是"没有动作就没有这一项"，因此它们必须不出现。
    for (const absent of ["Paste", "Bold", "Heading 1", "Insert table", "Undo"]) {
        expect(text, absent).not.toContain(absent);
    }
});
