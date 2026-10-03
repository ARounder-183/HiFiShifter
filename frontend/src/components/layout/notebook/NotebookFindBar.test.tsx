// @vitest-environment jsdom
/*
 * 查找条与文档变化的一致性。
 *
 * 【要钉死什么】查找条打开期间用户仍会继续编辑正文；匹配列表必须随文档更新，
 * 否则 Enter / ▲ / ▼ 会按**旧位置**选中错位的文本，计数也停在旧值。
 */

import { Editor } from "@tiptap/core";
import { act } from "react";
import { createRoot } from "react-dom/client";
import { expect, test } from "vitest";

import { I18nProvider } from "../../../i18n/I18nProvider";
import { NotebookFindBar } from "./NotebookFindBar";
import { buildNotebookExtensions } from "./notebookExtensions";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

interface Mounted {
    editor: Editor;
    host: HTMLElement;
    unmount: () => Promise<void>;
}

async function mountFindBar(): Promise<Mounted> {
    const editor = new Editor({
        element: document.createElement("div"),
        extensions: buildNotebookExtensions({ markdownShortcuts: false, slashCommands: false }),
        content: "alpha beta",
    });
    const host = document.createElement("div");
    document.body.append(host);
    const root = createRoot(host);
    await act(async () => {
        root.render(
            <I18nProvider>
                <NotebookFindBar
                    editor={editor}
                    sourceMode={false}
                    sourceValue=""
                    getSourceTextarea={() => null}
                    onClose={() => {}}
                />
            </I18nProvider>,
        );
    });
    return {
        editor,
        host,
        unmount: async () => {
            await act(async () => root.unmount());
            editor.destroy();
            host.remove();
        },
    };
}

/** 往受控 input 里输入文本（React 19 需要原生 setter + input 事件）。 */
async function type(input: HTMLInputElement, value: string): Promise<void> {
    await act(async () => {
        const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")?.set;
        setter?.call(input, value);
        input.dispatchEvent(new Event("input", { bubbles: true }));
    });
}

test("正文变化后匹配数量重新计算", async () => {
    const mounted = await mountFindBar();
    try {
        const input = mounted.host.querySelector<HTMLInputElement>(".hs-notebook-findbar input");
        expect(input).not.toBeNull();
        const count = () =>
            mounted.host.querySelector(".hs-notebook-findbar-count")?.textContent ?? "";

        await type(input as HTMLInputElement, "alpha");
        expect(count()).toBe("1/1");

        // 再插入一个 "alpha"：旧的 memo 依赖里没有文档内容，会一直显示 1/1。
        await act(async () => {
            mounted.editor.commands.insertContentAt(
                mounted.editor.state.doc.content.size,
                " alpha",
            );
        });
        expect(count()).toBe("1/2");
    } finally {
        await mounted.unmount();
    }
});
