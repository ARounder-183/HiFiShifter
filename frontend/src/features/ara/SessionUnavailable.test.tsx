/**
 * 插件会话不可用整页提示的契约。
 *
 * 【要钉死什么】终态的**协议码**决定界面给哪句指引 —— 这是本页存在的全部理由：
 * "宿主从未把本插件接入 ARA"（关窗重开无用，宿主根本不走 ARA 路径）与"编辑文档已
 * 关闭"（关窗重开有用）必须给出不同的下一步。按错误文本分支是被明令禁止的，因此
 * 只认码。
 */
// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, expect, test, vi } from "vitest";

vi.mock("../../services/api/diagnostics", () => ({
    openLogFolder: async () => ({ ok: true }),
}));

import { I18nProvider } from "../../i18n/I18nProvider";
import { enUS } from "../../i18n/en-US";
import { AppThemeProvider } from "../../theme/AppThemeProvider";
import {
    ERROR_CODE,
    markBackendTerminal,
    resetSessionGateForTests,
} from "../../services/sessionGate";
import { SessionUnavailable } from "./SessionUnavailable";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let container: HTMLDivElement;
let root: Root;

beforeEach(() => {
    localStorage.setItem("hifishifter.locale", "en-US");
    resetSessionGateForTests();
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
});

afterEach(() => {
    act(() => root.unmount());
    container.remove();
    resetSessionGateForTests();
    localStorage.clear();
});

function render(): string {
    act(() => {
        root.render(
            <I18nProvider>
                <AppThemeProvider>
                    <SessionUnavailable />
                </AppThemeProvider>
            </I18nProvider>,
        );
    });
    return container.textContent ?? "";
}

test("a never-bound session points at the ARA entry, not at reopening the window", () => {
    markBackendTerminal(
        "plug-in instance is not bound to an ARA document",
        ERROR_CODE.ARA_NOT_BOUND,
    );
    const text = render();
    expect(text).toContain(enUS.plugin_ara_not_bound);
    // 【为什么必须钉住反向】"关闭并重新打开插件窗口"对这条终态是**错的**建议：
    // 宿主根本不走 ARA 路径，重开会走完全相同的流程、得到完全相同的错误。
    expect(text).not.toContain(enUS.plugin_apply_session_ended);
});

test("a closed session keeps the close-and-reopen advice", () => {
    markBackendTerminal("document closed", ERROR_CODE.DOCUMENT_CLOSED);
    const text = render();
    expect(text).toContain(enUS.plugin_apply_session_ended);
    expect(text).not.toContain(enUS.plugin_ara_not_bound);
});

test("an old backend without a code falls back to the generic session-ended text", () => {
    // 旧后端不带 `errorCode`：只能展示通用说明，**不能**猜成某一种终态。
    markBackendTerminal("document closed");
    const text = render();
    expect(text).toContain(enUS.plugin_apply_session_ended);
});
