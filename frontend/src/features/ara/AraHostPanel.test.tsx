// @vitest-environment jsdom
/**
 * ARA 宿主会话面板的契约。
 *
 * 【这个文件钉住两件事】
 * 1. 会话语义（原有）：连接 / 提交冲突 / 脏工程替换确认都不能静默覆盖用户工程；
 * 2. 表面合规（新增）：面板只用设计系统原语 —— 会话控件曾经是原生 `<select>` +
 *    Radix 裸按钮 + 内联样式，与其余界面不成体系。
 *
 * 【为什么断言 `app-button` 类】`AppButton` 是按钮外观的唯一来源，它一定会带上
 * 这个类；裸 Radix `Button` 不会。用类名而不是文案，断言因此与语系无关。
 */
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, expect, test, vi } from "vitest";

import { I18nProvider } from "../../i18n/I18nProvider";
import { enUS } from "../../i18n/en-US";
import { AppThemeProvider } from "../../theme/AppThemeProvider";
import { AraHostPanel } from "./AraHostPanel";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

/*
 * jsdom 没有 `ResizeObserver`，而 Radix 的 Dialog/Select 在布局副作用里会用它。
 * 与 `ExportAudioDialog.cancel.test.tsx` 同一处处理。
 */
class ResizeObserverStub {
    observe() {}
    unobserve() {}
    disconnect() {}
}
(globalThis as { ResizeObserver?: unknown }).ResizeObserver ??= ResizeObserverStub;

let container: HTMLDivElement;
let root: Root;
let calls: Array<{ command: string; args?: Record<string, unknown> }>;
let conflict: boolean;
let staleDirty: boolean;

/** 强制 en-US：文案断言取词表本身，改文案不会让测试失效。 */
const LOCALE_KEY = "hifishifter.locale";

beforeEach(() => {
    calls = [];
    conflict = false;
    staleDirty = false;
    localStorage.setItem(LOCALE_KEY, "en-US");
    vi.spyOn(console, "error").mockImplementation(() => undefined);
    window.__TAURI__ = {
        core: {
            invoke: async <T,>(command: string, args?: Record<string, unknown>) => {
                calls.push({ command, args });
                if (command === "ara_list_instances")
                    return [{ instance_id: "instance", name: "REAPER vocal", pid: 42 }] as T;
                if (command === "ara_submit" && conflict)
                    return Promise.reject("Conflict: host changed");
                if (command === "ara_connect" && staleDirty && !args?.force)
                    return Promise.reject("dirty_project: confirm replacement of unsaved edits");
                return { ok: true, instance_id: "instance", revision: 2, model_revision: 3 } as T;
            },
        },
    };
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
});

afterEach(() => {
    act(() => root.unmount());
    container.remove();
    document.body.innerHTML = "";
    localStorage.removeItem(LOCALE_KEY);
    delete window.__TAURI__;
    vi.restoreAllMocks();
});

async function render(dirty: boolean, onTimelineChanged: () => Promise<unknown> = async () => undefined) {
    await act(async () =>
        root.render(
            <AppThemeProvider>
                <I18nProvider>
                    <AraHostPanel dirty={dirty} onTimelineChanged={onTimelineChanged} />
                </I18nProvider>
            </AppThemeProvider>,
        ),
    );
}

/** 面板内的按钮（含对话框里的，对话框被 portal 到 body）。 */
function buttons(): HTMLButtonElement[] {
    return Array.from(container.querySelectorAll("button"));
}

async function click(label: string, scope: ParentNode = container) {
    const button = Array.from(scope.querySelectorAll("button")).find((b) =>
        b.textContent?.includes(label),
    );
    expect(button, label).toBeTruthy();
    await act(async () => button!.click());
}

/** Radix 对话框 portal 到 body，且 role 是 dialog（不是 alertdialog）。 */
function dialog(): HTMLElement | null {
    return document.body.querySelector<HTMLElement>('[role="dialog"]');
}

test("session controls use design-system primitives instead of raw form chrome", async () => {
    await render(false);
    // 实例选择必须是组合框触发器（AppSelect），而不是原生 <select>。
    expect(container.querySelector('[role="combobox"]')).toBeTruthy();
    const bare = buttons().filter((b) => b.getAttribute("role") !== "combobox");
    expect(bare.length, "面板里应有会话动作按钮").toBeGreaterThan(0);
    for (const button of bare) {
        expect(button.className, button.textContent ?? "").toContain("app-button");
    }
});

test("backend dirty guard still opens explicit confirmation when frontend state is stale", async () => {
    staleDirty = true;
    await render(false);
    await click(enUS.ara_connect);
    expect(dialog()).toBeTruthy();
    await click(enUS.ara_replace_dirty_confirm, document.body);
    expect(dialog()).toBeNull();
    expect(calls.filter((c) => c.command === "ara_connect").map((c) => c.args?.force)).toEqual([
        false,
        true,
    ]);
});

test("connect refreshes the timeline and exposes a submit conflict without discarding the session", async () => {
    let reloads = 0;
    await render(false, async () => {
        reloads++;
    });
    await click(enUS.ara_connect);
    expect(reloads).toBe(1);
    expect(container.querySelector<HTMLElement>('[role="combobox"]')?.getAttribute("disabled")).not.toBeNull();
    expect(calls.find((c) => c.command === "ara_connect")?.args).toEqual({
        instanceId: "instance",
        force: false,
    });

    conflict = true;
    await click(enUS.ara_submit);
    expect(container.querySelector('[role="alert"]')?.textContent).toContain("Conflict");
    expect(reloads).toBe(1);
    expect(buttonNamed(enUS.ara_disconnect)?.disabled).toBe(false);

    conflict = false;
    await click(enUS.ara_refresh_host);
    expect(reloads).toBe(2);
    await click(enUS.ara_disconnect);
    expect(buttonNamed(enUS.ara_submit)?.disabled).toBe(true);
});

test("dirty connect waits for explicit replace confirmation and a canceled confirmation sends no snapshot", async () => {
    await render(true);
    await click(enUS.ara_connect);
    expect(calls.some((c) => c.command === "ara_connect")).toBe(false);
    await click(enUS.cancel, document.body);
    expect(calls.some((c) => c.command === "ara_connect")).toBe(false);

    await click(enUS.ara_connect);
    await click(enUS.ara_replace_dirty_confirm, document.body);
    expect(calls.find((c) => c.command === "ara_connect")?.args?.force).toBe(true);

    await click(enUS.ara_refresh_host);
    expect(calls.some((c) => c.command === "ara_refresh")).toBe(false);
    await click(enUS.ara_replace_dirty_confirm, document.body);
    expect(calls.find((c) => c.command === "ara_refresh")?.args).toEqual({ force: true });
});

function buttonNamed(label: string): HTMLButtonElement | undefined {
    return buttons().find((b) => b.textContent?.includes(label));
}
