// @vitest-environment jsdom
/**
 * 插件侧「导出到 HiFiShifter」控件的契约。
 *
 * 【这个文件钉住什么】
 * 1. 按钮发出的是插件的 `host_link_export` 命令；
 * 2. 导出成功后**目录路径可见**（用户要把它交给独立 App）；
 * 3. 详情里能看到状态 JSON 与最近一次提交的结果 —— 没有这两样，用户无从判断
 *    "宿主到底收没收到我的编辑"。
 */
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, expect, test, vi } from "vitest";

const invokeMock = vi.fn();

vi.mock("../../services/invoke", () => ({
    invoke: (...args: unknown[]) => invokeMock(...args),
    invokeErrorKind: () => "transient",
    invokeErrorCode: () => null,
}));

import { I18nProvider } from "../../i18n/I18nProvider";
import { enUS } from "../../i18n/en-US";
import { AppThemeProvider } from "../../theme/AppThemeProvider";
import type { HostLinkExportStatus } from "./hostLinkApi";
import { HostLinkExportControl } from "./HostLinkExportControl";
import { resetHostLinkPluginStoreForTests } from "./hostLinkPluginStore";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

class ResizeObserverStub {
    observe() {}
    unobserve() {}
    disconnect() {}
}
(globalThis as { ResizeObserver?: unknown }).ResizeObserver ??= ResizeObserverStub;

const LOCALE_KEY = "hifishifter.locale";
let container: HTMLDivElement;
let root: Root;
let status: HostLinkExportStatus;

const READY: HostLinkExportStatus = {
    attached: true,
    phase: "ready",
    identity: "host:REAPER:doc",
    hostLabel: "REAPER",
    sourceKind: "ara",
    directory: "/workspaces/one",
    exportedAt: 1_700_000_000_000,
    clips: 2,
    tracks: 1,
    audio: { complete: 2, partial: 0, extern: 0 },
    losses: 0,
    acceptsCommit: true,
    lastEdit: { ok: true, edits: 4, model: 5, atUnixMs: 1_700_000_001_000 },
};

beforeEach(() => {
    vi.useFakeTimers();
    invokeMock.mockReset();
    resetHostLinkPluginStoreForTests();
    status = { attached: false };
    invokeMock.mockImplementation((command: string) => {
        switch (command) {
            case "host_link_status":
                return Promise.resolve(status);
            case "host_link_export":
                status = READY;
                return Promise.resolve(status);
            case "host_link_release":
                status = { attached: false };
                return Promise.resolve(status);
            default:
                return Promise.resolve(undefined);
        }
    });
    localStorage.setItem(LOCALE_KEY, "en-US");
    vi.spyOn(console, "error").mockImplementation(() => undefined);
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
});

afterEach(() => {
    act(() => root.unmount());
    container.remove();
    document.body.innerHTML = "";
    localStorage.removeItem(LOCALE_KEY);
    vi.useRealTimers();
    vi.restoreAllMocks();
});

async function render() {
    await act(async () => {
        root.render(
            <AppThemeProvider>
                <I18nProvider>
                    <HostLinkExportControl />
                </I18nProvider>
            </AppThemeProvider>,
        );
    });
    await act(async () => {
        await vi.advanceTimersByTimeAsync(0);
    });
}

function text(): string {
    return container.textContent ?? "";
}

function buttonNamed(label: string, scope: ParentNode = container): HTMLButtonElement | undefined {
    return Array.from(scope.querySelectorAll("button")).find((b) => b.textContent?.includes(label));
}

async function click(label: string, scope: ParentNode = container) {
    const button = buttonNamed(label, scope);
    expect(button, label).toBeTruthy();
    await act(async () => button!.click());
    await act(async () => {
        await vi.advanceTimersByTimeAsync(0);
    });
}

test("未导出时给出按钮与「未导出」状态", async () => {
    await render();
    expect(buttonNamed(enUS.host_link_export)).toBeTruthy();
    expect(text()).toContain(enUS.host_link_phase_idle);
});

test("导出成功后展示目录路径与复制入口，并允许断开", async () => {
    await render();
    await click(enUS.host_link_export);
    expect(invokeMock).toHaveBeenCalledWith("host_link_export");
    expect(text()).toContain(enUS.host_link_phase_ready);
    expect(text()).toContain("/workspaces/one");
    expect(buttonNamed(enUS.host_link_copy_path)).toBeTruthy();
    expect(buttonNamed(enUS.host_link_release)).toBeTruthy();
});

test("详情对话框展示状态 JSON 与最近一次提交结果", async () => {
    await render();
    await click(enUS.host_link_export);
    await click(enUS.host_link_details);
    const pre = document.body.querySelector('[data-testid="host-link-status-json"]');
    expect(pre?.textContent).toContain('"directory": "/workspaces/one"');
    expect(document.body.textContent).toContain(enUS.host_link_last_edit_applied);
    expect(document.body.textContent).toContain("/workspaces/one");
});
