// @vitest-environment jsdom
/**
 * 宿主链接面板的契约。
 *
 * 【这个文件钉住什么】
 * 1. 工作区列表与只读（unreadable）项：读不出来的目录要作为**禁用项**并说明原因；
 * 2. 脏工程的确认流程：无论脏是前端已知还是**后端**发现，都不能静默覆盖本地工程；
 * 3. 提交结果与本地化错误：`status.ok === false` 时按协议码给可行动的说明；
 * 4. 能力声明：宿主不接受提交时按钮禁用并明说，而不是让用户点一次再吃错误。
 */
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, expect, test, vi } from "vitest";

const invokeMock = vi.fn();

vi.mock("../../services/invoke", () => ({
    invoke: (...args: unknown[]) => invokeMock(...args),
}));

import { I18nProvider } from "../../i18n/I18nProvider";
import { enUS } from "../../i18n/en-US";
import { AppThemeProvider } from "../../theme/AppThemeProvider";
import type { HostLinkSession } from "./hostLinkApi";
import { HostLinkPanel } from "./HostLinkPanel";
import { resetHostLinkStoreForTests } from "./hostLinkStore";

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
let calls: Array<{ command: string; args: unknown[] }>;

function session(overrides: Partial<HostLinkSession> = {}): HostLinkSession {
    return {
        attached: true,
        directory: "/workspaces/one",
        identity: "host:REAPER:doc",
        hostLabel: "REAPER",
        sourceKind: "ara",
        createdAt: 1_700_000_000_000,
        clips: 3,
        tracks: 2,
        audio: { complete: 2, partial: 1, extern: 0 },
        acceptsCommit: true,
        gatesGeometry: false,
        edits: 4,
        model: 5,
        skipped: "",
        skippedCount: 0,
        skippedDetail: [],
        losses: [],
        unsupportedPrecision: false,
        ...overrides,
    };
}

beforeEach(() => {
    calls = [];
    resetHostLinkStoreForTests();
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
    vi.restoreAllMocks();
});

/** 默认后端：一个可读工作区；每个命令返回可覆盖的结果。 */
function installBackend(overrides: Partial<Record<string, (...args: unknown[]) => unknown>> = {}) {
    invokeMock.mockImplementation((command: string, ...args: unknown[]) => {
        calls.push({ command, args });
        if (command in overrides) return Promise.resolve(overrides[command]!(...args));
        switch (command) {
            case "host_link_list":
                return Promise.resolve({
                    root: "/data/hifishifter",
                    items: [{ directory: "/workspaces/one", hostLabel: "REAPER" }],
                });
            case "host_link_open":
                return Promise.resolve(session());
            case "host_link_refresh":
                return Promise.resolve(session({ carriedParams: 2, carriedTracks: 1 }));
            case "host_link_submit":
                return Promise.resolve(session({ applied: true, pending: false }));
            case "host_link_release":
                return Promise.resolve({ ok: true, attached: false });
            default:
                return Promise.resolve(undefined);
        }
    });
}

async function render(
    dirty = false,
    onTimelineChanged: () => Promise<unknown> = async () => undefined,
) {
    await act(async () => {
        root.render(
            <AppThemeProvider>
                <I18nProvider>
                    <HostLinkPanel dirty={dirty} onTimelineChanged={onTimelineChanged} />
                </I18nProvider>
            </AppThemeProvider>,
        );
    });
    await flush();
}

async function flush() {
    await act(async () => {
        await Promise.resolve();
        await Promise.resolve();
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
    await flush();
}

function dialog(): HTMLElement | null {
    return document.body.querySelector<HTMLElement>('[role="dialog"]');
}

test("会话控件走设计系统原语：用 AppSelect 而不是原生控件", async () => {
    installBackend();
    await render();
    expect(container.querySelector('[role="combobox"]')).toBeTruthy();
    const bare = Array.from(container.querySelectorAll("button")).filter(
        (b) => b.getAttribute("role") !== "combobox",
    );
    expect(bare.length).toBeGreaterThan(0);
    for (const button of bare) {
        expect(button.className, button.textContent ?? "").toContain("app-button");
    }
});

test("读不出来的工作区作为禁用项并说明原因", async () => {
    installBackend({
        host_link_list: () => ({
            root: "/data/hifishifter",
            items: [{ directory: "/workspaces/broken", unreadable: "permission denied" }],
        }),
    });
    await render();
    // 说明挂在选中项下方，用户一眼能看到该去检查什么。
    expect(text()).toContain("/workspaces/broken");
    expect(text()).toContain("permission denied");
    // 没有可打开的工作区 ⇒ 打开按钮禁用。
    expect(buttonNamed(enUS.host_link_open)?.disabled).toBe(true);
});

test("后端发现的脏工程回到确认态，确认后用 force 重试", async () => {
    installBackend({
        host_link_open: (_directory, force) =>
            force ? session() : Promise.reject(new Error("dirty_project: unsaved changes")),
    });
    await render(false);
    await click(enUS.host_link_open);
    expect(calls.filter((c) => c.command === "host_link_open").map((c) => c.args[1])).toEqual([
        false,
    ]);
    expect(dialog()).toBeTruthy();

    await click(enUS.host_link_replace_dirty_confirm, document.body);
    expect(calls.filter((c) => c.command === "host_link_open").map((c) => c.args[1])).toEqual([
        false,
        true,
    ]);
});

test("前端已知脏工程先确认；取消不发送任何打开命令", async () => {
    installBackend();
    await render(true);
    await click(enUS.host_link_open);
    expect(calls.some((c) => c.command === "host_link_open")).toBe(false);
    await click(enUS.cancel, document.body);
    expect(calls.some((c) => c.command === "host_link_open")).toBe(false);

    await click(enUS.host_link_open);
    await click(enUS.host_link_replace_dirty_confirm, document.body);
    expect(calls.find((c) => c.command === "host_link_open")?.args[1]).toBe(true);
});

test("打开之后刷新时间轴，提交被拒时按码给本地化错误", async () => {
    let reloads = 0;
    installBackend({
        host_link_submit: () =>
            session({
                applied: false,
                pending: false,
                status: { ok: false, code: 12, atUnixMs: 1 },
            }),
    });
    await render(false, async () => {
        reloads++;
    });
    await click(enUS.host_link_open);
    expect(reloads).toBe(1);

    await click(enUS.host_link_submit);
    expect(text()).toContain(enUS.host_link_code_conflict);
});

test("宿主不接受提交时禁用提交并明说", async () => {
    installBackend({
        host_link_open: () => session({ acceptsCommit: false, gatesGeometry: true }),
    });
    await render();
    await click(enUS.host_link_open);
    expect(buttonNamed(enUS.host_link_submit)?.disabled).toBe(true);
    expect(text()).toContain(enUS.host_link_commit_unsupported);
    expect(text()).toContain(enUS.host_link_geometry_local_only);
});

test("重新导入后报告保留了多少曲线，并常驻说明", async () => {
    installBackend();
    await render();
    expect(text()).toContain(enUS.host_link_refresh_keeps_curves);
    await click(enUS.host_link_open);
    await click(enUS.host_link_refresh);
    expect(text()).toContain(
        enUS.host_link_carried_summary.replace("{params}", "2").replace("{tracks}", "1"),
    );
});

test("断开后清空会话，提交按钮回到禁用", async () => {
    installBackend();
    await render();
    await click(enUS.host_link_open);
    expect(buttonNamed(enUS.host_link_submit)?.disabled).toBe(false);
    await click(enUS.host_link_release);
    expect(buttonNamed(enUS.host_link_submit)?.disabled).toBe(true);
});
