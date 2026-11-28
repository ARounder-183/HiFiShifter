/**
 * 帮助菜单四档单选项的**数据契约**：4 个 menuitemradio + aria-checked。
 *
 * 【为什么断言数据而不是 DOM】条目现在是一段 AppMenuItemSpec 数据（菜单表面只有
 * 一套，条目只有一个渲染路径）。断言数据比断言"渲染出来的 role"更靠前一层：
 * 数据对了，渲染一定对（那条路径有它自己的用例）；数据错了，渲染再对也没用。
 */
// @vitest-environment jsdom
import { createRoot, type Root } from "react-dom/client";
import { act } from "react";
import { beforeEach, expect, test, vi } from "vitest";

const getLogSettings = vi.fn();
const setLogSettings = vi.fn();
const listen = vi.fn();

vi.mock("../../services/api/logSettings", () => ({
    getLogSettings: (...args: unknown[]) => getLogSettings(...args),
    setLogSettings: (...args: unknown[]) => setLogSettings(...args),
}));
vi.mock("../../services/hostEvents", () => ({
    listen: (...args: unknown[]) => listen(...args),
}));
vi.mock("../../i18n/I18nProvider", () => ({
    useI18n: () => ({
        t: (key: string) => key,
        tf: (key: string) => key,
        tVars: (key: string, vars: Record<string, string>) =>
            key + ":" + Object.values(vars).join(","),
    }),
}));

import { useLogVerbosityMenu, type LogVerbosityMenu } from "./useLogVerbosityMenu";

const VIEW = {
    ok: true,
    verbosity: "low",
    persistedVerbosity: "low",
    source: "persisted",
    overridden: false,
    fileEnabled: true,
    fileWritable: true,
    loggingHealthy: true,
    hostLoggerOwned: false,
    files: [],
    totalBytes: 0,
    maxFileBytes: 8 * 1024 * 1024,
    maxFiles: 3,
    policyVersion: 1,
    recentChanges: [],
};

(globalThis as unknown as { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let container: HTMLDivElement;
let root: Root;
let captured: LogVerbosityMenu | null = null;

function Harness() {
    captured = useLogVerbosityMenu();
    return null;
}

async function renderMenu(): Promise<LogVerbosityMenu> {
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
    await act(async () => {
        root.render(<Harness />);
    });
    // 等 getLogSettings 的 promise 落地。
    await act(async () => {
        await Promise.resolve();
        await Promise.resolve();
    });
    if (!captured) throw new Error("hook 未产出菜单数据");
    return captured;
}

beforeEach(() => {
    getLogSettings.mockReset().mockResolvedValue(VIEW);
    setLogSettings.mockReset();
    listen.mockReset().mockResolvedValue(() => {});
    captured = null;
    if (root) {
        act(() => root.unmount());
        container?.remove();
    }
});

test("产出四个互斥档位，默认 low 打勾", async () => {
    const menu = await renderMenu();
    expect(menu.items).toHaveLength(4);
    expect(menu.items.map((item) => item.key)).toEqual(["off", "low", "medium", "high"]);
    // 互斥档位必须是 radio：挂成 checkbox 时读屏会念"复选框"。
    expect(menu.items.every((item) => item.radio === true)).toBe(true);
    expect(menu.items.map((item) => item.checked)).toEqual([false, true, false, false]);
    expect(menu.triggerLabel).toContain("menu_log_verbosity");
    expect(menu.disabled).toBe(false);
});

test("启动期被环境变量覆盖时，四项全部禁用", async () => {
    getLogSettings.mockResolvedValue({ ...VIEW, overridden: true, source: "env" });
    const menu = await renderMenu();
    expect(menu.items.every((item) => item.disabled === true)).toBe(true);
});

test("子菜单里只有四档：没有标题行、没有分隔线", async () => {
    // 一级菜单只占一行；展开后不应再出现重复的标题、当前值或说明。
    const menu = await renderMenu();
    expect(menu.items.some((item) => item.heading === true)).toBe(false);
    expect(menu.items.some((item) => item.separatorBefore === true)).toBe(false);
    expect(menu.items.some((item) => item.items !== undefined)).toBe(false);
});
