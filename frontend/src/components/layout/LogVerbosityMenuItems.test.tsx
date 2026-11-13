/**
 * 帮助菜单四档单选项的渲染契约：4 个 menuitemradio + aria-checked。
 *
 * Radix 的 DropdownMenu 需要 Portal / DOM；这里用轻量的模块替身把 RadioGroup /
 * RadioItem / Label 映射成原生 role，避免为一条 UI 契约引入完整的浏览器交互栈。
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
vi.mock("@radix-ui/themes", async () => {
    const React = await import("react");
    const pass =
        (role: string) =>
        ({ children, ...rest }: { children?: React.ReactNode; disabled?: boolean }) =>
            React.createElement("div", { role, ...rest }, children);
    const Label = pass("presentation");
    const Separator = () => React.createElement("div", { role: "separator" });
    const Sub = ({ children }: { children?: React.ReactNode }) =>
        React.createElement("div", { role: "group" }, children);
    const SubTrigger = ({
        children,
        disabled,
    }: {
        children?: React.ReactNode;
        disabled?: boolean;
    }) =>
        React.createElement(
            "div",
            {
                role: "menuitem",
                "aria-haspopup": "menu",
                "aria-disabled": disabled,
                "data-trigger": "true",
            },
            children,
        );
    const SubContent = ({ children }: { children?: React.ReactNode }) =>
        React.createElement("div", { role: "menu" }, children);
    const RadioGroup = ({ children }: { children?: React.ReactNode }) =>
        React.createElement("div", { role: "group" }, children);
    const RadioItem = ({
        children,
        value,
        disabled,
    }: {
        children?: React.ReactNode;
        value: string;
        disabled?: boolean;
    }) =>
        React.createElement(
            "div",
            {
                role: "menuitemradio",
                "aria-checked": value === "low",
                "data-value": value,
                "aria-disabled": disabled,
            },
            children,
        );
    return {
        DropdownMenu: { Label, Separator, Sub, SubTrigger, SubContent, RadioGroup, RadioItem },
    };
});

import { LogVerbosityMenuItems } from "./LogVerbosityMenuItems";

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

async function renderMenu(): Promise<HTMLElement[]> {
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
    await act(async () => {
        root.render(<LogVerbosityMenuItems />);
    });
    // wait for the async getLogSettings promise to resolve
    await act(async () => {
        await Promise.resolve();
        await Promise.resolve();
    });
    return Array.from(container.querySelectorAll('[role="menuitemradio"]'));
}

beforeEach(() => {
    getLogSettings.mockReset().mockResolvedValue(VIEW);
    setLogSettings.mockReset();
    listen.mockReset().mockResolvedValue(() => {});
    if (root) {
        act(() => root.unmount());
        container?.remove();
    }
});

test("renders four radio items with the default low level checked", async () => {
    const items = await renderMenu();
    expect(items).toHaveLength(4);
    expect(items.map((item) => item.getAttribute("data-value"))).toEqual([
        "off",
        "low",
        "medium",
        "high",
    ]);
    expect(items[1].getAttribute("aria-checked")).toBe("true");
});

test("a startup override disables all four radio items", async () => {
    getLogSettings.mockResolvedValue({ ...VIEW, overridden: true, source: "env" });
    const items = await renderMenu();
    expect(items.every((item) => item.getAttribute("aria-disabled") === "true")).toBe(true);
});

test("the verbosity choices stay inside one submenu", async () => {
    await renderMenu();
    const triggers = container.querySelectorAll('[data-trigger="true"]');
    expect(triggers).toHaveLength(1);
    expect(triggers[0].textContent).toContain("menu_log_verbosity_current");
    expect(container.querySelectorAll('[role="menu"] [role="menuitemradio"]')).toHaveLength(4);
});
