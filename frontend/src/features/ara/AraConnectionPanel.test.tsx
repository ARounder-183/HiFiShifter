// @vitest-environment jsdom
// ARA 操作契约：连接、提交冲突及替换确认不能静默覆盖用户工程。
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, expect, test, vi } from "vitest";
import { AraConnectionPanel } from "./AraConnectionPanel";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
let container: HTMLDivElement;
let root: Root;
let calls: Array<{ command: string; args?: Record<string, unknown> }>;
let conflict: boolean;
let staleDirty: boolean;
beforeEach(() => {
    calls = [];
    conflict = false;
    staleDirty = false;
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
test("backend dirty guard still opens explicit confirmation when frontend state is stale", async () => {
    staleDirty = true;
    await act(async () =>
        root.render(<AraConnectionPanel dirty={false} onTimelineChanged={async () => undefined} />),
    );
    await click("连接");
    expect(container.querySelector('[role="alertdialog"]')).toBeTruthy();
    await click("替换未保存工程");
    expect(container.querySelector('[role="alertdialog"]')).toBeNull();
    expect(calls.filter((c) => c.command === "ara_connect").map((c) => c.args?.force)).toEqual([
        false,
        true,
    ]);
});
afterEach(() => {
    act(() => root.unmount());
    container.remove();
    delete window.__TAURI__;
    vi.restoreAllMocks();
});
async function click(label: string) {
    const button = Array.from(container.querySelectorAll("button")).find((b) =>
        b.textContent?.includes(label),
    );
    expect(button, label).toBeTruthy();
    await act(async () => button!.click());
}
test("connect refreshes original timeline and exposes submit conflict without discarding local session", async () => {
    let reloads = 0;
    await act(async () =>
        root.render(
            <AraConnectionPanel
                dirty={false}
                onTimelineChanged={async () => {
                    reloads++;
                }}
            />,
        ),
    );
    await click("连接");
    expect(reloads).toBe(1);
    expect(container.querySelector<HTMLSelectElement>("select")?.disabled).toBe(true);
    expect(calls.find((c) => c.command === "ara_connect")?.args).toEqual({
        instanceId: "instance",
        force: false,
    });
    conflict = true;
    await click("提交到REAPER");
    expect(container.querySelector('[role="alert"]')?.textContent).toContain("Conflict");
    expect(reloads).toBe(1);
    expect(
        Array.from(container.querySelectorAll("button")).find((b) =>
            b.textContent?.includes("断开"),
        )?.disabled,
    ).toBe(false);
    conflict = false;
    await click("刷新宿主");
    expect(reloads).toBe(2);
    await click("断开");
    expect(
        Array.from(container.querySelectorAll("button")).find((b) =>
            b.textContent?.includes("提交到REAPER"),
        )?.disabled,
    ).toBe(true);
});
test("dirty connect waits for explicit replace confirmation and canceled confirmation sends no snapshot", async () => {
    await act(async () =>
        root.render(<AraConnectionPanel dirty={true} onTimelineChanged={async () => undefined} />),
    );
    await click("连接");
    expect(calls.some((c) => c.command === "ara_connect")).toBe(false);
    await click("取消");
    expect(calls.some((c) => c.command === "ara_connect")).toBe(false);
    await click("连接");
    await click("替换未保存工程");
    expect(calls.find((c) => c.command === "ara_connect")?.args?.force).toBe(true);
    await click("刷新宿主");
    expect(calls.some((c) => c.command === "ara_refresh")).toBe(false);
    await click("替换未保存工程");
    expect(calls.find((c) => c.command === "ara_refresh")?.args).toEqual({ force: true });
});
