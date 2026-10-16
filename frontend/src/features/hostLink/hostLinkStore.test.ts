/**
 * 宿主链接会话 store 的契约。
 *
 * 【为什么单独测 store】面板会随浮窗开关反复挂载/卸载，而会话状态活在 store 里。
 * 这里钉住三件事：动作把返回的会话写进快照、脏工程把**原始错误**交回调用方
 * （面板据此进入确认态）、断开清空会话。
 */
import { beforeEach, expect, test, vi } from "vitest";

const invokeMock = vi.fn();

vi.mock("../../services/invoke", () => ({
    invoke: (...args: unknown[]) => invokeMock(...args),
}));

import type { HostLinkSession } from "./hostLinkApi";
import { getHostLinkSnapshot, hostLinkActions, resetHostLinkStoreForTests } from "./hostLinkStore";

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
        audio: { complete: 3, partial: 0, extern: 0 },
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
    invokeMock.mockReset();
    resetHostLinkStoreForTests();
});

test("list 写入根目录与条目", async () => {
    invokeMock.mockResolvedValue({
        root: "/data/hifishifter",
        items: [{ directory: "/workspaces/one", hostLabel: "REAPER" }],
    });
    expect(await hostLinkActions.list()).toBeNull();
    const snap = getHostLinkSnapshot();
    expect(snap.root).toBe("/data/hifishifter");
    expect(snap.items).toHaveLength(1);
    expect(snap.busy).toBe(false);
    expect(invokeMock).toHaveBeenCalledWith("host_link_list", undefined);
});

test("open 成功写会话并给出状态词", async () => {
    invokeMock.mockResolvedValue(session());
    expect(await hostLinkActions.open("/workspaces/one", false)).toBeNull();
    const snap = getHostLinkSnapshot();
    expect(snap.session?.directory).toBe("/workspaces/one");
    expect(snap.status).toBe("host_link_status_opened");
    expect(invokeMock).toHaveBeenCalledWith("host_link_open", "/workspaces/one", false);
});

test("open 的脏工程错误原样交回调用方（面板据此进入确认态）", async () => {
    const failure = new Error("dirty_project: the current project has unsaved changes");
    invokeMock.mockRejectedValue(failure);
    const returned = await hostLinkActions.open("/workspaces/one", false);
    expect(returned).toBe(failure);
    expect(getHostLinkSnapshot().error).toBe(failure);
    expect(getHostLinkSnapshot().session).toBeNull();
});

test("refresh 写入保留条数，submit 写入结果，release 清空会话", async () => {
    invokeMock.mockResolvedValueOnce(session({ carriedParams: 2, carriedTracks: 1 }));
    await hostLinkActions.refresh(false);
    expect(getHostLinkSnapshot().session?.carriedParams).toBe(2);
    expect(getHostLinkSnapshot().status).toBe("host_link_status_refreshed");

    invokeMock.mockResolvedValueOnce(session({ applied: true, pending: false }));
    await hostLinkActions.submit();
    expect(getHostLinkSnapshot().session?.applied).toBe(true);
    expect(getHostLinkSnapshot().status).toBe("host_link_status_submitted");

    invokeMock.mockResolvedValueOnce({ ok: true, attached: false });
    await hostLinkActions.release();
    expect(getHostLinkSnapshot().session).toBeNull();
    expect(getHostLinkSnapshot().status).toBe("host_link_status_released");
});

test("clearError 只清展示用的错误", async () => {
    invokeMock.mockRejectedValue(new Error("dirty_project: unsaved"));
    await hostLinkActions.open("/workspaces/one", false);
    expect(getHostLinkSnapshot().error).not.toBeNull();
    hostLinkActions.clearError();
    expect(getHostLinkSnapshot().error).toBeNull();
});
