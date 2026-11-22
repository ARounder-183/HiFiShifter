// @vitest-environment jsdom
/**
 * 「REAPER 音高互转」执行的收尾契约。
 *
 * 【回归点一：撤销组必须关】**分批**执行时外层开一次 `begin_undo_group`；它会把会话的
 * `suppress_history` 置真。漏掉 `end_undo_group` 之后，整个工程**所有**参数编辑都不再
 * 压撤销点 —— 用户按 Ctrl+Z 会退到很久以前，而界面看起来一切正常。因此"失败也要关"
 * 是这条链路上最贵的一条不变量（同一个坑 `useClipPitchDrag` 已经踩过一次）。
 *
 * 【回归点二：单批不开组】不开组时命令自己会压一个撤销点；多开一层组等于给"什么都没
 * 写"的情况也留一步空撤销。
 */
import React from "react";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, expect, test, vi } from "vitest";

/**
 * 回执形状按**后端契约**声明，而不是让 `vi.fn()` 从第一次调用反推。
 *
 * 【为什么必须显式给】没有参数 / 返回类型时 `vi.fn()` 的 `calls` 是 `[]`（零元组），
 * `calls[0][0]` 与 `mockImplementationOnce` 都会被推断成"不可能存在"，于是
 * `npm run typecheck` 在测试全绿的情况下挂掉 —— 构建门禁里正是这么抓到的。
 */
const mocks = vi.hoisted(() => {
    type Result = { ok: boolean; clipsConverted?: number; error?: string };
    return {
        beginUndoGroup: vi.fn((_label?: string) => Promise.resolve({ ok: true })),
        endUndoGroup: vi.fn(() => Promise.resolve({ ok: true })),
        importHostTakePitch: vi.fn((_ids: string[]) =>
            Promise.resolve<Result>({ ok: true, clipsConverted: 1 }),
        ),
        exportHostTakePitch: vi.fn((_ids: string[]) =>
            Promise.resolve<Result>({ ok: true, clipsConverted: 1 }),
        ),
    };
});

vi.mock("../../services/webviewApi", () => ({
    webApi: {
        beginUndoGroup: mocks.beginUndoGroup,
        endUndoGroup: mocks.endUndoGroup,
        importHostTakePitch: mocks.importHostTakePitch,
        exportHostTakePitch: mocks.exportHostTakePitch,
    },
}));

import { paramBlockFeedbackStore } from "../../components/layout/pianoRoll/paramBlockFeedback";
import { I18nProvider } from "../../i18n/I18nProvider";
import { useHostPitchBridge, type PitchBridgeDirection } from "./useHostPitchBridge";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

type RunFn = (direction: PitchBridgeDirection, ids: readonly string[]) => Promise<void>;
let run: RunFn | null = null;

function Harness(): null {
    const { run: runBridge } = useHostPitchBridge();
    React.useEffect(() => {
        run = runBridge;
        return () => {
            run = null;
        };
    }, [runBridge]);
    return null;
}

let container: HTMLDivElement;
let root: Root;

const ids = (count: number) => Array.from({ length: count }, (_, index) => `clip-${index}`);

async function mount() {
    await act(async () => {
        root.render(
            <I18nProvider>
                <Harness />
            </I18nProvider>,
        );
    });
}

async function execute(direction: PitchBridgeDirection, clipIds: readonly string[]) {
    await act(async () => {
        await run!(direction, clipIds);
    });
}

beforeEach(async () => {
    for (const mock of Object.values(mocks)) mock.mockClear();
    mocks.importHostTakePitch.mockImplementation(() =>
        Promise.resolve({ ok: true, clipsConverted: 1 }),
    );
    mocks.exportHostTakePitch.mockImplementation(() =>
        Promise.resolve({ ok: true, clipsConverted: 1 }),
    );
    localStorage.setItem("hifishifter.locale", "en-US");
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
    await mount();
});

afterEach(() => {
    act(() => root.unmount());
    container.remove();
    localStorage.removeItem("hifishifter.locale");
});

test("a single batch needs no undo group: the command makes its own step", async () => {
    await execute("import", ids(3));
    expect(mocks.beginUndoGroup).not.toHaveBeenCalled();
    expect(mocks.endUndoGroup).not.toHaveBeenCalled();
    expect(mocks.importHostTakePitch).toHaveBeenCalledTimes(1);
    expect(mocks.importHostTakePitch.mock.calls[0]?.[0]).toEqual(ids(3));
});

test("more than one batch runs inside exactly one undo group", async () => {
    await execute("export", ids(300));
    expect(mocks.beginUndoGroup).toHaveBeenCalledTimes(1);
    expect(mocks.beginUndoGroup).toHaveBeenCalledWith("host_pitch_export");
    expect(mocks.endUndoGroup).toHaveBeenCalledTimes(1);
    expect(mocks.exportHostTakePitch).toHaveBeenCalledTimes(3);
    expect(mocks.exportHostTakePitch.mock.calls.map((call) => call[0].length)).toEqual([
        128, 128, 44,
    ]);
});

test("the undo group is closed even when a batch fails", async () => {
    mocks.importHostTakePitch.mockImplementationOnce(() =>
        Promise.resolve({ ok: false, error: "host refused" }),
    );
    await execute("import", ids(200));

    expect(mocks.endUndoGroup).toHaveBeenCalledTimes(1);
    expect(mocks.importHostTakePitch).toHaveBeenCalledTimes(1);
    const feedback = paramBlockFeedbackStore.getSnapshot();
    expect(feedback?.tone).toBe("danger");
    expect(feedback?.tooltip).toContain("host refused");
});

test("the undo group is closed when the transport itself throws", async () => {
    mocks.exportHostTakePitch.mockImplementationOnce(() => Promise.reject(new Error("ipc gone")));
    await execute("export", ids(200));

    expect(mocks.endUndoGroup).toHaveBeenCalledTimes(1);
    expect(paramBlockFeedbackStore.getSnapshot()?.tone).toBe("danger");
});

test("an empty scope reports instead of calling the backend", async () => {
    await execute("import", []);
    expect(mocks.importHostTakePitch).not.toHaveBeenCalled();
    expect(mocks.beginUndoGroup).not.toHaveBeenCalled();
    expect(paramBlockFeedbackStore.getSnapshot()?.tone).toBe("warning");
});
