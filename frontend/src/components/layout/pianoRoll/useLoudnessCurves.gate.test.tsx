// @vitest-environment jsdom
/*
 * ★ 回归：拉伸提交期间，响度快照的**中间态取数**不得落地。
 *
 * ## 缺陷形态
 *
 * 时间轴拉伸提交的后端顺序是「写新几何 → 改写用户曲线」，而写新几何的
 * fulfilled handler 必然 `applyTimelineState()` ⇒ `paramsEpoch++` ⇒ 立刻发起
 * 一次取数。这次取数发生在曲线改写**之前**，带回的是
 * 「**新几何的基线 × 旧范围**的用户曲线」这一自相矛盾组合。
 *
 * 它一旦落地：收尾判据（基线键变了 ⇔ 基线反映新几何）判定"权威数据已到"，
 * 撤下拖拽期的几何映射 —— 波形就此停在错乱状态，直到用户再做一次别的操作
 * 触发取数才恢复。这正是用户报告的「松手后波形完全对不上」。
 *
 * ## 本用例钉住什么
 *
 * 1. 闸门合上时，`paramsEpoch` 变化**不发起**取数；
 * 2. 闸门合上时，**在飞**的那次取数即使返回也**不落地**（丢弃，不是推迟）；
 * 3. 释放闸门 + 补一次 `paramsEpoch` ⇒ 恰好一次权威取数并落地。
 */
import { act, useEffect } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const { getParamFrames } = vi.hoisted(() => ({ getParamFrames: vi.fn() }));

vi.mock("../../../services/api", () => ({
    paramsApi: {
        getParamFrames: (...args: unknown[]) => getParamFrames(...args),
    },
}));

import {
    holdLoudnessFetch,
    isLoudnessFetchHeld,
    releaseLoudnessFetch,
} from "../timeline/loudnessFetchGate";
import { useLoudnessCurves, type LoudnessSnapshot } from "./useLoudnessCurves";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

/** 一次取数请求的挂起句柄（测试手动决定何时返回）。 */
interface PendingRequest {
    param: string;
    resolve: (value: unknown) => void;
}

let pending: PendingRequest[] = [];

/** 造一份最小载荷（含溯源键，模拟"新几何的基线"）。 */
function payload(args: {
    edit: number[];
    orig?: number[];
    key: string | null;
}): unknown {
    return {
        ok: true,
        edit: args.edit,
        orig: args.orig ?? [],
        frame_period_ms: 5,
        analysis_pending: false,
        ...(args.key === null ? {} : { dyn_orig_key: args.key }),
    };
}

/** 让所有挂起请求返回同一份数据（volume / dyn 各一份载荷）。 */
function resolveAll(snapshot: { volume: number[]; dyn: number[]; orig: number[]; key: string | null }) {
    const requests = pending;
    pending = [];
    for (const request of requests) {
        request.resolve(
            request.param === "volume"
                ? payload({ edit: snapshot.volume, key: snapshot.key })
                : payload({ edit: snapshot.dyn, orig: snapshot.orig, key: snapshot.key }),
        );
    }
}

/**
 * hook 返回值的捕获槽。
 *
 * 【为什么用"对象属性"而不是直接赋值一个模块级变量】`react-hooks/globals`
 * 门禁把"在渲染期给外部变量赋值"判为副作用。写对象属性不是重新绑定那个变量，
 * 且本文件只关心渲染后的快照，不参与渲染输出。
 */
const captureRef: { current: ReturnType<typeof useLoudnessCurves> | null } = { current: null };

let container: HTMLDivElement;
let root: Root;

/**
 * 探针组件：把 hook 返回值交给 `captureRef`。
 *
 * 【为什么把 ref 当 prop 传】模块级变量在渲染期赋值会被 `react-hooks/immutability`
 * 判为违规；写成"父层创建、子层写入"的 ref 传递则是标准形态。
 */
function Probe({
    captureRef: sinkRef,
    ...props
}: {
    captureRef: { current: ReturnType<typeof useLoudnessCurves> | null };
    rootTrackId: string | null;
    projectFrames: number;
    framePeriodMs: number;
    paramsEpoch: number;
    refreshToken: number;
}) {
    const value = useLoudnessCurves(props);
    // 在 effect 里写 ref（渲染期写 ref 会被 react-hooks/refs 判违规）；
    // 测试侧用 act() 包裹，effect 已刷新，读到的是最新一次渲染的返回值。
    useEffect(() => {
        sinkRef.current = value;
    });
    return null;
}

function render(props: {
    paramsEpoch: number;
    refreshToken?: number;
    rootTrackId?: string | null;
    projectFrames?: number;
}): void {
    act(() => {
        root.render(
            <Probe
                captureRef={captureRef}
                rootTrackId={props.rootTrackId ?? "track-1"}
                projectFrames={props.projectFrames ?? 400}
                framePeriodMs={5}
                paramsEpoch={props.paramsEpoch}
                refreshToken={props.refreshToken ?? 0}
            />,
        );
    });
}

/** 等待已发出的 promise 链跑完（不引入真实计时）。 */
async function settle(): Promise<void> {
    await act(async () => {
        await Promise.resolve();
        await Promise.resolve();
        await Promise.resolve();
    });
}

beforeEach(() => {
    releaseLoudnessFetch();
    pending = [];
    getParamFrames.mockReset();
    getParamFrames.mockImplementation(
        (_trackId: string, param: string) =>
            new Promise((resolve) => {
                pending.push({ param, resolve });
            }),
    );
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
    captureRef.current = null;
});

afterEach(() => {
    act(() => root.unmount());
    document.body.innerHTML = "";
    releaseLoudnessFetch();
});

describe("拉伸提交期闸门：中间态快照不得落地", () => {
    it("闸门合上 ⇒ paramsEpoch 变化不发起取数", async () => {
        render({ paramsEpoch: 0 });
        await settle();
        // 首次挂载正常发起一次。
        expect(getParamFrames).toHaveBeenCalledTimes(2);
        resolveAll({ volume: [1], dyn: [1], orig: [0.5], key: "k-old" });
        await settle();

        holdLoudnessFetch();
        const callsBefore = getParamFrames.mock.calls.length;
        // 几何落库的等价物：paramsEpoch 递增。
        render({ paramsEpoch: 1 });
        await settle();
        expect(getParamFrames.mock.calls.length).toBe(callsBefore);
    });

    it("★ 闸门合上期间在飞的取数返回 ⇒ 丢弃，不落地", async () => {
        render({ paramsEpoch: 0 });
        await settle();
        expect(getParamFrames).toHaveBeenCalledTimes(2);

        // 在飞期间合上闸门（模拟：落库派发早于曲线改写）。
        holdLoudnessFetch();
        expect(isLoudnessFetchHeld()).toBe(true);
        // 这份数据是「新几何的基线 × 旧曲线」—— 正是不得落地的那一份。
        resolveAll({ volume: [1], dyn: [1], orig: [0.9], key: "k-new" });
        await settle();

        expect(captureRef.current?.snapshot).toBeNull();
        expect(captureRef.current?.snapshotFetchSeq).toBe(0);
    });

    it("★ 释放闸门 + 补一次 epoch ⇒ 恰好一次权威取数并落地", async () => {
        render({ paramsEpoch: 0 });
        await settle();
        holdLoudnessFetch();
        resolveAll({ volume: [1], dyn: [1], orig: [0.9], key: "k-new" });
        await settle();
        expect(captureRef.current?.snapshot).toBeNull();

        // 曲线改写完成：打开闸门并补一次取数（TimelinePanel 的收尾链）。
        releaseLoudnessFetch();
        render({ paramsEpoch: 1 });
        await settle();
        expect(getParamFrames).toHaveBeenCalledTimes(4);
        resolveAll({ volume: [1, 0.5], dyn: [1, 1], orig: [0.9, 0.8], key: "k-new" });
        await settle();

        const snapshot = captureRef.current?.snapshot as LoudnessSnapshot | null;
        expect(snapshot).not.toBeNull();
        expect(snapshot?.baselineKey).toBe("k-new");
        expect(snapshot?.volume).toEqual([1, 0.5]);
        expect(captureRef.current?.snapshotFetchSeq).toBe(2);
    });

    it("对照：无闸门时取数正常落地（闸门不是常闭开关）", async () => {
        render({ paramsEpoch: 0 });
        await settle();
        resolveAll({ volume: [1], dyn: [1], orig: [0.5], key: "k1" });
        await settle();
        expect((captureRef.current?.snapshot as LoudnessSnapshot | null)?.baselineKey).toBe("k1");
    });
});
