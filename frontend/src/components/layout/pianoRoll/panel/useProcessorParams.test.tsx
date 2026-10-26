// @vitest-environment jsdom
/**
 * useProcessorParams — 参数描述符异步到达后，当前参数的取值范围/默认值必须重算。
 *
 * 【为什么需要这条回归】currentParamRange/currentParamDefaultValue 曾只依赖
 * editParam，却从 processorParamsRef.current 读描述符。ref 的写入不会触发
 * useMemo 重算，于是首次挂载或切换算法后，动态参数会拿到 undefined/旧范围，
 * 拖拽、双击复位与量化步长全部落到 0..1 的兜底值。
 */
import { act, useRef } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, expect, it, vi } from "vitest";

import type { ProcessorParamDescriptor } from "../../../../types/api";
import type { ValueViewport } from "../types";

const paramsApiMock = vi.hoisted(() => ({
    getProcessorParams: vi.fn(),
    getStaticParam: vi.fn(),
}));

vi.mock("../../../../services/api/params", () => ({ paramsApi: paramsApiMock }));

import { useProcessorParams } from "./useProcessorParams";

const PARAMS: ProcessorParamDescriptor[] = [
    {
        id: "breath_gain",
        display_name: "Breath",
        group: "pitch",
        kind: {
            type: "automation_curve",
            unit: "",
            default_value: 0.25,
            min_value: 0,
            max_value: 1,
        },
    },
];

function Probe() {
    const paramViewsRef = useRef<Record<string, ValueViewport>>({});
    const result = useProcessorParams({
        rootTrack: { pitchAnalysisAlgo: "nsf_hifigan_onnx" } as never,
        rootTrackId: "track-1",
        editParam: "breath_gain",
        paramViewsRef,
        invalidate: () => {},
        t: (key) => key,
    });
    const range = result.currentParamRange
        ? result.currentParamRange.min + ":" + result.currentParamRange.max
        : "none";
    return <div data-range={range} data-default={String(result.currentParamDefaultValue)} />;
}

afterEach(() => {
    document.body.innerHTML = "";
    vi.clearAllMocks();
});

it("recomputes range/default after descriptors arrive", async () => {
    let resolveParams!: (value: ProcessorParamDescriptor[]) => void;
    paramsApiMock.getProcessorParams.mockReturnValue(
        new Promise<ProcessorParamDescriptor[]>((resolve) => {
            resolveParams = resolve;
        }),
    );
    paramsApiMock.getStaticParam.mockResolvedValue({ ok: false });

    const container = document.createElement("div");
    document.body.appendChild(container);
    const root = createRoot(container);
    await act(async () => {
        root.render(<Probe />);
    });
    expect(container.querySelector("[data-range]")?.getAttribute("data-range")).toBe("none");

    await act(async () => {
        resolveParams(PARAMS);
        await Promise.resolve();
    });

    expect(container.querySelector("[data-range]")?.getAttribute("data-range")).toBe("0:1");
    expect(container.querySelector("[data-default]")?.getAttribute("data-default")).toBe("0.25");
    await act(async () => root.unmount());
});
