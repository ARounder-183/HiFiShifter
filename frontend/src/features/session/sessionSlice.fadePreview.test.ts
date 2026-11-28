import { expect, test } from "vitest";

import type { ClipInfo } from "./sessionTypes";
import type { HostFadeMetadata } from "../../types/api";
import sessionReducer, { setClipFades } from "./sessionSlice";
import { initialState } from "./sessionSlice/initialState";

const continuous: HostFadeMetadata = {
    curve_mode: "continuous",
    in_curvature: 0.5,
    out_curvature: 0.5,
    in_s: 0,
    out_s: 0,
};

function stateWith(hostFades?: HostFadeMetadata) {
    const clip = {
        id: "clip-1",
        trackId: "track_main",
        fadeInSec: 0,
        fadeOutSec: 0,
        fadeInShape: 1,
        fadeOutShape: 1,
        fadeInDir: 0,
        fadeOutDir: 0,
        fadeInS: 0,
        fadeOutS: 0,
        hostFades,
    } as unknown as ClipInfo;
    return { ...initialState, clips: [clip] };
}

test("continuous 上 dir/S 写入时同步镜像到 hostFades", () => {
    const next = sessionReducer(
        stateWith(continuous),
        setClipFades({ clipId: "clip-1", fadeInDir: 0.4, fadeInS: 0.2 }),
    );
    expect(next.clips[0].fadeInDir).toBe(0.4);
    expect(next.clips[0].hostFades?.in_curvature).toBe(0.4);
    expect(next.clips[0].hostFades?.in_s).toBe(0.2);
});

test("continuous 上形状 5/6 同时纠正本地 dir/S，下一次拖拽从预设轴起步", () => {
    const shaped = sessionReducer(
        stateWith(continuous),
        setClipFades({ clipId: "clip-1", fadeInShape: 5 }),
    );
    expect(shaped.clips[0].fadeInDir).toBe(0);
    expect(shaped.clips[0].fadeInS).toBe(0.5);
    expect(shaped.clips[0].hostFades?.in_curvature).toBe(0);
    expect(shaped.clips[0].hostFades?.in_s).toBe(0.5);

    const dragged = sessionReducer(shaped, setClipFades({ clipId: "clip-1", fadeInDir: -0.3 }));
    expect(dragged.clips[0].hostFades?.in_curvature).toBe(-0.3);
    expect(dragged.clips[0].hostFades?.in_s).toBe(0.5);
});

test("legacy 上任何 patch 都不改 hostFades", () => {
    const legacy: HostFadeMetadata = { ...continuous, curve_mode: "legacy" };
    const next = sessionReducer(
        stateWith(legacy),
        setClipFades({ clipId: "clip-1", fadeInShape: 6, fadeInDir: -0.4 }),
    );
    expect(next.clips[0].hostFades).toEqual(legacy);
    expect(next.clips[0].fadeInDir).toBe(-0.4);
    expect(next.clips[0].fadeInS).toBe(0);
});

test("仅长度 patch 不动轴值", () => {
    const next = sessionReducer(
        stateWith(continuous),
        setClipFades({ clipId: "clip-1", fadeInSec: 1 }),
    );
    expect(next.clips[0].hostFades).toEqual(continuous);
    expect(next.clips[0].fadeInDir).toBe(0);
    expect(next.clips[0].fadeInS).toBe(0);
});
