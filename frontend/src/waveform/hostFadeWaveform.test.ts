// 宿主普通/auto淡化元数据贯穿clip→共享波形场景→几何；只影响绘图，不生成处理音频。
import { expect, test } from "vitest";
import type { ClipInfo } from "../features/session/sessionTypes";
import type { HostFadeMetadata } from "../types/api";
import {
    clipToSceneClip,
    expandClipToTakeSceneClips,
} from "../components/layout/timeline/takeLanes";
import { createTimelineAxis } from "../components/layout/renderKernel/timelineAxis";
import { buildWaveformScene, type WaveformSceneClip } from "./sceneBuilder";
import { buildWaveformGeometry } from "./geometry";

const metadata: HostFadeMetadata = {
    curve_mode: "reaper_new",
    in_curvature: -0.6,
    in_s: 0.65,
    out_curvature: 0.4,
    out_s: -0.3,
};
/** 完整原GUI clip夹具；不用宿主工程、真实文件或模型加载伪装UI包络测试。 */
function clip(): ClipInfo {
    return {
        id: "clip",
        trackId: "track",
        name: "A",
        color: "blue",
        startSec: 0,
        lengthSec: 1,
        sourcePath: "source.wav",
        sourceStartSec: 0,
        sourceEndSec: 1,
        durationSec: 1,
        sourceSampleRate: 100,
        playbackRate: 1,
        reversed: false,
        loopEnabled: false,
        channelMode: 0,
        gain: 1,
        muted: false,
        snapOffsetSec: 0,
        fadeInSec: 0.1,
        fadeOutSec: 0.1,
        autoFadeInSec: 0.4,
        autoFadeOutSec: 0.3,
        fadeInShape: 6,
        fadeInDir: 1,
        fadeOutShape: 6,
        fadeOutDir: -1,
        hostFades: metadata,
    };
}
/** 用真实共享场景构建器验证长度/元数据投影，不在断言里重造生产投影。 */
function scene(source: WaveformSceneClip) {
    return buildWaveformScene({
        axis: createTimelineAxis({ pxPerSec: 100, scrollLeftPx: 0, viewportWidthPx: 100 }),
        widthPx: 100,
        rows: [{ topPx: 0, waveformTopPx: 0, waveformHeightPx: 100, clips: [source] }],
    });
}
/** 复制借用的顶点缓冲，避免下一次几何构建覆盖上一份断言数据。 */
function geometry(source: WaveformSceneClip): Float32Array {
    return buildWaveformGeometry({
        scene: scene(source),
        color: "rgba(255,255,255,1)",
        getPeaks: () => ({
            min: new Float32Array(100).fill(-1),
            max: new Float32Array(100).fill(1),
            dataStartSec: 0,
            dataDurationSec: 1,
            channels: 1,
        }),
    }).vertices.slice();
}
test("ordinary/auto fade length and axes survive the shared take/scene projection", () => {
    const source = clipToSceneClip(clip());
    expect(source).not.toBeNull();
    const projected = scene(source!);
    expect(projected.segments.length).toBeGreaterThan(0);
    projected.segments.forEach((segment) => {
        expect(segment.hostFades).toBe(metadata);
        expect(segment.fadeInSec).toBe(0.4);
        expect(segment.fadeOutSec).toBe(0.3);
    });
    const takes = {
        ...clip(),
        activeTakeId: "a",
        takes: [
            {
                id: "a",
                name: "A",
                sourcePath: "source.wav",
                sourceStartSec: 0,
                sourceEndSec: 1,
                gain: 1,
                playbackRate: 1,
                reversed: false,
                loopEnabled: false,
                channelMode: 0,
            },
            {
                id: "b",
                name: "B",
                sourcePath: "source.wav",
                sourceStartSec: 0,
                sourceEndSec: 1,
                gain: 1,
                playbackRate: 1,
                reversed: false,
                loopEnabled: false,
                channelMode: 0,
            },
        ],
    };
    const lanes = expandClipToTakeSceneClips(takes, true, 40);
    expect(lanes).not.toBeNull();
    lanes!.forEach((lane) => expect(lane.hostFades).toBe(metadata));
});
test("waveform consumes the same HFS axes, ignores deprecated shape and responds to updates", () => {
    const source = clipToSceneClip(clip())!;
    const curved = geometry(source);
    expect(curved.length).toBeGreaterThan(0);
    expect(
        geometry({ ...source, fadeInShape: 0, fadeInDir: 0, fadeOutShape: 0, fadeOutDir: 0 }),
    ).toEqual(curved);
    const linear = geometry({
        ...source,
        hostFades: { ...metadata, in_curvature: 0, in_s: 0, out_curvature: 0, out_s: 0 },
    });
    expect(linear).not.toEqual(curved);
    expect(geometry({ ...source, hostFades: { ...metadata, in_curvature: 0.8 } })).not.toEqual(
        curved,
    );
});
