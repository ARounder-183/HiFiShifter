import { test } from "vitest";

import { buildWaveformScene, type WaveformSceneClip } from "./sceneBuilder.ts";
import { createTimelineAxis } from "../components/layout/timeline/runtime/timelineAxis.ts";

test("waveform/sceneBuilder.test.ts scripted checks", async () => {
    function assertEqual(actual: unknown, expected: unknown, label: string): void {
        const actualJson = JSON.stringify(actual);
        const expectedJson = JSON.stringify(expected);
        if (actualJson !== expectedJson) {
            throw new Error(`${label}: expected ${expectedJson}, received ${actualJson}`);
        }
    }

    function clip(overrides: Partial<WaveformSceneClip> = {}): WaveformSceneClip {
        return {
            id: "clip",
            sourcePath: "/audio.wav",
            startSec: 0,
            lengthSec: 5,
            sourceStartSec: 0,
            sourceEndSec: 5,
            durationSec: 10,
            sourceSampleRate: 44100,
            playbackRate: 1,
            reversed: false,
            loopEnabled: false,
            gain: 1,
            muted: false,
            fadeInSec: 0,
            fadeOutSec: 0,
            fadeInShape: 0,
            fadeInDir: 0,
            fadeOutShape: 0,
            fadeOutDir: 0,
            ...overrides,
        };
    }

    {
        const scene = buildWaveformScene({
            axis: createTimelineAxis({
                pxPerSec: 100,
                scrollLeftPx: 1000,
                viewportWidthPx: 1000,
            }),
            widthPx: 1000,
            rows: [
                {
                    topPx: 0,
                    waveformTopPx: 18,
                    waveformHeightPx: 72,
                    clips: [clip({ startSec: 8, lengthSec: 5, sourceStartSec: 2 })],
                },
            ],
        });

        assertEqual(
            scene.segments.map((segment) => segment.screenRect),
            [{ x: 0, y: 18, width: 300, height: 72 }],
            "visible clips use viewport-local coordinates",
        );
        assertEqual(
            scene.segments.map((segment) => [segment.sourceStartSec, segment.sourceEndSec]),
            [[4, 7]],
            "viewport clipping advances the source interval",
        );
    }

    {
        const scene = buildWaveformScene({
            axis: createTimelineAxis({
                pxPerSec: 100,
                scrollLeftPx: 0,
                viewportWidthPx: 1000,
            }),
            widthPx: 1000,
            rows: [
                {
                    topPx: 0,
                    waveformTopPx: 0,
                    waveformHeightPx: 80,
                    clips: [
                        clip({
                            lengthSec: 3,
                            sourceStartSec: 99,
                            sourceEndSec: 9,
                            playbackRate: 2,
                            reversed: true,
                        }),
                    ],
                },
            ],
        });

        assertEqual(
            scene.segments.map((segment) => [segment.sourceStartSec, segment.sourceEndSec]),
            [[3, 9]],
            "reversed clips anchor their source window at sourceEndSec",
        );
        assertEqual(scene.segments[0]?.reversed, true, "reversed direction is retained");
    }

    {
        const scene = buildWaveformScene({
            axis: createTimelineAxis({
                pxPerSec: 10,
                scrollLeftPx: 0,
                viewportWidthPx: 150,
            }),
            widthPx: 150,
            rows: [
                {
                    topPx: 0,
                    waveformTopPx: 0,
                    waveformHeightPx: 80,
                    clips: [
                        clip({
                            lengthSec: 15,
                            sourceStartSec: 8,
                            sourceEndSec: 10,
                            loopEnabled: true,
                        }),
                    ],
                },
            ],
        });

        assertEqual(
            scene.segments.map((segment) => [segment.sourceStartSec, segment.sourceEndSec]),
            [
                [8, 10],
                [0, 10],
                [0, 3],
            ],
            "looped clips render the media tail followed by whole-media periods",
        );
        assertEqual(
            scene.markers.map((marker) => marker.timelineSec),
            [2, 12],
            "loop markers use the same period boundaries as waveform segments",
        );
    }

    {
        const scene = buildWaveformScene({
            axis: createTimelineAxis({
                pxPerSec: 100,
                scrollLeftPx: 0,
                viewportWidthPx: 500,
            }),
            widthPx: 500,
            rows: [
                {
                    topPx: 0,
                    waveformTopPx: 0,
                    waveformHeightPx: 80,
                    leadingOverlapSecByClipId: { clip: 2 },
                    clips: [
                        clip({
                            autoFadeInSec: 1,
                            fadeInSec: 4,
                            muted: true,
                        }),
                    ],
                },
            ],
        });

        assertEqual(
            scene.segments.map((segment) => [segment.screenRect.x, segment.screenRect.width]),
            [
                [0, 200],
                [200, 300],
            ],
            "leading overlap splits the scene at the exact timeline boundary",
        );
        assertEqual(
            scene.segments.map((segment) => segment.alpha),
            [0.2, 0.4],
            "muted overlap alpha is applied once per split segment",
        );
        assertEqual(scene.segments[0]?.fadeInSec, 1, "automatic fade overrides manual fade");
    }

    {
        const scene = buildWaveformScene({
            axis: createTimelineAxis({
                pxPerSec: 100,
                scrollLeftPx: 0,
                viewportWidthPx: 1000,
            }),
            widthPx: 1000,
            rows: [
                {
                    topPx: 40,
                    waveformTopPx: 18,
                    waveformHeightPx: 72,
                    clips: [
                        clip({
                            laneTopPx: 20,
                            laneHeightPx: 10,
                            inactive: true,
                        }),
                    ],
                },
            ],
        });

        assertEqual(
            scene.segments.map((segment) => segment.screenRect),
            [{ x: 0, y: 78, width: 500, height: 10 }],
            "take lanes override the row waveform band relative to the body top",
        );
        assertEqual(
            scene.segments.map((segment) => [segment.alpha, segment.inactive]),
            [[0.78, true]],
            "inactive lanes dim segment alpha and carry the flag to geometry",
        );
    }

    {
        const scene = buildWaveformScene({
            axis: createTimelineAxis({
                pxPerSec: 10,
                scrollLeftPx: 0,
                viewportWidthPx: 150,
            }),
            widthPx: 150,
            rows: [
                {
                    topPx: 0,
                    waveformTopPx: 18,
                    waveformHeightPx: 72,
                    clips: [
                        clip({
                            lengthSec: 15,
                            sourceStartSec: 8,
                            sourceEndSec: 10,
                            loopEnabled: true,
                            muted: true,
                            laneTopPx: 6,
                            laneHeightPx: 24,
                            inactive: true,
                        }),
                    ],
                },
            ],
        });

        assertEqual(
            scene.markers.map((marker) => [marker.yPx, marker.heightPx, marker.inactive]),
            [
                [24, 24, true],
                [24, 24, true],
            ],
            "loop markers follow the take lane band and carry the dim flag",
        );
        assertEqual(
            scene.segments.map((segment) => segment.alpha),
            [0.4 * 0.78, 0.4 * 0.78, 0.4 * 0.78],
            "muted and inactive dim factors compose multiplicatively",
        );
    }
});

/**
 * 窗口局部坐标 ≡ 视口坐标 + 平移（P2c 的核心不变量）。
 *
 * 背景：`WaveformSurface` 为了让几何能跨帧复用（平移只改一个 uniform），
 * 不再按「视口」构建场景，而是按「视口 + 余量」的**窗口**构建。实现上不
 * 改 `buildWaveformScene` 的签名，而是传一个派生 axis：
 * `scrollLeftPx = windowStartPx`、`viewportWidthPx = windowWidthPx`
 * —— 于是 `secToViewportPx()` 恰好产出 `contentPx − windowStart`，即窗口
 * 局部坐标。
 *
 * 不变量：窗口坐标系下的结果，必须等于视口坐标系下的结果整体平移
 * `(scrollLeft − windowStart, scrollTop − windowTop)`。这条一旦破了，波形
 * 相对 clip / 网格就会错位，因此用多组随机参数锁死。
 */
test("waveform/sceneBuilder.test.ts 窗口局部坐标 ≡ 视口坐标 + 平移", async () => {

    /**
     * 带容差的数值比较。
     *
     * 为什么不能用严格相等：视口场景算的是 `contentPx − scrollLeft`，窗口
     * 场景算的是 `contentPx − (scrollLeft − margin)`，两次减法的大数不同，
     * 结果可能差 1 个 ULP（实测约 1e-13 px）。这在渲染上完全不可见，严格
     * 逐位比较只会制造假阳性。
     */
    function close(a: unknown, b: unknown, tol: number): boolean {
        if (typeof a === "number" && typeof b === "number") return Math.abs(a - b) <= tol;
        if (Array.isArray(a) && Array.isArray(b)) {
            return a.length === b.length && a.every((v, i) => close(v, b[i], tol));
        }
        if (a && b && typeof a === "object" && typeof b === "object") {
            const ra = a as Record<string, unknown>;
            const rb = b as Record<string, unknown>;
            const keys = Object.keys(ra);
            return (
                keys.length === Object.keys(rb).length &&
                keys.every((key) => close(ra[key], rb[key], tol))
            );
        }
        return a === b;
    }
    function assertClose(actual: unknown, expected: unknown, label: string): void {
        if (!close(actual, expected, 1e-6)) {
            throw new Error(
                `${label}: expected ${JSON.stringify(expected)}, received ${JSON.stringify(actual)}`,
            );
        }
    }

    /** 确定性伪随机（回归必须可复现，禁用 Math.random）。 */
    function createRng(seed: number): () => number {
        let state = seed >>> 0;
        return () => {
            state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
            return state / 0x1_0000_0000;
        };
    }

    const base = {
        sourcePath: "/audio.wav",
        durationSec: 60,
        sourceSampleRate: 44100,
        playbackRate: 1,
        reversed: false,
        loopEnabled: false,
        gain: 1,
        muted: false,
        fadeInSec: 0,
        fadeOutSec: 0,
        fadeInShape: 0,
        fadeInDir: 0,
        fadeOutShape: 0,
        fadeOutDir: 0,
    };

    /** 实际比对到的 marker / 段数量，用于确认样本非空（防止测试空跑）。 */
    let comparedSegments = 0;
    let comparedMarkers = 0;

    for (let seed = 1; seed <= 30; seed += 1) {
        const rng = createRng(seed);
        const pxPerSec = 8 + rng() * 200;
        const viewportWidthPx = 600 + Math.round(rng() * 900);
        const scrollLeftPx = Math.round(rng() * 4000);
        const scrollTopPx = Math.round(rng() * 400);
        const rowHeight = 48 + Math.round(rng() * 80);
        const marginPx = Math.round(viewportWidthPx * 0.25);

        // 若干轨道行，每行若干 clip，起点覆盖视口内外以同时验证裁剪。
        const rows = Array.from({ length: 6 }, (_unused, rowIndex) => ({
            topPx: rowIndex * rowHeight,
            waveformTopPx: 18,
            waveformHeightPx: Math.max(1, rowHeight - 20),
            clips: Array.from({ length: 12 }, (_unused2, clipIndex) => {
                // 起点散布在视口内外的过渡带（视口前 20% 到后 40%），长度取
                // 视口的一小部分 —— 这样既有大量**完整落在视口内**的段可供
                // 严格比对，也有跨边界的段验证裁剪不参与比对。
                const viewportSec = viewportWidthPx / pxPerSec;
                const startSec = (scrollLeftPx + (rng() * 1.6 - 0.3) * viewportWidthPx) / pxPerSec;
                const lengthSec = viewportSec * (0.02 + rng() * 0.2);
                // 每 3 个构造一个「必定产生 loop marker」的 clip：媒体时长
                // 2 s、clip 长 25 s ⇒ head 2 s、周期 2 s，marker 落在
                // 2/4/…/24 s。
                // 两个坑都要避开：① 循环周期取 `resolveLoopMediaDurationSec`
                // （durationFrames → durationSec），**不是** sourceEnd−sourceStart，
                // 所以必须把 durationSec 也设成 2；② 周期必须远小于视口宽度
                // 换算的秒数，否则 marker 全落在可见区间之外。
                if (clipIndex % 3 === 0) {
                    return {
                        ...base,
                        id: `r${rowIndex}-c${clipIndex}`,
                        startSec,
                        lengthSec: 25,
                        durationSec: 2,
                        sourceStartSec: 0,
                        sourceEndSec: 2,
                        loopEnabled: true,
                    };
                }
                return {
                    ...base,
                    id: `r${rowIndex}-c${clipIndex}`,
                    startSec,
                    lengthSec,
                    sourceStartSec: rng() * 20,
                    sourceEndSec: 20 + rng() * 30,
                };
            }),
        }));

        // 改前语义：直接按视口构建。
        const viewportScene = buildWaveformScene({
            axis: createTimelineAxis({ pxPerSec, scrollLeftPx, viewportWidthPx }),
            widthPx: viewportWidthPx,
            viewportTopPx: scrollTopPx,
            rows,
        });

        // 新语义：按「视口 + 余量」的窗口构建（派生 axis）。
        const windowStartPx = scrollLeftPx - marginPx;
        const windowWidthPx = viewportWidthPx + 2 * marginPx;
        const windowTopPx = scrollTopPx - Math.max(0, rowHeight);
        const windowScene = buildWaveformScene({
            axis: createTimelineAxis({
                pxPerSec,
                scrollLeftPx: windowStartPx,
                scrollTopPx: windowTopPx,
                viewportWidthPx: windowWidthPx,
            }),
            widthPx: windowWidthPx,
            viewportTopPx: windowTopPx,
            rows,
        });

        // 窗口更宽 ⇒ 段数不少于视口场景（多出的是余量里被纳入的片段）。
        if (windowScene.segments.length < viewportScene.segments.length) {
            throw new Error(`seed=${seed}: window scene has fewer segments than viewport scene`);
        }

        // 逐段比对：窗口局部坐标 == 视口坐标 + 平移量。
        //
        // 只比对**完整落在视口内**的段：贴着视口边缘的段会被按边界裁剪，
        // 两个场景的裁剪边界不同（视口 vs 窗口），其矩形本就应该不同。
        // 完全落在内部的段不受任何裁剪影响，源区间与尺寸在两个场景里一致，
        // 只有平移量不同——这正是要锁的不变量。
        const offsetX = scrollLeftPx - windowStartPx;
        const offsetY = scrollTopPx - windowTopPx;
        const rect = (segment: {
            screenRect: { x: number; y: number; width: number; height: number };
        }) => ({
            x: segment.screenRect.x,
            y: segment.screenRect.y,
            width: segment.screenRect.width,
            height: segment.screenRect.height,
        });
        // 键必须带上 tile 的局部区间：loop clip 的每个周期 tile 都有相同的
        // sourceStart/End，只按它俩建键会让多个 tile 互相覆盖。
        const segmentKey = (segment: {
            clipId: string;
            sourceStartSec: number;
            sourceEndSec: number;
            clipLocalStartSec: number;
            clipLocalEndSec: number;
        }): string =>
            `${segment.clipId}@${segment.sourceStartSec}@${segment.sourceEndSec}@${segment.clipLocalStartSec}@${segment.clipLocalEndSec}`;
        const viewportBySource = new Map(
            viewportScene.segments.map((segment) => [segmentKey(segment), rect(segment)]),
        );
        for (const segment of windowScene.segments) {
            const translatedX = segment.screenRect.x - offsetX;
            // 被视口左右边界裁过的段跳过（它们的矩形在两场景中本就不同）。
            if (translatedX < -1e-6) continue;
            if (translatedX + segment.screenRect.width > viewportWidthPx + 1e-6) continue;
            const expected = viewportBySource.get(segmentKey(segment));
            if (expected === undefined) continue;
            comparedSegments += 1;
            assertClose(
                {
                    x: translatedX,
                    y: segment.screenRect.y - offsetY,
                    width: segment.screenRect.width,
                    height: segment.screenRect.height,
                },
                expected,
                `seed=${seed} clip=${segment.clipId} window-local ≡ viewport + offset`,
            );
        }

        // marker 同样只差一个平移。
        for (const marker of windowScene.markers) {
            const expected = viewportScene.markers.find(
                (candidate) =>
                    candidate.clipId === marker.clipId &&
                    Math.abs(candidate.timelineSec - marker.timelineSec) < 1e-9,
            );
            if (expected === undefined) continue;
            comparedMarkers += 1;
            assertClose(
                { x: marker.xPx - offsetX, y: marker.yPx - offsetY },
                { x: expected.xPx, y: expected.yPx },
                `seed=${seed} marker ${marker.clipId} window-local ≡ viewport + offset`,
            );
        }
    }

    // 防止测试空跑：段与 marker 都必须有实际比对样本（marker 分支极易因
    // 随机参数不巧而一个都不产生）。
    if (comparedSegments < 100) {
        throw new Error(`compared segment samples too few: ${comparedSegments}`);
    }
    if (comparedMarkers < 1) {
        throw new Error("compared marker samples too few: 0");
    }
});
