import { test } from "vitest";

import { buildSparseClipRenderModel } from "./timelineCanvasModel.js";
import { createTimelineAxis } from "./timelineAxis.js";

test("components/layout/timeline/runtime/timelineCanvasModel.test.ts scripted checks", async () => {
    function assertEqual(actual: unknown, expected: unknown, label: string): void {
        const actualJson = JSON.stringify(actual);
        const expectedJson = JSON.stringify(expected);
        if (actualJson !== expectedJson) {
            throw new Error(`${label}: expected ${expectedJson}, received ${actualJson}`);
        }
    }

    const model = buildSparseClipRenderModel({
        visibleTracks: [
            { id: "track-a", color: "#ff7a00" },
            { id: "track-b", color: "#00a3ff" },
        ],
        startTrackIndex: 0,
        visibleTrackClipsById: {
            "track-a": [
                {
                    id: "clip-a",
                    trackId: "track-a",
                    name: "Verse",
                    startSec: 2,
                    lengthSec: 3,
                    gain: 1,
                    playbackRate: 1.25,
                    muted: false,
                    fadeInSec: 0.25,
                    fadeOutSec: 0.5,
                    fadeInShape: 5,
                    fadeInDir: 0,
                    fadeOutShape: 5,
                    fadeOutDir: 0,
                },
                {
                    id: "clip-b",
                    trackId: "track-a",
                    name: "Fill",
                    startSec: 8,
                    lengthSec: 1,
                    gain: 0.5,
                    playbackRate: 0.8,
                    muted: true,
                    fadeInSec: 0,
                    fadeOutSec: 0,
                    fadeInShape: 0,
                    fadeInDir: 0,
                    fadeOutShape: 0,
                    fadeOutDir: 0,
                },
            ],
            "track-b": [
                {
                    id: "clip-c",
                    trackId: "track-b",
                    name: "Hook",
                    startSec: 4,
                    lengthSec: 2,
                    gain: 1,
                    playbackRate: 2,
                    muted: false,
                    fadeInSec: 0,
                    fadeOutSec: 0,
                    fadeInShape: 5,
                    fadeInDir: 0.25,
                    fadeOutShape: 2,
                    fadeOutDir: -0.25,
                },
            ],
        },
        axis: createTimelineAxis({ pxPerSec: 100, viewportWidthPx: 1000 }),
        rowHeight: 48,
        selectedClipId: "clip-b",
        multiSelectedClipIds: ["clip-c", "clip-b"],
        renamingClipId: "clip-a",
    });

    assertEqual(
        model.drawClips.map((clip) => ({
            id: clip.id,
            leftPx: clip.leftPx,
            topPx: clip.topPx,
            widthPx: clip.widthPx,
            fadeInPx: clip.fadeInPx,
            fadeOutPx: clip.fadeOutPx,
            trackColor: clip.trackColor,
            playbackRate: clip.playbackRate,
            selected: clip.selected,
            muted: clip.muted,
            isRenaming: clip.isRenaming,
        })),
        [
            {
                id: "clip-a",
                leftPx: 200,
                topPx: 0,
                widthPx: 300,
                fadeInPx: 25,
                fadeOutPx: 50,
                trackColor: "#ff7a00",
                playbackRate: 1.25,
                selected: false,
                muted: false,
                isRenaming: true,
            },
            {
                id: "clip-b",
                leftPx: 800,
                topPx: 0,
                widthPx: 100,
                fadeInPx: 0,
                fadeOutPx: 0,
                trackColor: "#ff7a00",
                playbackRate: 0.8,
                selected: true,
                muted: true,
                isRenaming: false,
            },
            {
                id: "clip-c",
                leftPx: 400,
                topPx: 48,
                widthPx: 200,
                fadeInPx: 0,
                fadeOutPx: 0,
                trackColor: "#00a3ff",
                playbackRate: 2,
                selected: true,
                muted: false,
                isRenaming: false,
            },
        ],
        "canvas keeps drawing overlay clips so visuals stay unified",
    );

    assertEqual(
        model.overlayClipIdsByTrackId,
        {
            "track-a": ["clip-a", "clip-b"],
            "track-b": ["clip-c"],
        },
        "sparse overlay ids",
    );
});

test("components/layout/timeline/runtime/timelineCanvasModel.test.ts 重叠检测：排序扫描 ≡ 全对比较", async () => {
    function assertEqual(actual: unknown, expected: unknown, label: string): void {
        const actualJson = JSON.stringify(actual);
        const expectedJson = JSON.stringify(expected);
        if (actualJson !== expectedJson) {
            throw new Error(`${label}: expected ${expectedJson}, received ${actualJson}`);
        }
    }

    /** 确定性伪随机（基准与回归必须可复现，禁用 Math.random）。 */
    function createRng(seed: number): () => number {
        let state = seed >>> 0;
        return () => {
            state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
            return state / 0x1_0000_0000;
        };
    }

    type TestClip = {
        id: string;
        trackId: string;
        name: string;
        startSec: number;
        lengthSec: number;
        gain: number;
        playbackRate: number;
        muted: boolean;
        fadeInSec: number;
        fadeOutSec: number;
        fadeInShape: number;
        fadeInDir: number;
        fadeOutShape: number;
        fadeOutDir: number;
    };

    function makeClip(id: string, trackId: string, startSec: number, lengthSec: number): TestClip {
        return {
            id,
            trackId,
            name: id,
            startSec,
            lengthSec,
            gain: 1,
            playbackRate: 1,
            muted: false,
            fadeInSec: 0,
            fadeOutSec: 0,
            fadeInShape: 0,
            fadeInDir: 0,
            fadeOutShape: 0,
            fadeOutDir: 0,
        };
    }

    /** 参照实现：原 O(n²) 全对比较，作为正确性基准。 */
    function bruteForceOverlapIds(clips: TestClip[]): string[] {
        const ids = new Set<string>();
        for (let i = 0; i < clips.length; i += 1) {
            const a = clips[i];
            for (let j = i + 1; j < clips.length; j += 1) {
                const b = clips[j];
                const aStart = a.startSec;
                const aEnd = aStart + a.lengthSec;
                const bStart = b.startSec;
                const bEnd = bStart + b.lengthSec;
                if (Math.min(aEnd, bEnd) > Math.max(aStart, bStart) + 1e-9) {
                    ids.add(a.id);
                    ids.add(b.id);
                }
            }
        }
        return [...ids].sort();
    }

    // 三种排布各测多组：
    // 1) 随机散布（大量重叠，含长 clip 跨多个短 clip 的退化情形）；
    // 2) 首尾相接（DAW 最常见排布，重叠对数 = 0，验证 break 不误判）；
    // 3) 完全重合（极端重叠）。
    const layouts: Array<{
        label: string;
        build: (rng: () => number, count: number) => TestClip[];
    }> = [
        {
            label: "random",
            build: (rng, count) =>
                Array.from({ length: count }, (_unused, index) =>
                    makeClip(
                        `c${index}`,
                        "t",
                        Math.round(rng() * 400) / 4,
                        Math.round(rng() * 200) / 4 + 0.25,
                    ),
                ),
        },
        {
            label: "back-to-back",
            build: (_rng, count) =>
                Array.from({ length: count }, (_unused, index) =>
                    makeClip(`c${index}`, "t", index * 3, 3),
                ),
        },
        {
            label: "identical",
            build: (_rng, count) =>
                Array.from({ length: count }, (_unused, index) =>
                    makeClip(`c${index}`, "t", 10, 5),
                ),
        },
    ];

    const axis = createTimelineAxis({ pxPerSec: 100, viewportWidthPx: 1000 });
    for (const layout of layouts) {
        for (let seed = 1; seed <= 40; seed += 1) {
            const clips = layout.build(createRng(seed), 24);
            const model = buildSparseClipRenderModel({
                visibleTracks: [{ id: "t" }],
                startTrackIndex: 0,
                visibleTrackClipsById: { t: clips },
                axis,
                rowHeight: 48,
                // 全部清空：让 overlay 集合只反映重叠检测，不掺入选中等其它来源。
                selectedClipId: null,
                multiSelectedClipIds: [],
                renamingClipId: null,
            });
            assertEqual(
                [...(model.overlayClipIdsByTrackId.t ?? [])].sort(),
                bruteForceOverlapIds(clips),
                `overlap ids · ${layout.label} · seed=${seed}`,
            );
        }
    }
});
