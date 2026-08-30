import {
    SNAP_OFFSET_HANDLE_SIZE_PX,
    SNAP_OFFSET_HIT_HEIGHT_PX,
    snapOffsetHandleXPx,
} from "../constants";

type VisibleTrack = {
    id: string;
    topPx: number;
};

type VisibleClip = {
    id: string;
    trackId: string;
    startSec: number;
    lengthSec: number;
    /** 吸附偏移（秒，相对 Clip 起点）：命中区跟随 ◣ 三角位置。 */
    snapOffsetSec?: number;
};

export type TimelineHitZone = "empty" | "body" | "trim_left" | "trim_right" | "snap_offset";

function compareVisibleClipRenderOrder(a: VisibleClip, b: VisibleClip): number {
    const delta = a.startSec - b.startSec;
    if (Math.abs(delta) > 1e-9) {
        return delta;
    }
    return String(a.id).localeCompare(String(b.id));
}

export function buildTimelineHitTestIndex(args: {
    rowHeight: number;
    pxPerSec: number;
    visibleTracks: VisibleTrack[];
    visibleClips: VisibleClip[];
}): {
    rowHeight: number;
    pxPerSec: number;
    tracksById: Map<string, VisibleTrack>;
    clipsByTrackId: Map<string, VisibleClip[]>;
} {
    return {
        rowHeight: args.rowHeight,
        pxPerSec: args.pxPerSec,
        tracksById: new Map(args.visibleTracks.map((track) => [track.id, track] as const)),
        clipsByTrackId: (() => {
            const grouped = new Map<string, VisibleClip[]>();

            for (const clip of args.visibleClips) {
                const next = grouped.get(clip.trackId);
                if (next) {
                    next.push(clip);
                } else {
                    grouped.set(clip.trackId, [clip]);
                }
            }

            for (const clips of grouped.values()) {
                clips.sort(compareVisibleClipRenderOrder);
            }

            return new Map(
                args.visibleTracks.map((track) => [track.id, grouped.get(track.id) ?? []]),
            );
        })(),
    };
}

export function hitTestTimeline(
    point: {
        screenX: number;
        screenY: number;
        scrollLeftPx: number;
        scrollTopPx: number;
    },
    index: ReturnType<typeof buildTimelineHitTestIndex>,
): {
    trackId: string | null;
    clipId: string | null;
    zone: TimelineHitZone;
} {
    const track = [...index.tracksById.values()].find((candidate) => {
        const topPx = candidate.topPx - point.scrollTopPx;
        return point.screenY >= topPx && point.screenY < topPx + index.rowHeight;
    });

    if (!track) {
        return {
            trackId: null,
            clipId: null,
            zone: "empty",
        };
    }

    // 行内局部 y（用于 SnapOffset 角部区判定：行底部条带）。
    const localY = point.screenY - (track.topPx - point.scrollTopPx);

    const worldSec = (point.scrollLeftPx + point.screenX) / Math.max(1e-9, index.pxPerSec);
    const clip = [...(index.clipsByTrackId.get(track.id) ?? [])]
        .reverse()
        .find(
            (candidate) =>
                worldSec >= candidate.startSec &&
                worldSec <= candidate.startSec + candidate.lengthSec,
        );

    if (!clip) {
        return {
            trackId: track.id,
            clipId: null,
            zone: "empty",
        };
    }

    const localLeftSec = worldSec - clip.startSec;

    // SnapOffset 命中区：跟随 ◣ 三角位置（三角 x 由 snapOffsetSec 直接换算，
    // **刻意不做宽度回退钳制** —— 越界由绘制端裁剪，见 constants.ts），横向
    // 取三角 ± 少量余量，纵向限行底部条带。优先于 trim/body：三角所在处归
    // 吸附偏移手势（对齐 REAPER 布局）。
    {
        const triX = snapOffsetHandleXPx(clip.snapOffsetSec, index.pxPerSec);
        const localXpx = localLeftSec * index.pxPerSec;
        if (
            localY >= index.rowHeight - SNAP_OFFSET_HIT_HEIGHT_PX &&
            localXpx >= triX - 4 &&
            localXpx <= triX + SNAP_OFFSET_HANDLE_SIZE_PX + 1
        ) {
            return {
                trackId: track.id,
                clipId: clip.id,
                zone: "snap_offset",
            };
        }
    }

    return {
        trackId: track.id,
        clipId: clip.id,
        zone:
            localLeftSec <= 0.08
                ? "trim_left"
                : clip.startSec + clip.lengthSec - worldSec <= 0.08
                  ? "trim_right"
                  : "body",
    };
}
