import React, { useMemo } from "react";
import { useSyncExternalStore } from "react";
import { useAppSelector } from "../../../app/hooks";
import { useAppTheme } from "../../../theme/AppThemeProvider";
import { type RootState } from "../../../app/store";
import {
    getSnapHighlightSnapshot,
    subscribeSnapHighlight,
    type SnapHighlightEntry,
    type SnapHighlightKind,
} from "../../../utils/snapHighlight";
import { NEW_TRACK_SENTINEL } from "./constants";

/**
 * 吸附竖线高亮层（轨道视图）。
 *
 * 订阅 snapHighlight 总线，把当前拖拽手势中生效的吸附渲染为竖线高亮：
 * - 吸附对象（目标）与被吸附对象（正在操作的对象）的吸附处同时高亮；
 * - 网格线/光标/采样率类目标为通栏竖线；Clip 类目标与被吸附 Clip 之间
 *   绘制一条跨越两者所在行的连线（“到底是吸附了哪个 Clip”），并在各自
 *   行内绘制贴边亮条；
 * - 不同吸附类型使用不同颜色（--qt-snap-* 主题变量，深浅色各一套）。
 *
 * 性能：总线快照不可变且只在 publish/clear 时替换引用；本组件为叶子，
 * useSyncExternalStore 按引用比较，拖拽期间每帧一次极小规模重渲染。
 */

/** 吸附类型 → 颜色主题变量。 */
const KIND_COLOR_VAR: Record<SnapHighlightKind, string> = {
    grid: "var(--qt-snap-grid)",
    clipStart: "var(--qt-snap-clip)",
    clipEnd: "var(--qt-snap-clip)",
    snapOffset: "var(--qt-snap-offset)",
    sourceStart: "var(--qt-snap-source)",
    sourceEnd: "var(--qt-snap-source)",
    selection: "var(--qt-snap-selection)",
    cursor: "var(--qt-snap-cursor)",
    sampleRate: "var(--qt-snap-sample)",
    loopBoundary: "var(--qt-snap-loop)",
};

export const SnapHighlightLayer: React.FC<{
    pxPerSec: number;
    rowHeight: number;
    /** 有序轨道列表（决定行的 y 坐标）。 */
    tracks: ReadonlyArray<{ id: string }>;
    contentHeight: number;
}> = ({ pxPerSec, rowHeight, tracks, contentHeight }) => {
    const snapshot = useSyncExternalStore(subscribeSnapHighlight, getSnapHighlightSnapshot);
    // 设置开关（吸附/网格设置 → 吸附总开关 → "显示吸附竖线高亮"）：
    // 关闭时整层不渲染。发布侧照常工作（开销可忽略），重新开启后下一次
    // 发布即恢复显示。
    const snapHighlightEnabled = useAppSelector(
        (state: RootState) => state.session.timelineSnap.snapHighlightEnabled !== false,
    );
    const rowIndexById = useMemo(() => {
        const map = new Map<string, number>();
        tracks.forEach((track, index) => map.set(track.id, index));
        return map;
    }, [tracks]);

    if (!snapHighlightEnabled || snapshot.entries.length === 0) return null;

    const safeRowHeight = Math.max(1, rowHeight);
    const safePxPerSec = Math.max(1e-9, pxPerSec);

    return (
        <div className="absolute inset-0 pointer-events-none z-[13] overflow-hidden">
            {snapshot.entries.map((entry) => (
                <SnapEntryGroup
                    key={entry.id}
                    entry={entry}
                    pxPerSec={safePxPerSec}
                    rowHeight={safeRowHeight}
                    rowIndexById={rowIndexById}
                    trackCount={tracks.length}
                    contentHeight={contentHeight}
                />
            ))}
        </div>
    );
};

const SnapEntryGroup: React.FC<{
    entry: SnapHighlightEntry;
    pxPerSec: number;
    rowHeight: number;
    rowIndexById: Map<string, number>;
    trackCount: number;
    contentHeight: number;
}> = ({ entry, pxPerSec, rowHeight, rowIndexById, trackCount, contentHeight }) => {
    // 光晕只适合暗底：浅色主题下吸附线/亮条用无光晕的纯色。
    const { mode: themeMode } = useAppTheme();
    const darkMode = themeMode === "dark";
    if (entry.markers.length === 0) return null;

    // ── 计算连线的纵向范围 ──
    // 任一 marker 无 trackId（网格/光标/采样率/未知轨道）→ 通栏；
    // 否则取所有 marker 所在行的并集：吸附对象行 ∪ 被吸附对象行。
    let fullHeight = false;
    let minTop = Number.POSITIVE_INFINITY;
    let maxBottom = Number.NEGATIVE_INFINITY;
    for (const marker of entry.markers) {
        if (marker.trackId == null) {
            fullHeight = true;
            continue;
        }
        let idx = rowIndexById.get(marker.trackId);
        if (idx == null && marker.trackId === NEW_TRACK_SENTINEL) {
            // 拖拽落新轨：ghost 行在现有轨道之下。
            idx = trackCount;
        }
        if (idx == null) {
            fullHeight = true;
            continue;
        }
        minTop = Math.min(minTop, idx * rowHeight);
        maxBottom = Math.max(maxBottom, (idx + 1) * rowHeight);
    }
    const rangeTop = fullHeight ? 0 : Math.max(0, minTop);
    const rangeBottom = fullHeight ? contentHeight : Math.min(contentHeight, maxBottom);
    const rangeHeight = Math.max(0, rangeBottom - rangeTop);
    const color = KIND_COLOR_VAR[entry.kind];

    // ── 去重后的竖线 x（同一 sec 只画一条主线）──
    const lineXs: number[] = [];
    for (const marker of entry.markers) {
        const x = marker.sec * pxPerSec;
        if (!lineXs.some((existing) => Math.abs(existing - x) < 0.5)) {
            lineXs.push(x);
        }
    }

    // ── 行内贴边亮条：按 (sec, trackId) 去重 ──
    const accents: Array<{ key: string; left: number; top: number; height: number }> = [];
    const accentSeen = new Set<string>();
    for (const marker of entry.markers) {
        if (marker.trackId == null) continue;
        let idx = rowIndexById.get(marker.trackId);
        if (idx == null && marker.trackId === NEW_TRACK_SENTINEL) {
            idx = trackCount;
        }
        if (idx == null) continue;
        const key = `${Math.round(marker.sec * pxPerSec * 2)}:${marker.trackId}`;
        if (accentSeen.has(key)) continue;
        accentSeen.add(key);
        accents.push({
            key,
            left: marker.sec * pxPerSec,
            top: idx * rowHeight + 2,
            height: Math.max(4, rowHeight - 4),
        });
    }

    return (
        <>
            {/* 主线：连接吸附对象与被吸附对象的吸附处 */}
            {rangeHeight > 0
                ? lineXs.map((x, index) => (
                      <div
                          key={`line-${index}`}
                          className="absolute"
                          style={{
                              left: x - 1,
                              top: rangeTop,
                              width: 2,
                              height: rangeHeight,
                              backgroundColor: color,
                              opacity: darkMode ? 0.85 : 0.7,
                              boxShadow: darkMode
                                  ? `0 0 6px 1px color-mix(in oklab, ${color} 60%, transparent)`
                                  : "none",
                          }}
                      />
                  ))
                : null}
            {/* 行内亮边：吸附处落在具体 Clip / 轨道行时的贴边高亮。
                光晕只保留在深色主题（暗底上光晕可读）；浅色主题下去掉光晕、
                用纯色细条，避免饱和色在浅底上糊成一片。 */}
            {accents.map((accent) => (
                <div
                    key={accent.key}
                    className="absolute rounded-[1px]"
                    style={{
                        left: accent.left - 1.5,
                        top: accent.top,
                        width: 3,
                        height: accent.height,
                        backgroundColor: darkMode
                            ? `color-mix(in oklab, ${color} 70%, white 30%)`
                            : color,
                        boxShadow: darkMode
                            ? `0 0 8px 2px color-mix(in oklab, ${color} 75%, transparent)`
                            : "none",
                    }}
                />
            ))}
        </>
    );
};
