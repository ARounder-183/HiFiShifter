/*
 * 拖拽落点判定 —— 纯几何。
 *
 * 渲染层（DockDropOverlay）与拖拽逻辑（useDockDrag）共用这里的函数：如果两边
 * 各算一次，"看到的高亮"与"实际落点"必然会在某些尺寸下漂移，而这类漂移用户
 * 一眼就能看出、却极难复现描述。
 *
 * 落点规则（对齐 REAPER / VEGAS 的直觉）：
 * - 指针在某个 Zone 的**中央区**（距四边都超过感应带）→ 并入该 Zone 的标签组；
 * - 否则落到**最近的那条边** → 在该边方向拆出新组；
 * - 指针不在任何 Zone 内 → 由调用方决定浮动。
 *
 * 分割产生的 Zone 矩形互不重叠（分割即划分），所以"指针落在哪个 Zone"最多
 * 只有一个答案，无需按面积排序去重。
 */

import type { DockDropZone, DockRect } from "./dockTypes";

export function pointInRect(rect: DockRect, x: number, y: number): boolean {
    return x >= rect.x && x <= rect.x + rect.w && y >= rect.y && y <= rect.y + rect.h;
}

/**
 * 判定指针在给定矩形内的落点。
 *
 * 感应带会被钳制到矩形短边的 40%：窄条（比如折叠后的标签条）如果按固定
 * 像素算感应带，四条带会互相吃掉，中央区消失，用户就再也无法"并入标签组"。
 */
export function resolveDropZone(
    rect: DockRect,
    pointer: { x: number; y: number },
    edgeBandPx: number,
): DockDropZone | null {
    if (rect.w <= 0 || rect.h <= 0) return null;
    if (!pointInRect(rect, pointer.x, pointer.y)) return null;

    const band = Math.max(8, Math.min(edgeBandPx, Math.min(rect.w, rect.h) * 0.4));

    const left = pointer.x - rect.x;
    const right = rect.x + rect.w - pointer.x;
    const top = pointer.y - rect.y;
    const bottom = rect.y + rect.h - pointer.y;

    const minH = Math.min(left, right);
    const minV = Math.min(top, bottom);

    if (minH > band && minV > band) return "center";
    if (minH <= minV) return left <= right ? "left" : "right";
    return top <= bottom ? "top" : "bottom";
}

/** 一个可停靠目标：Zone 及其当前视口矩形。 */
export interface DockZoneRect {
    zoneId: string;
    rect: DockRect;
}

/**
 * 从全部 Zone 中选出指针所在的那个。
 *
 * 矩形互不重叠，所以取首个命中即可；仍然做了"面积最小者优先"的兜底，
 * 以容忍将来引入重叠容器（如浮动 Zone 与停靠 Zone 同时参与判定）时的歧义。
 */
export function pickDropTarget(
    zones: readonly DockZoneRect[],
    pointer: { x: number; y: number },
): DockZoneRect | null {
    let best: DockZoneRect | null = null;
    for (const zone of zones) {
        if (!pointInRect(zone.rect, pointer.x, pointer.y)) continue;
        if (!best || zone.rect.w * zone.rect.h < best.rect.w * best.rect.h) best = zone;
    }
    return best;
}

/**
 * 把落点换算成"新组在目标组的哪一侧"。
 *
 * 与 `resolveDropZone` 分开是因为语义不同：`center` 不是"某一侧"，而是
 * "并入同一组"。调用方据此分派到 `{kind:"tab"}` 或 `{kind:"split"}`。
 */
export function dropZoneToSide(zone: DockDropZone): "left" | "right" | "top" | "bottom" | null {
    return zone === "center" || zone === "float" ? null : zone;
}

/**
 * 拖拽落点预览矩形（用于画半透明提示框）。
 *
 * 与真实落点同源，因此预览框与实际结果必然一致 —— 这是本模块存在的理由。
 */
export function dropPreviewRect(
    target: DockRect,
    zone: DockDropZone,
    splitterPx: number,
): DockRect | null {
    if (zone === "float") return null;
    if (zone === "center") return { ...target };

    const half = splitterPx / 2;
    switch (zone) {
        case "left":
            return { x: target.x, y: target.y, w: Math.max(0, target.w / 2 - half), h: target.h };
        case "right":
            return {
                x: target.x + target.w / 2 + half,
                y: target.y,
                w: Math.max(0, target.w / 2 - half),
                h: target.h,
            };
        case "top":
            return { x: target.x, y: target.y, w: target.w, h: Math.max(0, target.h / 2 - half) };
        default:
            return {
                x: target.x,
                y: target.y + target.h / 2 + half,
                w: target.w,
                h: Math.max(0, target.h / 2 - half),
            };
    }
}

/**
 * 浮动窗落点：夹紧到主窗口可用区内。
 *
 * 只保证"标题栏可见"而不是整个窗口可见 —— 允许窗体部分伸出视口是 DAW 的
 * 常见用法（把长面板推到边上），但标题栏必须留着，否则用户再也抓不回来。
 */
export function clampFloatRect(
    rect: DockRect,
    viewport: { w: number; h: number },
    titleBarPx: number,
): DockRect {
    const w = Math.min(rect.w, Math.max(200, viewport.w));
    const h = Math.min(rect.h, Math.max(120, viewport.h));
    const minVisible = 48;
    const maxX = viewport.w - minVisible;
    const minX = -(w - minVisible);
    const maxY = viewport.h - Math.max(titleBarPx, minVisible);
    const minY = 0;
    return {
        x: Math.round(Math.min(maxX, Math.max(minX, rect.x))),
        y: Math.round(Math.min(maxY, Math.max(minY, rect.y))),
        w: Math.round(w),
        h: Math.round(h),
    };
}

/**
 * 把浮动几何解析成具体矩形：带锚点时按**当前视口**推导位置。
 *
 * 锚点每帧按视口解析，因此主窗口缩放后浮窗仍在右下角；用户手动移动过之后锚点已
 * 被清除（见 `setFloatGeometry`），此处只是原样返回。
 */
export function resolveFloatRect(
    geometry: DockRect & { anchor?: string | null; anchorMarginPx?: number },
    viewport: { w: number; h: number },
): DockRect {
    if (geometry.anchor !== "bottom-right") {
        return { x: geometry.x, y: geometry.y, w: geometry.w, h: geometry.h };
    }
    const margin = geometry.anchorMarginPx ?? 24;
    return {
        // 视口比窗体还小时退回边距原点（宁可盖住内容，也不要跑到屏幕外）。
        x: Math.max(margin, Math.round(viewport.w - geometry.w - margin)),
        y: Math.max(margin, Math.round(viewport.h - geometry.h - margin)),
        w: geometry.w,
        h: geometry.h,
    };
}

/**
 * 浮动窗吸附：靠近视口边缘或其它浮动窗边时对齐。
 *
 * 返回吸附后的位置；`threshold` 为 0 时等价于不吸附（直接返回原值）。
 */
export function snapFloatPosition(
    position: { x: number; y: number; w: number; h: number },
    others: readonly DockRect[],
    viewport: { w: number; h: number },
    threshold: number,
): { x: number; y: number } {
    if (threshold <= 0) return { x: position.x, y: position.y };

    // 候选吸附线：视口边缘 + 其它窗体的边缘。
    const xs: number[] = [0, viewport.w];
    const ys: number[] = [0, viewport.h];
    for (const other of others) {
        xs.push(other.x, other.x + other.w, other.x - position.w, other.x + other.w - position.w);
        ys.push(other.y, other.y + other.h, other.y - position.h, other.y + other.h - position.h);
    }

    let bestX = position.x;
    let bestDx = threshold;
    for (const candidate of xs) {
        const delta = Math.abs(candidate - position.x);
        if (delta < bestDx) {
            bestDx = delta;
            bestX = candidate;
        }
    }

    let bestY = position.y;
    let bestDy = threshold;
    for (const candidate of ys) {
        const delta = Math.abs(candidate - position.y);
        if (delta < bestDy) {
            bestDy = delta;
            bestY = candidate;
        }
    }

    return { x: bestX, y: bestY };
}
