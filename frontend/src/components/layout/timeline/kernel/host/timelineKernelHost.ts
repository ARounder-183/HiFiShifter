/**
 * 时间轴渲染内核 · Spike 宿主（命令式运行时）
 *
 * 【主要内容】
 * 把内核各模块装配成一个可运行的宿主：创建 WebGL2 上下文与两个 program、字形图集
 * 与布局器、滚动内核与渲染循环；绑定 wheel / 滚动条拖拽 / 键盘输入；在 rAF 内按层序
 * 绘制「网格 → clip 块面 → 文字」；并在内容 / 缩放变化时重建几何。
 *
 * 【作用】
 * 这是「自绘滚动 + 单 WebGL2」的最小可运行验证载体：
 * - 没有任何随原生滚动移动的 DOM 内容层 → 不需要同帧同步提交，渲染完全由 rAF 调度；
 * - 几何以内容坐标常驻 GPU → 滚动帧只更新 `u_viewOrigin`（`repaint`，零重建）。
 * 真机用它跑 PERF 场景，验证帧率与手感，作为是否推进阶段 1 的决策依据。
 *
 * 【与其他模块的关系】
 * - 上游：React 外壳（`TimelineKernelSpikeView`）提供 DOM 节点与数据镜像。
 * - 复用（Spike 的明确例外）：`runtime/timelineAxis`（坐标投影）、
 *   `runtime/buildTimelineTicks`（刻度）、`runtime/timelineCanvasModel`（clip 几何）、
 *   `runtime/timelineCanvasStyle`（样式 / 字体）。
 * - 独立性：纯 TS（无 React）；GL 资源与事件监听在 `dispose` 中全部释放。
 *
 * 【Spike 已知简化（记录于 Spike 报告）】
 * 1. 只画网格 / clip 块面 / clip 名称文字；细节层（旋钮 / 徽标 / fade 曲线 / 波形 /
 *    播放头 / 标尺 / 轨道头）尚未接入。
 * 2. 字形图集单页（`maxPages: 1`）；页满后新字形不再生成（跳过绘制）。
 * 3. 主题色取固定前景色，未接入主题变量。
 * 4. clip 数据直接来自 session（字段结构与 `buildSparseClipRenderModel` 的入参兼容），
 *    未接入选区 / 悬停 / 重叠等交互态。
 */

import type { ClipInfo, TrackInfo } from "../../../../../features/session/sessionTypes";
import { buildTimelineTicks, type TimelineTick } from "../../runtime/buildTimelineTicks";
import { createTimelineAxis, type TimelineAxis } from "../../runtime/timelineAxis";
import { buildSparseClipRenderModel } from "../../runtime/timelineCanvasModel";
import { resolveFontFamily } from "../../runtime/timelineCanvasStyle";
import { createGlyphLayout } from "../glyph/glyphLayout";
import { createGlyphRasterizer, GLYPH_LINE_HEIGHT_RATIO } from "../glyph/glyphRasterizer";
import { createGlCanvas } from "../gl/glContext";
import { buildGlyphQuads, type GlyphQuad } from "../gl/glyphQuads";
import { createGlyphProgram } from "../gl/glyphProgram";
import { CLIP_INSTANCE_FLOATS, writeFlatInstance } from "../gl/instanceLayout";
import { createSdfBoxProgram } from "../gl/sdfBoxProgram";
import { computeScrollbar, scrollDeltaFromThumbDrag } from "../input/scrollbars";
import { normalizeWheelDelta } from "../input/normalizeWheel";
import { createRenderLoop } from "../renderLoop";
import { createClipInstanceBuilder } from "../scene/clipInstances";
import { buildGridInstances } from "../scene/gridInstances";
import { createScrollKernel } from "../scrollKernel";

/** `buildTimelineTicks` 的入参类型（用于让数据镜像的字段类型自动对齐）。 */
type BuildTicksArgs = Parameters<typeof buildTimelineTicks>[0];

/** 宿主渲染所需的数据镜像（由 React 外壳提供，每次 render 更新）。 */
export interface TimelineKernelData {
    readonly tracks: readonly TrackInfo[];
    readonly clips: readonly ClipInfo[];
    /** 工程总时长（秒）。 */
    readonly projectSec: number;
    readonly darkMode: boolean;
    readonly bpm: number;
    readonly beatsPerBar: number;
    readonly grid: BuildTicksArgs["grid"];
    readonly primaryTimeUnit: BuildTicksArgs["primaryUnit"];
    readonly secondaryTimeUnit: BuildTicksArgs["secondaryUnit"];
    readonly minLabelSpacingPx: number;
    readonly tempoMap: BuildTicksArgs["tempoMap"];
}

/** 宿主构造参数。 */
export interface TimelineKernelHostArgs {
    /** 宿主容器（尺寸来源，`ResizeObserver` 目标）。 */
    readonly container: HTMLElement;
    /** 绘制画布（WebGL2）。 */
    readonly canvas: HTMLCanvasElement;
    /** 水平滚动条 thumb 元素。 */
    readonly hScrollbarThumb: HTMLElement;
    /** 竖直滚动条 thumb 元素。 */
    readonly vScrollbarThumb: HTMLElement;
    /** 数据镜像读取函数（每帧调用，避免 React 与 runtime 相互持有）。 */
    readonly data: () => TimelineKernelData;
}

/** 宿主句柄。 */
export interface TimelineKernelHost {
    /** 标记场景需要重建（内容 / 缩放 / 主题变化后调用）。 */
    invalidateScene(): void;
    /** 释放全部资源与事件监听。 */
    dispose(): void;
}

/** 横向构建余量（CSS px）：滚动在此范围内不重建几何。 */
const HORIZONTAL_MARGIN_PX = 512;

/** 竖直 overscan 行数：轨道窗口上下各多建若干行。 */
const VERTICAL_OVERSCAN_ROWS = 2;

/** clip 名称字号（CSS px）。 */
const CLIP_NAME_FONT_SIZE_PX = 11;

/** clip 名称内边距（CSS px）。 */
const CLIP_NAME_PADDING_X_PX = 4;
const CLIP_NAME_PADDING_Y_PX = 2;

/** 缩放上下限（与生产常量同量级）。 */
const MIN_PX_PER_SEC = 0.5;
const MAX_PX_PER_SEC = 8000;

/** 初始缩放与行高。 */
const INITIAL_PX_PER_SEC = 100;
const INITIAL_ROW_HEIGHT = 80;

/** 键盘单步滚动量（CSS px）。 */
const KEYBOARD_STEP_PX = 60;

/**
 * 创建内核宿主。
 *
 * 流程：
 * 1. 建 GL 上下文与 program（失败抛错，由 React 外壳展示回退提示）；
 * 2. 建滚动内核（内容尺寸 / 轨道数 / 视口高从数据镜像与尺寸现读）；
 * 3. 建渲染循环并订阅滚动变更；
 * 4. 绑定输入（wheel / 滚动条 / 键盘 / 尺寸变化）；
 * 5. 首次标脏。
 *
 * @param args 宿主参数。
 * @returns 宿主句柄；调用方须在卸载时 `dispose()`。
 */
export function createTimelineKernelHost(args: TimelineKernelHostArgs): TimelineKernelHost {
    const { container, canvas, hScrollbarThumb, vScrollbarThumb, data } = args;

    const glCanvas = createGlCanvas(canvas);
    if (!glCanvas) throw new Error("WebGL2 不可用");
    const gl = glCanvas.gl;
    const sdfBox = createSdfBoxProgram(gl);
    const glyphProgram = createGlyphProgram(gl);

    const readDpr = () => window.devicePixelRatio || 1;
    const rasterizer = createGlyphRasterizer({ pageSizePx: 2048, maxPages: 1, dpr: readDpr() });
    const glyphLayout = createGlyphLayout(
        (text, fontKey) => rasterizer?.measure(text, fontKey) ?? 0,
    );
    const clipBuilder = createClipInstanceBuilder();

    // 尺寸镜像：由 ResizeObserver 维护，供滚动内核与光栅化读取（O(1)，不触发布局）。
    let viewportWidthPx = Math.max(1, container.clientWidth);
    let viewportHeightPx = Math.max(1, container.clientHeight);

    const scroll = createScrollKernel({
        pxPerSec: INITIAL_PX_PER_SEC,
        rowHeight: INITIAL_ROW_HEIGHT,
        projectSec: () => Math.max(0, data().projectSec),
        trackCount: () => data().tracks.length,
        viewportHeightPx: () => viewportHeightPx,
        minPxPerSec: MIN_PX_PER_SEC,
        maxPxPerSec: MAX_PX_PER_SEC,
    });

    // ── 场景缓存（内容坐标常驻；滚动在余量内不重建）─────────────────────
    let combinedInstances = new Float32Array(0);
    let combinedCount = 0;
    let glyphQuads: GlyphQuad[] = [];
    let sceneDirty = true;
    let builtPxPerSec = Number.NaN;
    let builtScrollLeftPx = Number.NaN;
    let builtDarkMode = !data().darkMode;
    let builtSignature = "";

    /** 组装当前 axis（内容坐标投影）。 */
    function currentAxis(): TimelineAxis {
        const view = scroll.get();
        return createTimelineAxis({
            pxPerSec: view.pxPerSec,
            scrollLeftPx: view.scrollLeft,
            scrollTopPx: view.scrollTop,
            viewportWidthPx,
            dpr: readDpr(),
        });
    }

    /** 场景签名：内容 / 主题变化时快速失效（避免逐字段比较）。 */
    function sceneSignature(d: TimelineKernelData): string {
        return `${d.clips.length}|${d.tracks.length}|${d.projectSec}|${d.grid}|${d.bpm}|${d.beatsPerBar}`;
    }

    /** 重建网格 + clip 实例（写入同一个合并缓冲：网格在前、clip 在后）。 */
    function rebuildInstances(axis: TimelineAxis): void {
        const d = data();
        const view = scroll.get();
        const windowLeft = Math.max(0, view.scrollLeft - HORIZONTAL_MARGIN_PX);
        const windowWidth = viewportWidthPx + HORIZONTAL_MARGIN_PX * 2;

        // 竖直窗口：只构建可见轨道行（含 overscan）。
        const visibleRowCount = Math.max(1, Math.ceil(viewportHeightPx / view.rowHeight));
        const firstRow = Math.max(
            0,
            Math.floor(view.scrollTop / view.rowHeight) - VERTICAL_OVERSCAN_ROWS,
        );
        const lastRow = Math.min(
            d.tracks.length,
            firstRow + visibleRowCount + VERTICAL_OVERSCAN_ROWS * 2,
        );
        const visibleTracks = d.tracks.slice(firstRow, Math.max(firstRow, lastRow));

        const ticks: TimelineTick[] = buildTimelineTicks({
            axis,
            bpm: d.bpm,
            beatsPerBar: d.beatsPerBar,
            grid: d.grid,
            primaryUnit: d.primaryTimeUnit,
            secondaryUnit: d.secondaryTimeUnit,
            minLabelSpacingPx: d.minLabelSpacingPx,
            tempoMap: d.tempoMap,
        });

        const gridInstances = buildGridInstances({
            ticks,
            windowLeftPx: windowLeft,
            windowWidthPx: windowWidth,
            contentBottomPx: d.tracks.length * view.rowHeight,
            dpr: axis.dpr,
            weakRgba: d.darkMode ? [1, 1, 1, 0.08] : [0, 0, 0, 0.06],
            strongRgba: d.darkMode ? [1, 1, 1, 0.16] : [0, 0, 0, 0.12],
        });

        // clip 几何：复用既有稀疏模型（窗口裁剪 + 内容坐标投影）。
        const clipsByTrackId: Record<string, ClipInfo[]> = {};
        for (const track of visibleTracks) clipsByTrackId[track.id] = [];
        const safePxPerSec = Math.max(1e-9, view.pxPerSec);
        const bufferSec = Math.max(1.5, HORIZONTAL_MARGIN_PX / safePxPerSec);
        const startSec = Math.max(0, view.scrollLeft / safePxPerSec - bufferSec);
        const endSec = (view.scrollLeft + viewportWidthPx) / safePxPerSec + bufferSec;
        for (const clip of d.clips) {
            const list = clipsByTrackId[clip.trackId];
            if (list === undefined) continue;
            if (clip.startSec + clip.lengthSec < startSec || clip.startSec > endSec) continue;
            list.push(clip);
        }

        const model = buildSparseClipRenderModel({
            visibleTracks: visibleTracks.map((track) => ({ id: track.id, color: track.color })),
            startTrackIndex: firstRow,
            // ClipInfo 与 SparseRenderClip 的字段结构兼容（缺省字段由模型内部兜底）。
            visibleTrackClipsById: clipsByTrackId as never,
            axis,
            rowHeight: view.rowHeight,
            selectedClipId: null,
            multiSelectedClipIds: [],
            renamingClipId: null,
        });

        const clipResult = clipBuilder.build({
            clips: model.drawClips,
            darkMode: d.darkMode,
            fontFamily: resolveFontFamily(),
            seamColor: d.darkMode ? "rgb(31, 31, 31)" : "rgb(237, 240, 245)",
        });

        // 合并：网格实例在前、clip 实例在后（同一 draw call 内的层叠顺序）。
        const gridCount = gridInstances.length;
        const needed = (gridCount + clipResult.count) * CLIP_INSTANCE_FLOATS;
        if (combinedInstances.length < needed) combinedInstances = new Float32Array(needed);
        for (let index = 0; index < gridCount; index += 1) {
            writeFlatInstance(combinedInstances, index, gridInstances[index]);
        }
        combinedInstances.set(
            clipResult.instances.subarray(0, clipResult.count * CLIP_INSTANCE_FLOATS),
            gridCount * CLIP_INSTANCE_FLOATS,
        );
        combinedCount = gridCount + clipResult.count;

        // 文字：clip 名称（超宽截断由布局器完成）。
        const fontKey = `${CLIP_NAME_FONT_SIZE_PX}px ${resolveFontFamily()}`;
        const textRgba: readonly [number, number, number, number] = d.darkMode
            ? [0.93, 0.96, 1, 0.95]
            : [0.12, 0.16, 0.22, 0.95];
        const quads: GlyphQuad[] = [];
        for (const clip of model.drawClips) {
            const maxWidth = clip.widthPx - CLIP_NAME_PADDING_X_PX * 2;
            if (maxWidth <= 0 || clip.name.length === 0) continue;
            const run = glyphLayout.layout(clip.name, fontKey, maxWidth);
            if (run.glyphs.length === 0) continue;
            quads.push(
                ...buildGlyphQuads({
                    glyphs: run.glyphs,
                    originX: clip.leftPx + CLIP_NAME_PADDING_X_PX,
                    originY: clip.topPx + CLIP_NAME_PADDING_Y_PX,
                    heightPx: CLIP_NAME_FONT_SIZE_PX * GLYPH_LINE_HEIGHT_RATIO,
                    atlasPageSizePx: rasterizer?.pageSizePx() ?? 1,
                    resolveSlot: (char) => rasterizer?.acquire(char, fontKey) ?? null,
                    rgba: textRgba,
                }),
            );
        }
        glyphQuads = quads;

        // 新字形写入图集后按页上传纹理。
        const dirtyPages = rasterizer?.consumeDirtyPages() ?? [];
        if (rasterizer !== null && dirtyPages.length > 0) {
            for (const page of dirtyPages) {
                const pixels = rasterizer.readPage(page);
                if (pixels !== null) glyphProgram.uploadAtlas(pixels, rasterizer.pageSizePx());
            }
        }

        builtPxPerSec = view.pxPerSec;
        builtScrollLeftPx = view.scrollLeft;
        builtDarkMode = d.darkMode;
        builtSignature = sceneSignature(d);
        sceneDirty = false;
    }

    /** 判断是否需要重建几何。 */
    function ensureScene(axis: TimelineAxis): void {
        const view = scroll.get();
        const d = data();
        const needsRebuild =
            sceneDirty ||
            builtPxPerSec !== view.pxPerSec ||
            builtDarkMode !== d.darkMode ||
            builtSignature !== sceneSignature(d) ||
            Math.abs(view.scrollLeft - builtScrollLeftPx) > HORIZONTAL_MARGIN_PX;
        if (needsRebuild) rebuildInstances(axis);
    }

    /** 更新自绘滚动条 thumb 的几何（每帧一次）。 */
    function updateScrollbars(): void {
        const view = scroll.get();
        const horizontal = computeScrollbar({
            contentSizePx: view.pxPerSec * Math.max(0, data().projectSec),
            viewportSizePx: viewportWidthPx,
            scrollPx: view.scrollLeft,
            maxScrollPx: scroll.maxScrollLeft(),
        });
        hScrollbarThumb.style.width = `${Math.max(0, horizontal.thumbLengthPx)}px`;
        hScrollbarThumb.style.transform = `translateX(${Math.max(0, horizontal.thumbStartPx)}px)`;
        hScrollbarThumb.style.display = horizontal.scrollable ? "block" : "none";

        const vertical = computeScrollbar({
            contentSizePx: data().tracks.length * view.rowHeight,
            viewportSizePx: viewportHeightPx,
            scrollPx: view.scrollTop,
            maxScrollPx: scroll.maxScrollTop(),
        });
        vScrollbarThumb.style.height = `${Math.max(0, vertical.thumbLengthPx)}px`;
        vScrollbarThumb.style.transform = `translateY(${Math.max(0, vertical.thumbStartPx)}px)`;
        vScrollbarThumb.style.display = vertical.scrollable ? "block" : "none";
    }

    /** 绘制一帧：清屏 → 网格 + clip（一次 draw call）→ 文字。 */
    function draw(): void {
        const view = scroll.get();
        const target = glCanvas!.resize(viewportWidthPx, viewportHeightPx, readDpr());
        glCanvas!.clear();
        const axis = currentAxis();
        ensureScene(axis);
        if (combinedCount > 0) {
            sdfBox.render(
                combinedInstances,
                combinedCount,
                target,
                view.scrollLeft,
                view.scrollTop,
            );
        }
        if (glyphQuads.length > 0) {
            glyphProgram.render(glyphQuads, target, view.scrollLeft, view.scrollTop);
        }
        updateScrollbars();
    }

    const loop = createRenderLoop({ draw });
    loop.start();
    const unsubscribeScroll = scroll.subscribe(() => loop.invalidate());

    // ── 输入：wheel（滚动 / Ctrl 缩放）───────────────────────────────
    function onWheel(event: WheelEvent): void {
        const rect = container.getBoundingClientRect();
        const ctx = { lineHeightPx: 16, pageHeightPx: Math.max(1, rect.height) };
        const deltaY = normalizeWheelDelta(event.deltaY, event.deltaMode, ctx);
        const deltaX = normalizeWheelDelta(event.deltaX, event.deltaMode, ctx);
        event.preventDefault();
        if (event.ctrlKey || event.metaKey) {
            const factor = deltaY < 0 ? 1.1 : 0.9;
            const view = scroll.get();
            scroll.setZoom(view.pxPerSec * factor, event.clientX - rect.left);
            return;
        }
        const view = scroll.get();
        if (Math.abs(deltaX) > 0.01) scroll.setScrollLeft(view.scrollLeft + deltaX);
        if (Math.abs(deltaY) > 0.01) scroll.setScrollTop(view.scrollTop + deltaY);
    }
    container.addEventListener("wheel", onWheel, { passive: false });

    // ── 输入：滚动条拖拽 ─────────────────────────────────────────────
    let dragAxis: "x" | "y" | null = null;
    let dragStartPointer = 0;
    let dragStartScroll = 0;

    /** 造一个 thumb 拖拽的按下处理器。 */
    function makeThumbPointerDown(axis: "x" | "y") {
        return (event: PointerEvent) => {
            event.preventDefault();
            event.stopPropagation();
            dragAxis = axis;
            dragStartPointer = axis === "x" ? event.clientX : event.clientY;
            dragStartScroll = axis === "x" ? scroll.get().scrollLeft : scroll.get().scrollTop;
            (event.currentTarget as HTMLElement).setPointerCapture?.(event.pointerId);
        };
    }

    /** 把拖拽位移换算为滚动位置增量并提交。 */
    function applyThumbDrag(pointerDeltaPx: number): void {
        const view = scroll.get();
        if (dragAxis === "x") {
            const geometry = computeScrollbar({
                contentSizePx: view.pxPerSec * Math.max(0, data().projectSec),
                viewportSizePx: viewportWidthPx,
                scrollPx: dragStartScroll,
                maxScrollPx: scroll.maxScrollLeft(),
            });
            scroll.setScrollLeft(
                dragStartScroll +
                    scrollDeltaFromThumbDrag(pointerDeltaPx, geometry, scroll.maxScrollLeft()),
            );
            return;
        }
        const geometry = computeScrollbar({
            contentSizePx: data().tracks.length * view.rowHeight,
            viewportSizePx: viewportHeightPx,
            scrollPx: dragStartScroll,
            maxScrollPx: scroll.maxScrollTop(),
        });
        scroll.setScrollTop(
            dragStartScroll +
                scrollDeltaFromThumbDrag(pointerDeltaPx, geometry, scroll.maxScrollTop()),
        );
    }

    function onPointerMove(event: PointerEvent): void {
        if (dragAxis === null) return;
        applyThumbDrag((dragAxis === "x" ? event.clientX : event.clientY) - dragStartPointer);
    }

    function onPointerUp(): void {
        dragAxis = null;
    }

    const onHDown = makeThumbPointerDown("x");
    const onVDown = makeThumbPointerDown("y");
    hScrollbarThumb.addEventListener("pointerdown", onHDown);
    vScrollbarThumb.addEventListener("pointerdown", onVDown);
    window.addEventListener("pointermove", onPointerMove);
    window.addEventListener("pointerup", onPointerUp);

    // ── 输入：键盘滚动 ───────────────────────────────────────────────
    function onKeyDown(event: KeyboardEvent): void {
        const view = scroll.get();
        switch (event.key) {
            case "PageDown":
                event.preventDefault();
                scroll.setScrollLeft(view.scrollLeft + viewportWidthPx * 0.9);
                break;
            case "PageUp":
                event.preventDefault();
                scroll.setScrollLeft(view.scrollLeft - viewportWidthPx * 0.9);
                break;
            case "Home":
                event.preventDefault();
                scroll.setScrollLeft(0);
                break;
            case "End":
                event.preventDefault();
                scroll.setScrollLeft(scroll.maxScrollLeft());
                break;
            case "ArrowLeft":
                event.preventDefault();
                scroll.setScrollLeft(view.scrollLeft - KEYBOARD_STEP_PX);
                break;
            case "ArrowRight":
                event.preventDefault();
                scroll.setScrollLeft(view.scrollLeft + KEYBOARD_STEP_PX);
                break;
            default:
                break;
        }
    }
    container.addEventListener("keydown", onKeyDown);

    // ── 尺寸与 DPR 变化 ──────────────────────────────────────────────
    const resizeObserver = new ResizeObserver((entries) => {
        for (const entry of entries) {
            viewportWidthPx = Math.max(1, entry.contentRect.width);
            viewportHeightPx = Math.max(1, entry.contentRect.height);
        }
        // 外部边界变化后必须重新钳制（见 ScrollKernel 的约束 1）。
        scroll.reclamp();
        sceneDirty = true;
        loop.invalidate();
    });
    resizeObserver.observe(container);

    function onWindowResize(): void {
        sceneDirty = true;
        loop.invalidate();
    }
    window.addEventListener("resize", onWindowResize);

    // 首次标脏（尺寸与数据就绪后绘制第一帧）。
    loop.invalidate();

    return {
        invalidateScene() {
            sceneDirty = true;
            loop.invalidate();
        },

        dispose() {
            unsubscribeScroll();
            loop.stop();
            resizeObserver.disconnect();
            container.removeEventListener("wheel", onWheel);
            container.removeEventListener("keydown", onKeyDown);
            hScrollbarThumb.removeEventListener("pointerdown", onHDown);
            vScrollbarThumb.removeEventListener("pointerdown", onVDown);
            window.removeEventListener("pointermove", onPointerMove);
            window.removeEventListener("pointerup", onPointerUp);
            window.removeEventListener("resize", onWindowResize);
            rasterizer?.dispose();
            sdfBox.dispose();
            glyphProgram.dispose();
            glCanvas.dispose();
        },
    };
}
