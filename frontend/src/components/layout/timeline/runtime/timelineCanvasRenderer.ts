/**
 * Clip 细节层（Canvas2D）渲染器。
 *
 * 【主要内容】在**内容坐标**下绘制 clip 的细节，并按批次合批提交块面：
 * 旋钮 / 徽标 / 名称 / 淡入淡出曲线 / 吸附三角 / 静音覆盖 / lane 分界线，
 * 以及 MIDI / 音高参考块的音高折线与 "▽" 回绕标记。
 *
 * 【作用】内核把 clip 的**主体块面**交给 GL（`glBodies`）后，本函数仍作为
 * Canvas2D 细节层被每帧调用（`kernel/host/timelineKernelHost.redrawDetails`）。
 * 需要压在块面之上的内容一律走"**最后一遍落笔**"模式：收集 → 等全部块面
 * 提交完 → 统一绘制（否则重叠区里后续批次的块面会把先画的内容盖掉）。
 *
 * 【与其他模块的关系】
 * - 上游：`timelineCanvasModel.buildSparseClipRenderModel` 产出几何与可选
 *   细节字段（`silenceSpansPx` / `takeLaneSeparatorOffsetsPx` /
 *   `midiPitchCurvePx` / `midiLoopMarkerOffsetsPx`）。
 * - 横向：样式一律走 `timelineCanvasStyle`；音高折线的数学走
 *   `midiPitchCurve`；回绕标记的绘制走 `utils/loopRender.drawLoopMarkers`。
 * - 本文件**不得**自行做时间↔像素换算（模型侧已投影；拿到的是 CSS 像素）。
 */

import {
    buildTimelineClipVisualStyle,
    CLIP_CORNER_RADIUS_PX,
    resolveFontFamily,
    resolveThemeColor,
} from "./timelineCanvasStyle.js";
import { SNAP_OFFSET_HANDLE_SIZE_PX } from "../constants.js";
import {
    buildClipBodyInstance,
    buildGuideInstance,
    CLIP_INSTANCE_FLOATS,
    type GlClipBodySink,
} from "./timelineClipGlRenderer.js";
import { fadeGainSigned } from "../reaperFade.js";
import { drawLoopMarkers } from "../../../../utils/loopRender.js";

function drawFadeCurveStroke(
    ctx: CanvasRenderingContext2D,
    args: {
        leftPx: number;
        topPx: number;
        widthPx: number;
        heightPx: number;
        shape: number;
        dir: number;
        mode: "in" | "out";
    },
): void {
    const widthPx = Math.max(1, args.widthPx);
    const heightPx = Math.max(1, args.heightPx);
    const shapeId = Math.trunc(Number.isFinite(args.shape) ? args.shape : 255);
    if (shapeId === 0 && Math.abs(args.dir) < 1e-9) {
        // 直线快路径。淡入 = 增益沿 x 上升（左下→右上）；淡出相反。
        // y 轴向下：增益 1 → 屏幕上方（topPx）。
        ctx.beginPath();
        if (args.mode === "in") {
            ctx.moveTo(args.leftPx, args.topPx + heightPx);
            ctx.lineTo(args.leftPx + widthPx, args.topPx);
        } else {
            ctx.moveTo(args.leftPx, args.topPx);
            ctx.lineTo(args.leftPx + widthPx, args.topPx + heightPx);
        }
        ctx.stroke();
        return;
    }

    // ── 偏差驱动的自适应细分 ────────────────────────────────────────
    // 固定采样数（此前 ≤96 点且按宽度均分）在极端缩放下相邻采样点相距
    // 几十甚至上百像素，陡峭预设的末段"爆发区"在两采样点之间会被画成
    // 一条直弦——视觉上曲线"没接到角上"。这里改为按【屏幕空间偏差】
    // 递归细分：弦中点到真实曲线的偏差超过 0.6px 就继续拆分，直到
    // 折线与真实曲线处处贴合。端点 t=0/1 始终包含（增益在两端被核心
    // 函数精确钳制），因此曲线必然精确落在左下/右上（或反向）边角上。
    const gainAt = (t: number): number => fadeGainSigned(args.shape, args.dir, args.mode, t);
    const xAt = (t: number): number => args.leftPx + t * widthPx;
    const yAt = (t: number): number => args.topPx + heightPx * (1 - gainAt(t));

    const MAX_POINTS = 1200;
    const TOLERANCE_PX = 0.6;

    interface Segment {
        t0: number;
        t1: number;
        x0: number;
        y0: number;
        x1: number;
        y1: number;
        dev: number;
        tm: number;
        xm: number;
        ym: number;
    }

    const evaluateDeviation = (
        t0: number,
        t1: number,
        x0: number,
        y0: number,
        x1: number,
        y1: number,
    ) => {
        // 偏差度量使用【中点 + 两个四分点】联合探测：仅取弦中点 vs 曲线
        // 中点时，点对称的 S 曲线（g(0.5)=0.5，g(t)+g(1-t)=1）偏差恒为 0，
        // 会被误判为"足够平直"而画成直线 —— 正是"两类 S 曲线永远是直线"
        // 的根因。多点探测对任何单调形状都可靠。
        const tm = (t0 + t1) / 2;
        const tq0 = t0 + (t1 - t0) * 0.25;
        const tq1 = t0 + (t1 - t0) * 0.75;
        const xm = xAt(tm);
        const ym = yAt(tm);
        const xq0 = xAt(tq0);
        const yq0 = yAt(tq0);
        const xq1 = xAt(tq1);
        const yq1 = yAt(tq1);
        const devMid = Math.hypot(xm - (x0 + x1) / 2, ym - (y0 + y1) / 2);
        const devQ0 = Math.hypot(xq0 - (x0 + (x1 - x0) * 0.25), yq0 - (y0 + (y1 - y0) * 0.25));
        const devQ1 = Math.hypot(xq1 - (x0 + (x1 - x0) * 0.75), yq1 - (y0 + (y1 - y0) * 0.75));
        const dev = Math.max(devMid, devQ0, devQ1);
        return { tm, xm, ym, dev };
    };

    const segments: Segment[] = [];
    const pushSegment = (
        t0: number,
        t1: number,
        x0: number,
        y0: number,
        x1: number,
        y1: number,
    ) => {
        const { tm, xm, ym, dev } = evaluateDeviation(t0, t1, x0, y0, x1, y1);
        segments.push({ t0, t1, x0, y0, x1, y1, dev, tm, xm, ym });
    };

    pushSegment(0, 1, args.leftPx, yAt(0), args.leftPx + widthPx, yAt(1));

    // 始终细分偏差最大的段；上限保护极端缩放下的工作量。
    while (segments.length < MAX_POINTS) {
        let worstIndex = -1;
        let worstDev = TOLERANCE_PX;
        for (let i = 0; i < segments.length; i += 1) {
            if (segments[i].dev > worstDev) {
                worstDev = segments[i].dev;
                worstIndex = i;
            }
        }
        if (worstIndex < 0) break;
        const seg = segments[worstIndex];
        segments.splice(worstIndex, 1);
        pushSegment(seg.t0, seg.tm, seg.x0, seg.y0, seg.xm, seg.ym);
        pushSegment(seg.tm, seg.t1, seg.xm, seg.ym, seg.x1, seg.y1);
    }

    segments.sort((a, b) => a.t0 - b.t0);
    ctx.beginPath();
    ctx.moveTo(segments[0].x0, segments[0].y0);
    for (const seg of segments) {
        ctx.lineTo(seg.x1, seg.y1);
    }
    ctx.stroke();
}

export function drawTimelineCanvas(
    ctx: CanvasRenderingContext2D,
    args: {
        width: number;
        height: number;
        clips: Array<{
            id: string;
            trackId: string;
            leftPx: number;
            topPx: number;
            widthPx: number;
            heightPx: number;
            headerHeightPx: number;
            fadeInPx: number;
            fadeOutPx: number;
            fadeInShape: number;
            fadeOutShape: number;
            fadeInDir: number;
            fadeOutDir: number;
            selected: boolean;
            /** 是否绘制悬停提示环（见模型侧 `hovered` 的说明）。 */
            hovered?: boolean;
            muted: boolean;
            gain: number;
            playbackRate: number;
            groupId?: string;
            name: string;
            isMidiClip?: boolean;
            trackColor?: string;
            isRenaming?: boolean;
            /** 吸附偏移（像素，相对 Clip 左缘）—— 左下角 ◣ 标记。 */
            snapOffsetPx?: number;
            /** 前导重叠区宽度（像素，从左缘起算）。>0 时上 clip 在该区域半透。 */
            leadingOverlapPx?: number;
            /** 静音检测预览区段（像素，相对 clip 左缘）：半透明红色覆盖。 */
            silenceSpansPx?: Array<{ leftPx: number; widthPx: number }>;
            /** 多 Take lane 分界线的 y 偏移（相对 body 顶部，CSS px）。 */
            takeLaneSeparatorOffsetsPx?: number[];
            /**
             * MIDI / 音高参考块的音高折线点（相对 clip 左缘 / body 顶部，CSS px）。
             *
             * `y = NaN` 表示**断点**（该处无音符）：绘制端在此抬起画笔起新子
             * 路径，否则音符间的静音会被连成一条不存在的斜线。
             */
            midiPitchCurvePx?: Array<{ x: number; y: number }>;
            /** 音高折线的描边色（缺省走 clip 调色板回退色）。 */
            midiPitchStroke?: string;
            /**
             * Loop 回绕 / 媒体边界 "▽" 标记的 x 偏移（相对 clip 左缘，CSS px）。
             *
             * 已由模型侧按统一 axis 投影——绘制端拿不到 pxPerSec，不得自行换算。
             */
            midiLoopMarkerOffsetsPx?: number[];
        }>;
        /** 轨道横向分界线（延伸到工程末尾之后）。 */
        rowGuides?: {
            startTrackIndex: number;
            rowCount: number;
            rowHeight: number;
            /** 轨道内容底部边界；与网格使用同一个 trackGridHeight。 */
            contentBottomPx?: number;
        };
        /** 当前视口水平/竖直偏移（内容坐标），供分界线横跨可见区域。 */
        viewportLeft?: number;
        viewportTopPx?: number;
        fontFamily?: string;
        activeGroupIds?: Set<string>;
        disabledGroupIds?: string[];
        /** 主题模式（React 侧显式传入，切换时驱动画布同帧重绘） */
        darkMode?: boolean;
        /**
         * GL 块面渲染器（P3，dev 开关控制）。
         *
         * 提供时，clip 的**主体块面**（header / body / 前导重叠区 / 分隔线 /
         * 分隔缝 / 描边）改由它实例化绘制，本函数只画细节层（旋钮 / 徽标 /
         * 文字 / 淡变 / 吸附三角）。不提供时全部走 Canvas2D 合批路径。
         *
         * 编组激活的 clip 会自动退回 Canvas2D：它的外圈描边伸出矩形 2px，
         * 超出 GL 渲染器的单矩形模型。
         */
        glBodies?: GlClipBodySink | null;
        /**
         * 编组激活时**只画描边、不画块面**（供「细节层位于波形之上」的调用方使用）。
         *
         * 【为什么需要】编组激活的 clip 默认会退回 Canvas2D 画整块（块面 + 深金外圈
         * 描边），因为外圈描边伸出矩形 2px、超出 GL 渲染器的单矩形模型。旧实现的
         * 块面画布在波形**之下**，所以块面不会遮住波形；渲染内核的细节层在波形
         * **之上**，若照旧画块面就会把该 clip 的波形盖掉。
         *
         * 置 true 时：块面仍由 GL 承担（`useGl` 不再被编组激活否决），Canvas2D 只补
         * 那一圈描边——视觉与旧实现等价（描边只在 clip 边缘，不与波形重叠）。
         *
         * @default false
         */
        groupOutlineOverGl?: boolean;
        /**
         * 视口左上角的**内容坐标**（CSS 像素）。
         *
         * Canvas2D 路径靠 `ctx.translate` 实现，GL 路径没有这个变换，必须
         * 显式给出。仅在传了 `glBodies` 时使用；为兼容既有调用方可选，缺省 0。
         */
        originXPx?: number;
        originYPx?: number;
    },
): void {
    const fontFamily = args.fontFamily || resolveFontFamily();
    // Clip 前景方向随主题（深色主题 = 暗色块 + 浅色前景）。
    // 优先用 React 侧显式传入的 darkMode（保证切主题当帧即重绘），
    // 未传时回退读 DOM（兼容旧调用方）。
    const darkMode =
        args.darkMode ??
        (typeof document !== "undefined" && document.documentElement.dataset.theme === "dark");
    // 行分界线颜色：Canvas2D 路径与 GL 路径共用（GL 路径在 flushBatch 里
    // 把它写进实例的 border 色槽）。
    const guideBorderColor = resolveThemeColor("--qt-border", "rgba(148, 163, 184, 0.22)");

    // 全物理清屏：与 rasterize 同契约（round(css*dpr)），CSS 尺寸清屏在
    // 向上取整时会在画布底部遗留 0~0.5 物理行的永久残影。
    const clearDpr = window.devicePixelRatio || 1;
    ctx.save();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, Math.round(args.width * clearDpr), Math.round(args.height * clearDpr));
    ctx.restore();

    // 轨道横向分界线由 sticky 画布统一绘制：工程末尾之后的空白区也要有
    // 同样的分界线，且滚动/缩放时与 Clip 体同帧同步。
    // 行分界线（内容坐标 y + 线高）。Canvas2D 路径直接画；GL 路径把它作为
    // 「平面矩形」实例交给 glBodies，与 clip 同一次 draw call 绘制。
    const guideLines: Array<{ y: number; h: number }> = [];
    if (args.rowGuides && args.rowGuides.rowCount > 0) {
        const { startTrackIndex, rowCount, rowHeight, contentBottomPx } = args.rowGuides;
        const viewportLeft = Number.isFinite(args.viewportLeft) ? (args.viewportLeft as number) : 0;
        const viewportTopPx = Number.isFinite(args.viewportTopPx)
            ? (args.viewportTopPx as number)
            : 0;
        const bottomPx = Number.isFinite(contentBottomPx)
            ? (contentBottomPx as number)
            : Number.POSITIVE_INFINITY;
        // 主题色走进程级缓存（见 timelineCanvasStyle.resolveThemeColor）：
        // 这里每帧都会执行，裸调 getComputedStyle 会触发强制样式重算。
        const borderColor = guideBorderColor;
        const dpr = window.devicePixelRatio || 1;
        // GL 模式：行分界线作为「平面矩形」实例交给 glBodies，与 clip 同一次
        // draw call 绘制，且排在 clip 实例**之前**——保持「线在下、clip 在上」
        // 的原层叠顺序（Canvas2D 路径里分界线先画、clip 后画会盖住它）。
        // 这样也修掉了 2D canvas 位于 GL canvas 之上导致的 0.5px 压线差异。
        if (args.glBodies) {
            for (let index = 1; index <= rowCount; index += 1) {
                const rawY = (startTrackIndex + index) * rowHeight;
                if (rawY < viewportTopPx - 1 || rawY > viewportTopPx + args.height + 2) continue;
                if (rawY > bottomPx + 1e-6) continue;
                guideLines.push({
                    y: (Math.round(rawY * dpr) - 0.5) / dpr,
                    h: 1 / dpr,
                });
            }
        } else {
            ctx.save();
            ctx.strokeStyle = borderColor;
            // 行分界线对齐设备像素：分数 DPR 下 1px CSS 线会被抗锯齿拆成
            // 1~2 物理像素的渐变线，粗细随落点相位漂移。
            ctx.lineWidth = 1 / dpr;
            for (let index = 1; index <= rowCount; index += 1) {
                const rawY = (startTrackIndex + index) * rowHeight;
                if (rawY < viewportTopPx - 1 || rawY > viewportTopPx + args.height + 2) continue;
                if (rawY > bottomPx + 1e-6) continue;
                const y = (Math.round(rawY * dpr) - 0.5) / dpr;
                ctx.beginPath();
                ctx.moveTo(viewportLeft, y);
                ctx.lineTo(viewportLeft + args.width, y);
                ctx.stroke();
            }
            ctx.restore();
        }
    }

    // 同轨 clip 的左缘集合：用于判定"本 clip 右缘是否紧贴另一个 clip"
    // （相邻 clip 间画泳道底色分隔缝，见下方绘制）—— 避免把两个相连的
    // clip 误认成连续的一块。leftPx 来自同一投影函数，紧贴时可能引入
    // 亚像素误差，因此按 0.5px 容差匹配。
    const sameTrackClipLefts = new Map<string, number[]>();
    for (const c of args.clips) {
        let list = sameTrackClipLefts.get(c.trackId);
        if (!list) sameTrackClipLefts.set(c.trackId, (list = []));
        list.push(c.leftPx);
    }
    for (const list of sameTrackClipLefts.values()) list.sort((a, b) => a - b);
    const hasAdjacentRight = (trackId: string, rightEdgePx: number): boolean => {
        const list = sameTrackClipLefts.get(trackId);
        if (!list) return false;
        // 二分查找首个 >= rightEdgePx - 0.5 的左缘。
        let lo = 0;
        let hi = list.length;
        while (lo < hi) {
            const mid = (lo + hi) >> 1;
            if (list[mid] < rightEdgePx - 0.5) lo = mid + 1;
            else hi = mid;
        }
        return lo < list.length && Math.abs(list[lo] - rightEdgePx) <= 0.5;
    };

    // ══════════════════════════════════════════════════════════════════
    // Clip 绘制：准备 → 合批提交
    //
    // 【为什么改】原实现对每个 clip 做一次 `ctx.clip()`（圆角矩形遮罩）。
    // 遮罩是 Canvas2D 里最贵的操作之一——它要为后续所有绘制建立裁剪层，
    // 400 个 clip 即 400 次，粗估 4~8 ms/帧，是本画布最大的单项开销。
    //
    // 【为什么可以去掉 clip()】圆角矩形只在**四角**有弧线。header 的下边
    // 与 body 的上边都落在弧线之外（headerHeight 远大于半径），body 的下边
    // 才受下方两角影响。因此改用「每角独立半径」的 roundRect 分别填充
    // header / body，结果与「整体裁剪 + 平涂矩形」逐像素等价。
    //
    // 【为什么能合批】同一批里的 clip 彼此不遮挡：填充与描边都严格落在各自
    // 矩形内（相邻分隔缝已收窄到不越界，见下），因此组内绘制顺序无关。
    // 会遮挡前面 clip 的 clip（前导重叠、编组外圈描边）作为**屏障**先提交
    // 当前批次，z 序由此严格保持。
    // ══════════════════════════════════════════════════════════════════

    /** 一个待填充/描边的矩形（每角独立半径）。 */
    interface RectOp {
        x: number;
        y: number;
        w: number;
        h: number;
        /** 四角半径 [左上, 右上, 右下, 左下]（CSS 像素）。 */
        radii: [number, number, number, number];
    }
    interface FillOp extends RectOp {
        style: string;
        alpha: number;
    }
    interface StrokeOp extends RectOp {
        style: string;
        lineWidth: number;
    }

    type ClipStyle = ReturnType<typeof buildTimelineClipVisualStyle>;

    /** 一个 clip 的绘制计划：几何 + 样式 + 合批所需的矩形清单。 */
    interface PreparedClip {
        clip: (typeof args.clips)[number];
        style: ClipStyle;
        left: number;
        top: number;
        width: number;
        height: number;
        headerHeight: number;
        bodyTop: number;
        bodyHeight: number;
        radius: number;
        leadingOverlapPx: number;
        isGroupActive: boolean;
        isGroupDisabled: boolean;
        /** 右缘是否紧贴下一个同轨 clip（需要画分隔缝）。 */
        hasSeam: boolean;
        /**
         * 是否交给 GL 绘制块面。
         *
         * 编组激活的 clip **不走 GL**：它的外圈描边伸出矩形 2px，会与邻居
         * 重叠着色，超出 GL 渲染器的单矩形模型，故退回 Canvas2D 路径
         * （数量受选中/编组限制，通常很少）。
         */
        useGl: boolean;
        fills: FillOp[];
        strokes: StrokeOp[];
        /** true = 本 clip 会遮挡批次内已有的 clip，入队前必须提交当前批次。 */
        barrier: boolean;
    }

    /** 相邻 clip 分隔缝颜色（泳道底色）。 */
    const seamColor = darkMode ? "rgb(31, 31, 31)" : "rgb(237, 240, 245)";

    /**
     * 计算一个 clip 的几何、样式与合批矩形。
     *
     * 只做计算，不产生任何绘制调用。
     */
    function prepareClip(clip: (typeof args.clips)[number]): PreparedClip {
        const clipLeft = clip.leftPx;
        const clipTop = clip.topPx;
        const clipWidth = Math.max(1, clip.widthPx);
        const clipHeight = Math.max(1, clip.heightPx);
        const headerHeight = Math.max(1, Math.min(clip.heightPx, clip.headerHeightPx));
        const bodyTop = clipTop + headerHeight;
        const bodyHeight = Math.max(1, clipHeight - headerHeight);
        const isGroupActive =
            clip.groupId != null && args.activeGroupIds?.has(clip.groupId) === true;
        const isGroupDisabled =
            clip.groupId != null && (args.disabledGroupIds?.includes(clip.groupId) ?? false);
        const style = buildTimelineClipVisualStyle({
            widthPx: clipWidth,
            trackColor: clip.trackColor,
            selected: clip.selected,
            muted: clip.muted,
            gain: clip.gain,
            playbackRate: clip.playbackRate,
            name: clip.name,
            fontFamily,
            isPitchAdjustment: clip.isMidiClip,
            groupId: clip.groupId,
            isGroupActive,
            isGroupDisabled,
            darkMode,
        });
        // 圆角半径按 Clip 实际尺寸收敛：极短 / 极矮的 Clip 不能把圆角画爆。
        const radius = Math.max(0, Math.min(CLIP_CORNER_RADIUS_PX, clipWidth / 2, clipHeight / 2));
        // 前导重叠区（被同轨前一个 clip 压住的部分）宽度。
        const leadingOverlapPx = Math.max(0, Math.min(clipWidth - 1, clip.leadingOverlapPx ?? 0));
        const baseAlpha = style.mutedAlpha;

        const fills: FillOp[] = [];
        if (leadingOverlapPx > 0.5) {
            // 重叠区半透，让下 clip 的色块/波形透出（避免两层不透明色块
            // 叠加成脏色）。左右两段各自只保留外侧的圆角。
            fills.push({
                style: style.headerFill,
                alpha: baseAlpha * 0.55,
                x: clipLeft,
                y: clipTop,
                w: leadingOverlapPx,
                h: headerHeight,
                radii: [radius, 0, 0, 0],
            });
            fills.push({
                style: style.headerFill,
                alpha: baseAlpha,
                x: clipLeft + leadingOverlapPx,
                y: clipTop,
                w: clipWidth - leadingOverlapPx,
                h: headerHeight,
                radii: [0, radius, 0, 0],
            });
            fills.push({
                style: style.bodyFill,
                alpha: baseAlpha * 0.55,
                x: clipLeft,
                y: bodyTop,
                w: leadingOverlapPx,
                h: bodyHeight,
                radii: [0, 0, 0, radius],
            });
            fills.push({
                style: style.bodyFill,
                alpha: baseAlpha,
                x: clipLeft + leadingOverlapPx,
                y: bodyTop,
                w: clipWidth - leadingOverlapPx,
                h: bodyHeight,
                radii: [0, 0, radius, 0],
            });
        } else {
            // header 只有上方两角是圆的；body 只有下方两角是圆的。
            fills.push({
                style: style.headerFill,
                alpha: baseAlpha,
                x: clipLeft,
                y: clipTop,
                w: clipWidth,
                h: headerHeight,
                radii: [radius, radius, 0, 0],
            });
            fills.push({
                style: style.bodyFill,
                alpha: baseAlpha,
                x: clipLeft,
                y: bodyTop,
                w: clipWidth,
                h: bodyHeight,
                radii: [0, 0, radius, radius],
            });
        }

        // header/body 分隔线：亮色块上的细深线，仅做分区提示。
        // 它落在 headerHeight 上，远在圆角弧线之下，因此整宽可见、无需收角。
        fills.push({
            style: "rgba(0, 0, 0, 0.14)",
            alpha: baseAlpha,
            x: clipLeft,
            y: clipTop + headerHeight,
            w: clipWidth,
            h: 1,
            radii: [0, 0, 0, 0],
        });

        // 相邻 clip 分隔缝：右缘紧贴下一个同轨 clip 时，在两块之间画一条
        // 泳道底色竖线。原实现画在 [right-0.5, right+0.5]，但右半边会被
        // 下一个 clip 覆盖，实际可见的只有左半边——这里直接只画可见的那
        // 0.5px，使其落在自身矩形内，从而**不参与跨 clip 的遮挡关系**，
        // 合批时才安全。
        const hasSeam = hasAdjacentRight(clip.trackId, clipLeft + clipWidth);
        if (hasSeam) {
            fills.push({
                style: seamColor,
                alpha: baseAlpha,
                x: clipLeft + clipWidth - 0.5,
                y: clipTop,
                w: 0.5,
                h: clipHeight,
                radii: [0, 0, 0, 0],
            });
        }

        const strokes: StrokeOp[] = [];
        // 编组激活 = 深金描边 + 外圈（编组语义，非选中语义）。
        if (isGroupActive) {
            strokes.push({
                style: "rgba(146, 104, 10, 0.8)",
                lineWidth: 1,
                x: clipLeft + 0.5,
                y: clipTop + 0.5,
                w: Math.max(0, clipWidth - 1),
                h: Math.max(0, clipHeight - 1),
                radii: [radius, radius, radius, radius],
            });
            strokes.push({
                style: "rgba(146, 104, 10, 0.8)",
                lineWidth: 1,
                x: clipLeft - 1.5,
                y: clipTop - 1.5,
                w: Math.max(0, clipWidth + 3),
                h: Math.max(0, clipHeight + 3),
                radii: [0, 0, 0, 0],
            });
        }
        // 描边：选中 = 白色 2px；未选中 = 淡收边 1px。
        strokes.push({
            style: style.borderStroke,
            lineWidth: style.borderLineWidth,
            x: clipLeft + 0.5,
            y: clipTop + 0.5,
            w: Math.max(0, clipWidth - 1),
            h: Math.max(0, clipHeight - 1),
            radii: [radius, radius, radius, radius],
        });
        // 悬停提示环：1px 深色、画在矩形**外侧**（旧实现的
        // `boxShadow: 0 0 0 1px rgba(0, 0, 0, 0.35)` 是向外扩散 1px）。
        // `hovered` 已排除编组内的 clip（见模型侧说明）。
        if (clip.hovered === true) {
            strokes.push({
                style: "rgba(0, 0, 0, 0.35)",
                lineWidth: 1,
                x: clipLeft - 0.5,
                y: clipTop - 0.5,
                w: clipWidth,
                h: clipHeight,
                radii: [radius, radius, radius, radius],
            });
        }

        return {
            clip,
            style,
            left: clipLeft,
            top: clipTop,
            width: clipWidth,
            height: clipHeight,
            headerHeight,
            bodyTop,
            bodyHeight,
            radius,
            leadingOverlapPx,
            isGroupActive,
            isGroupDisabled,
            hasSeam,
            // 编组激活的 clip 默认退回 Canvas2D（外圈描边超出 GL 的单矩形模型）；
            // 细节层在波形之上时改用「只补描边」模式，块面仍由 GL 承担（见
            // `groupOutlineOverGl` 的说明）。
            useGl: args.glBodies != null && (!isGroupActive || args.groupOutlineOverGl === true),
            fills,
            strokes,
            // 屏障：前导重叠会盖住前一个 clip；编组外圈描边会伸出自身矩形
            // 2px 盖住邻居。二者都必须先提交此前已排入的批次。
            barrier: leadingOverlapPx > 0.5 || isGroupActive,
        };
    }

    /** 绘制无法合批的逐 clip 细节（旋钮 / 徽标 / 文字 / 淡变 / 吸附三角）。 */
    function drawClipDetails(item: PreparedClip): void {
        const { clip, style } = item;
        const { left: clipLeft, top: clipTop, width: clipWidth, height: clipHeight } = item;
        const bodyTop = item.bodyTop;
        const bodyHeight = item.bodyHeight;
        const radius = item.radius;

        ctx.globalAlpha = style.mutedAlpha;

        if (style.showGainKnob) {
            const knobCenterX = clipLeft + style.gainKnobCenterOffsetX;
            const knobCenterY = clipTop + style.gainKnobCenterOffsetY;
            ctx.fillStyle = style.gainKnobFill;
            ctx.strokeStyle = style.gainKnobStroke;
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.arc(knobCenterX, knobCenterY, style.gainKnobRadius, 0, Math.PI * 2);
            ctx.fill();
            ctx.stroke();
            ctx.beginPath();
            ctx.fillStyle = style.gainKnobCoreFill;
            ctx.arc(knobCenterX, knobCenterY, 1.7, 0, Math.PI * 2);
            ctx.fill();
            const angle = ((style.gainKnobAngleDeg - 90) * Math.PI) / 180;
            const indicatorOuterX = knobCenterX + Math.cos(angle) * (style.gainKnobRadius - 1.1);
            const indicatorOuterY = knobCenterY + Math.sin(angle) * (style.gainKnobRadius - 1.1);
            const indicatorInnerX = knobCenterX + Math.cos(angle) * 1.6;
            const indicatorInnerY = knobCenterY + Math.sin(angle) * 1.6;
            ctx.beginPath();
            ctx.strokeStyle = style.gainKnobIndicator;
            ctx.lineWidth = 1.2;
            ctx.moveTo(indicatorInnerX, indicatorInnerY);
            ctx.lineTo(indicatorOuterX, indicatorOuterY);
            ctx.stroke();
        }

        if (style.showChainBadge) {
            const badgeX = clipLeft + style.chainBadgeOffsetX;
            const badgeY = clipTop + style.chainBadgeOffsetY;
            const badgeW = style.chainBadgeWidth;
            const badgeH = style.chainBadgeHeight;
            const badgeR = style.chainBadgeRadius;
            const badgeCx = badgeX + badgeW / 2;
            const badgeCy = badgeY + badgeH / 2;

            ctx.beginPath();
            ctx.roundRect(badgeX, badgeY, badgeW, badgeH, badgeR);
            ctx.fillStyle = style.chainBadgeFill;
            ctx.fill();
            ctx.strokeStyle = style.chainBadgeStroke;
            ctx.lineWidth = 1;
            ctx.stroke();

            // Simple chain-link icon: two overlapping circles with a connecting bar
            ctx.strokeStyle = style.chainBadgeTextFill;
            ctx.lineWidth = 1.5;
            ctx.lineCap = "round";
            const leftCx = badgeCx - 3;
            const rightCx = badgeCx + 3;
            const linkR = 2.5;
            ctx.beginPath();
            ctx.ellipse(leftCx, badgeCy - 0.5, linkR, linkR * 0.7, 0, 0, Math.PI * 2);
            ctx.stroke();
            ctx.beginPath();
            ctx.ellipse(rightCx, badgeCy + 0.5, linkR, linkR * 0.7, 0, 0, Math.PI * 2);
            ctx.stroke();
            ctx.beginPath();
            ctx.moveTo(leftCx + linkR * 0.5, badgeCy - 1);
            ctx.lineTo(rightCx - linkR * 0.5, badgeCy + 0);
            ctx.stroke();
            ctx.lineCap = "butt";

            // Draw diagonal slash when group is disabled
            if (item.isGroupDisabled) {
                ctx.beginPath();
                ctx.strokeStyle = style.chainBadgeStroke;
                ctx.lineWidth = 1.5;
                ctx.moveTo(badgeX + 3, badgeY + 2);
                ctx.lineTo(badgeX + badgeW - 3, badgeY + badgeH - 2);
                ctx.stroke();
            }
        }

        if (style.showMuteBadge) {
            const badgeX = clipLeft + style.muteBadgeOffsetX;
            const badgeY = clipTop + style.muteBadgeOffsetY;
            ctx.beginPath();
            ctx.roundRect(
                badgeX,
                badgeY,
                style.muteBadgeWidth,
                style.muteBadgeHeight,
                style.muteBadgeRadius,
            );
            ctx.fillStyle = style.muteBadgeFill;
            ctx.fill();
            ctx.strokeStyle = style.muteBadgeStroke;
            ctx.lineWidth = 1;
            ctx.stroke();
            ctx.fillStyle = style.muteBadgeTextFill;
            ctx.font = `bold 9px ${fontFamily}`;
            ctx.textBaseline = "middle";
            ctx.textAlign = "center";
            ctx.fillText(
                style.muteBadgeLabel,
                badgeX + style.muteBadgeWidth / 2,
                badgeY + style.muteBadgeHeight / 2 + 0.5,
            );
            ctx.textAlign = "start";
        }

        if (style.showFormantBadge) {
            const badgeX = clipLeft + style.formantBadgeOffsetX;
            const badgeY = clipTop + style.formantBadgeOffsetY;
            ctx.beginPath();
            ctx.roundRect(
                badgeX,
                badgeY,
                style.formantBadgeWidth,
                style.formantBadgeHeight,
                style.formantBadgeRadius,
            );
            ctx.fillStyle = style.formantBadgeFill;
            ctx.fill();
            ctx.strokeStyle = style.formantBadgeStroke;
            ctx.lineWidth = 1;
            ctx.stroke();
            ctx.fillStyle = style.formantBadgeTextFill;
            ctx.font = `bold 9px ${fontFamily}`;
            ctx.textBaseline = "middle";
            ctx.textAlign = "center";
            ctx.fillText(
                style.formantBadgeLabel,
                badgeX + style.formantBadgeWidth / 2,
                badgeY + style.formantBadgeHeight / 2 + 0.5,
            );
            ctx.textAlign = "start";
        }

        if (style.showGainLabel) {
            ctx.fillStyle = style.textFill;
            ctx.font = `10px ${fontFamily}`;
            ctx.textBaseline = "middle";
            // 宽度取自样式解析（而非现场 measureText）：命中端
            // （`clipHeaderControls`）消费同一个值来划标签命中区，两处必须是
            // 同一份数据，否则「看到的」与「可点的」会漂移。
            const gainX = clipLeft + clipWidth - style.gainLabelWidth - 6;
            if (style.showPlaybackRate) {
                const rateX = gainX - style.rateLabelWidth - 8;
                ctx.fillText(style.playbackRateLabel, rateX, clipTop + 9);
            }
            ctx.fillText(style.gainLabel, gainX, clipTop + 9);
        }

        if (!clip.isRenaming && style.showName && style.displayName.length > 0) {
            const textStartX = clipLeft + style.leadingControlsWidth;
            const textEndX = style.showGainLabel
                ? clipLeft + clipWidth - style.trailingReservePx + 4
                : clipLeft + clipWidth - 8;
            const availableWidth = Math.max(0, textEndX - textStartX);
            if (availableWidth > 12) {
                ctx.save();
                ctx.beginPath();
                ctx.rect(textStartX, clipTop, availableWidth, item.headerHeight);
                ctx.clip();
                ctx.fillStyle = style.textFill;
                ctx.font = `12px ${fontFamily}`;
                ctx.textBaseline = "middle";
                ctx.fillText(style.displayName, textStartX, clipTop + 9);
                ctx.restore();
            }
        }

        // ── 淡入淡出 ────────────────────────────────────────────
        // 压暗区：原实现依赖外层圆角裁剪来收住下方两角，这里改用每角独立
        // 半径，等价且无需遮罩。
        if (clip.fadeInPx > 0) {
            const fadeW = Math.min(clipWidth, clip.fadeInPx);
            ctx.fillStyle = "rgba(0, 0, 0, 0.32)";
            ctx.beginPath();
            ctx.roundRect(clipLeft, bodyTop, fadeW, bodyHeight, [0, 0, 0, radius]);
            ctx.fill();
            ctx.strokeStyle = "rgba(255, 255, 255, 0.65)";
            ctx.lineWidth = 1.2;
            drawFadeCurveStroke(ctx, {
                leftPx: clipLeft,
                topPx: bodyTop,
                widthPx: fadeW,
                heightPx: bodyHeight,
                shape: clip.fadeInShape,
                dir: clip.fadeInDir,
                mode: "in",
            });
        }
        if (clip.fadeOutPx > 0) {
            const fadeW = Math.min(clipWidth, clip.fadeOutPx);
            const fadeX = clipLeft + clipWidth - fadeW;
            ctx.fillStyle = "rgba(0, 0, 0, 0.32)";
            ctx.beginPath();
            ctx.roundRect(fadeX, bodyTop, fadeW, bodyHeight, [0, 0, radius, 0]);
            ctx.fill();
            ctx.strokeStyle = "rgba(255, 255, 255, 0.65)";
            ctx.lineWidth = 1.2;
            drawFadeCurveStroke(ctx, {
                leftPx: fadeX,
                topPx: bodyTop,
                widthPx: fadeW,
                heightPx: bodyHeight,
                shape: clip.fadeOutShape,
                dir: clip.fadeOutDir,
                mode: "out",
            });
        }

        // ── SnapOffset（吸附偏移）三角标记 ────────────────────────
        // 左下角等腰直角三角形（直角在左下，◣）。**左侧竖直边严格对齐
        // 偏移位置**（与波形内橙色竖虚线同 x）—— 不做宽度回退钳制；
        // 三角靠近/越过 Clip 末尾的部分按 Clip 矩形裁剪。
        if (clipWidth >= 12 && clipHeight >= 14) {
            const offsetPx = Math.max(0, Number(clip.snapOffsetPx) || 0);
            const triX = clipLeft + offsetPx;
            const triYBottom = clipTop + clipHeight;
            const size = SNAP_OFFSET_HANDLE_SIZE_PX;
            const triRight = triX + size;
            const clipRight = clipLeft + clipWidth;
            ctx.globalAlpha *= offsetPx > 1e-9 ? 0.95 : 0.55;
            // 三角整体落在 Clip 内（常态：吸附偏移靠近开头）时**不需要裁剪**。
            // 原实现无条件 `ctx.clip()` 一次，等于给每个可见 clip 都加一次
            // 遮罩——本画布最大的单项开销，在全览缩放下是 400 次/帧。
            // 只有三角真的越过 Clip 右缘时才退回裁剪路径（极罕见）。
            const needsClip = triRight > clipRight;
            if (needsClip) {
                ctx.save();
                ctx.beginPath();
                ctx.rect(clipLeft, clipTop, clipWidth, clipHeight);
                ctx.clip();
            }
            ctx.beginPath();
            ctx.moveTo(triX, triYBottom - size);
            ctx.lineTo(triX, triYBottom);
            ctx.lineTo(triRight, triYBottom);
            ctx.closePath();
            ctx.fillStyle = style.snapOffsetTriFill;
            ctx.fill();
            ctx.strokeStyle = style.snapOffsetTriStroke;
            ctx.lineWidth = 1;
            ctx.stroke();
            if (needsClip) ctx.restore();
        }

        ctx.globalAlpha = 1;
    }

    // ── 合批提交 ────────────────────────────────────────────────
    // 同一批内的 clip 互不遮挡 ⇒ 按「样式」分组累积进 Path2D，最后每种样式
    // 只发一次 fill / stroke。样式种类由轨道色数与选中态决定，通常是个位数，
    // 因此绘制调用数从「clip 数 × 每 clip 操作数」塌缩到「样式数 × 2」。
    interface FillGroup {
        style: string;
        alpha: number;
        path: Path2D;
    }
    interface StrokeGroup {
        style: string;
        lineWidth: number;
        path: Path2D;
    }

    const pending: PreparedClip[] = [];

    /**
     * GL 实例数据的复用缓冲。
     *
     * 容量按需倍增、跨帧复用（稳态零分配），与波形 P2a 的顶点缓冲池同一
     * 思路：数据上传 GPU 后 CPU 侧即不再需要。
     */
    let glInstanceBuffer = new Float32Array(0);

    function flushBatch(): void {
        if (pending.length === 0) return;

        // ── GL 路径：块面走实例化渲染 ────────────────────────────
        // 走 GL 的 clip 不再进入下方的 Path2D 合批（两种途径画同一块面
        // 会叠色）。退回 Canvas2D 的只有编组激活的 clip（数量很少）。
        // 细节层（旋钮 / 徽标 / 文字…）无论哪条路径都照旧逐 clip 画在本
        // canvas 上——它位于 GL canvas **之上**，所以块面先画、细节后画
        // 的顺序天然成立。
        const glBodies = args.glBodies ?? null;
        let canvasItems = pending;
        if (glBodies !== null) {
            let glClipCount = 0;
            for (const item of pending) {
                if (item.useGl) glClipCount += 1;
            }
            const guideCount = guideLines.length;
            const totalCount = glClipCount + guideCount;
            if (totalCount > 0) {
                const needed = totalCount * CLIP_INSTANCE_FLOATS;
                if (glInstanceBuffer.length < needed) {
                    glInstanceBuffer = new Float32Array(needed * 2);
                }
                let index = 0;
                // 分界线排在 clip 实例**之前**：同一 draw call 内后画的盖住
                // 先画的，与 Canvas2D 路径「先画线、后画 clip」的层叠一致。
                for (const guide of guideLines) {
                    buildGuideInstance(
                        glInstanceBuffer,
                        index,
                        args.viewportLeft ?? 0,
                        guide.y,
                        args.width,
                        guide.h,
                        guideBorderColor,
                    );
                    index += 1;
                }
                for (const item of pending) {
                    if (!item.useGl) continue;
                    buildClipBodyInstance(
                        glInstanceBuffer,
                        index,
                        item.clip,
                        item.style,
                        item.hasSeam ? seamColor : null,
                    );
                    index += 1;
                }
                glBodies.render(
                    glInstanceBuffer,
                    totalCount,
                    args.width,
                    args.height,
                    window.devicePixelRatio || 1,
                    args.originXPx ?? 0,
                    args.originYPx ?? 0,
                );
            }
            canvasItems = pending.filter((item) => !item.useGl);
        }

        const fillGroups = new Map<string, FillGroup>();
        const strokeGroups = new Map<string, StrokeGroup>();

        for (const item of canvasItems) {
            for (const op of item.fills) {
                if (op.w <= 0 || op.h <= 0) continue;
                const key = `${op.style}|${op.alpha}`;
                let group = fillGroups.get(key);
                if (group === undefined) {
                    group = { style: op.style, alpha: op.alpha, path: new Path2D() };
                    fillGroups.set(key, group);
                }
                group.path.roundRect(op.x, op.y, op.w, op.h, op.radii);
            }
            for (const op of item.strokes) {
                if (op.w <= 0 || op.h <= 0) continue;
                const key = `${op.style}|${op.lineWidth}`;
                let group = strokeGroups.get(key);
                if (group === undefined) {
                    group = { style: op.style, lineWidth: op.lineWidth, path: new Path2D() };
                    strokeGroups.set(key, group);
                }
                group.path.roundRect(op.x, op.y, op.w, op.h, op.radii);
            }
        }

        for (const group of fillGroups.values()) {
            ctx.globalAlpha = group.alpha;
            ctx.fillStyle = group.style;
            ctx.fill(group.path);
        }
        for (const group of strokeGroups.values()) {
            ctx.globalAlpha = 1;
            ctx.strokeStyle = group.style;
            ctx.lineWidth = group.lineWidth;
            ctx.stroke(group.path);
        }
        ctx.globalAlpha = 1;

        // 细节层必须在本批次的填充与描边**之后**逐 clip 绘制。
        for (const item of pending) drawClipDetails(item);

        pending.length = 0;
    }

    /**
     * 静音检测预览的红色覆盖矩形（内容坐标），在**所有** clip 绘制完成后统一落笔。
     *
     * 【为什么单独一遍】红色层必须压在全部块面之上。重叠区里后一个 clip 的块面
     * 可能属于**后续批次**（`barrier` 由前导重叠 / 半透块面触发），若在各 clip 的
     * 细节阶段就画，先画的会被后画的块面盖掉——实测表现为「只有一部分静音区可见」。
     */
    const silenceOverlays: Array<{
        left: number;
        top: number;
        width: number;
        height: number;
    }> = [];
    /** 多 Take lane 分界线（同样在最后一遍落笔，避免被后续批次的块面盖掉）。 */
    const takeLaneSeparators: Array<{ left: number; top: number; width: number }> = [];
    /**
     * MIDI / 音高参考块的音高折线与 "▽" 回绕标记（同样在**最后一遍**落笔）。
     *
     * 【为什么单独一遍】与静音红色覆盖同理：重叠区里后一个 clip 的块面可能属于
     * **后续批次**，在本 clip 的细节阶段就画会被盖掉。折线又必须压在**本 clip
     * 的 body 色块之上**，因此只能等全部块面提交后统一画。
     *
     * 【数据来源】模型侧 `midiPitchCurvePx`（数学来自随内核改造成为孤儿的
     * `components/waveform/MidiPitchTrackCanvas.tsx`，已搬进 `midiPitchCurve.ts`）。
     * 该组件唯一的挂载点 `TrackLane` 被删除后，MIDI clip 的 body 因此变成空白
     * ——这是本次修复的功能回归点。
     */
    const midiPitchCurves: Array<{
        points: Array<{ x: number; y: number }>;
        color: string;
        alpha: number;
        /** 该 clip 的 body 矩形（内容坐标），用于**逐条**裁剪。 */
        clip: { left: number; top: number; width: number; height: number };
    }> = [];
    /** 回绕标记：颜色 + 位置（相对 clip 左缘）+ 所属 clip 的 body 矩形。 */
    const midiLoopMarkers: Array<{
        xs: number[];
        color: string;
        /** body 顶部（内容坐标 y）：标记贴在该 clip 的 body 上缘。 */
        top: number;
        height: number;
        /** 所属 clip 的左右缘（内容坐标）：标记水平方向按 clip 裁剪。 */
        left: number;
        right: number;
    }> = [];

    for (const clip of args.clips) {
        const item = prepareClip(clip);
        if (item.barrier) flushBatch();
        pending.push(item);
        if (clip.silenceSpansPx !== undefined) {
            for (const span of clip.silenceSpansPx) {
                silenceOverlays.push({
                    left: item.left + span.leftPx,
                    top: item.bodyTop,
                    width: span.widthPx,
                    height: item.bodyHeight,
                });
            }
        }
        if (clip.takeLaneSeparatorOffsetsPx !== undefined) {
            for (const offset of clip.takeLaneSeparatorOffsetsPx) {
                takeLaneSeparators.push({
                    left: item.left,
                    top: item.bodyTop + offset,
                    width: item.width,
                });
            }
        }
        // MIDI 音高折线：点坐标相对 clip 左缘 / body 顶部，绘制时加回原点。
        // 与静音覆盖一样，**逐条**记录裁剪矩形（一次全局裁剪只能覆盖一个 clip）。
        if (clip.midiPitchCurvePx !== undefined && clip.midiPitchCurvePx.length >= 2) {
            midiPitchCurves.push({
                points: clip.midiPitchCurvePx,
                color: clip.midiPitchStroke ?? "rgba(34, 211, 238, 0.78)",
                // 与搬迁前 MidiPitchTrackCanvas 同一透明度语义：静音整体压暗。
                alpha: clip.muted ? 0.4 : 0.85,
                clip: {
                    left: item.left,
                    top: item.bodyTop,
                    width: item.width,
                    height: item.bodyHeight,
                },
            });
        }
        if (clip.midiLoopMarkerOffsetsPx !== undefined) {
            midiLoopMarkers.push({
                xs: clip.midiLoopMarkerOffsetsPx,
                color: clip.midiPitchStroke ?? "rgba(34, 211, 238, 0.78)",
                top: item.bodyTop,
                height: item.bodyHeight,
                left: item.left,
                right: item.left + item.width,
            });
        }
    }
    flushBatch();

    if (silenceOverlays.length > 0) {
        // 与旧实现 `ClipItem` 的红色覆盖层同源：半透明红，覆盖 clip 的 body 区
        // （不含 header，避免盖住名称 / 徽标等可交互标记）。
        ctx.save();
        ctx.globalAlpha = 1;
        ctx.fillStyle = "rgba(239, 68, 68, 0.3)";
        for (const rect of silenceOverlays) {
            ctx.fillRect(rect.left, rect.top, rect.width, rect.height);
        }
        ctx.restore();
    }

    if (takeLaneSeparators.length > 0) {
        // 多 Take lane 分界线：亮色块上一律深色分线（白线在彩色块上看不见），
        // 与旧实现 `ClipItem` 的分隔线同源。首条 lane 的顶边即 header 边界，
        // 已由模型侧 `slice(1)` 排除。
        ctx.save();
        ctx.globalAlpha = 1;
        ctx.fillStyle = "rgba(0, 0, 0, 0.18)";
        for (const line of takeLaneSeparators) {
            ctx.fillRect(line.left, line.top, line.width, 1);
        }
        ctx.restore();
    }

    // ── MIDI / 音高参考块的音高折线 ──────────────────────────────────
    // 【为什么在最后一遍】折线要压在**本 clip 的 body 色块**之上，而重叠区里
    // 靠后的 clip 块面可能属于后续批次（见 `barrier`）；在细节阶段逐 clip 画
    // 会被那些块面盖掉（与静音红色覆盖是同一个坑）。此处所有块面已提交完毕，
    // 折线因此一定可见。
    if (midiPitchCurves.length > 0) {
        ctx.save();
        ctx.lineJoin = "round";
        ctx.lineCap = "round";
        ctx.lineWidth = 1.5;
        for (const curve of midiPitchCurves) {
            // **逐条**裁剪：多个 clip 的折线在同一遍落笔，一次全局 clip 只能
            // 覆盖其中一个（其余折线会溢出到相邻行）。
            ctx.save();
            ctx.beginPath();
            ctx.rect(curve.clip.left, curve.clip.top, curve.clip.width, curve.clip.height);
            ctx.clip();
            ctx.beginPath();
            ctx.globalAlpha = curve.alpha;
            ctx.strokeStyle = curve.color;
            let penDown = false;
            for (const point of curve.points) {
                // 点坐标相对 clip 左缘 / body 顶部，绘制时把原点加回去。
                const x = curve.clip.left + point.x;
                const y = curve.clip.top + point.y;
                // NaN 的 y = 断点（该处无音符）：抬起画笔，另起子路径。
                // 若连成一条线，音符之间的静音会被画成不存在的斜线。
                if (Number.isNaN(y)) {
                    penDown = false;
                    continue;
                }
                if (!penDown) {
                    ctx.moveTo(x, y);
                    penDown = true;
                } else {
                    ctx.lineTo(x, y);
                }
            }
            ctx.stroke();
            ctx.restore();
        }
        ctx.restore();
    }

    // ── Loop 回绕 / 媒体边界 "▽" 标记 ────────────────────────────────
    // 复用 `loopRender.drawLoopMarkers`（与波形分段边界同一套锚点数学，
    // 见 `midiPitchCurve.resolveClipLoopMarkerOffsetsSec`）。标记位置已由模型
    // 侧投影为相对 clip 左缘的像素，这里把原点平移到 body 左上角后再调用——
    // 该函数以 (0,0) 为区域左上角、向下画三角。
    if (midiLoopMarkers.length > 0) {
        ctx.save();
        for (const marker of midiLoopMarkers) {
            if (marker.xs.length === 0) continue;
            ctx.save();
            ctx.beginPath();
            // 水平方向裁到 clip 本体（垂直方向本就落在 body 内）。
            ctx.rect(marker.left, marker.top, marker.right - marker.left, marker.height);
            ctx.clip();
            ctx.translate(marker.left, marker.top);
            // 标记恒为不透明（与搬迁前 `MidiPitchTrackCanvas` 一致：只有折线受
            // `muted` 压暗，回绕标记不受影响）。显式赋值以免受上游环境 alpha 影响。
            ctx.globalAlpha = 1;
            drawLoopMarkers(ctx, marker.xs, marker.height, marker.color);
            ctx.restore();
        }
        ctx.restore();
    }
}
