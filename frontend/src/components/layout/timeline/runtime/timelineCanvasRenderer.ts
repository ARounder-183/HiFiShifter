import {
    buildTimelineClipVisualStyle,
    CLIP_CORNER_RADIUS_PX,
    resolveFontFamily,
    resolveThemeColor,
} from "./timelineCanvasStyle.js";
import { SNAP_OFFSET_HANDLE_SIZE_PX } from "../constants.js";
import {
    buildClipBodyInstance,
    CLIP_INSTANCE_FLOATS,
    type GlClipBodySink,
} from "./timelineClipGlRenderer.js";
import { fadeGainSigned } from "../reaperFade.js";

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

    // 全物理清屏：与 rasterize 同契约（round(css*dpr)），CSS 尺寸清屏在
    // 向上取整时会在画布底部遗留 0~0.5 物理行的永久残影。
    const clearDpr = window.devicePixelRatio || 1;
    ctx.save();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, Math.round(args.width * clearDpr), Math.round(args.height * clearDpr));
    ctx.restore();

    // 轨道横向分界线由 sticky 画布统一绘制：工程末尾之后的空白区也要有
    // 同样的分界线，且滚动/缩放时与 Clip 体同帧同步。
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
        const borderColor = resolveThemeColor("--qt-border", "rgba(148, 163, 184, 0.22)");
        const dpr = window.devicePixelRatio || 1;
        ctx.save();
        ctx.strokeStyle = borderColor;
        // 行分界线对齐设备像素：分数 DPR 下 1px CSS 线会被抗锯齿拆成
        // 1~2 物理像素的渐变线，粗细随落点相位漂移。
        ctx.lineWidth = 1 / dpr;
        for (let index = 1; index <= rowCount; index += 1) {
            const rawY = (startTrackIndex + index) * rowHeight;
            // 边界判定用原始坐标：吸附只是绘制细节。最后一条轨道的底边与
            // contentBottomPx 重合，若用吸附后的 y 过 guard（容差 1e-6），
            // +0.5/dpr 的偏移会让这根线被误跳——底边横线消失。
            if (rawY < viewportTopPx - 1 || rawY > viewportTopPx + args.height + 2) continue;
            if (rawY > bottomPx + 1e-6) continue;
            // 线体落在边界上方的设备像素行内：画布高度恰为内容底边，
            // 若线心压在边界 +(0.5/dpr) 上，整根线会被画布下缘裁掉。
            const y = (Math.round(rawY * dpr) - 0.5) / dpr;
            ctx.beginPath();
            ctx.moveTo(viewportLeft, y);
            ctx.lineTo(viewportLeft + args.width, y);
            ctx.stroke();
        }
        ctx.restore();
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
            useGl: args.glBodies != null && !isGroupActive,
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
            const metrics = ctx.measureText(style.gainLabel);
            const gainX = clipLeft + clipWidth - metrics.width - 6;
            if (style.showPlaybackRate) {
                const rateMetrics = ctx.measureText(style.playbackRateLabel);
                const rateX = gainX - rateMetrics.width - 8;
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
            let glCount = 0;
            for (const item of pending) {
                if (item.useGl) glCount += 1;
            }
            if (glCount > 0) {
                const needed = glCount * CLIP_INSTANCE_FLOATS;
                if (glInstanceBuffer.length < needed) {
                    glInstanceBuffer = new Float32Array(needed * 2);
                }
                let index = 0;
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
                    glCount,
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

    for (const clip of args.clips) {
        const item = prepareClip(clip);
        if (item.barrier) flushBatch();
        pending.push(item);
    }
    flushBatch();
}
