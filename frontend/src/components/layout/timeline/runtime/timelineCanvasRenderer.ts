import {
    buildTimelineClipVisualStyle,
    CLIP_CORNER_RADIUS_PX,
    resolveFontFamily,
} from "./timelineCanvasStyle.js";
import { SNAP_OFFSET_HANDLE_SIZE_PX } from "../constants.js";
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
    },
): void {
    const fontFamily = args.fontFamily || resolveFontFamily();
    // Clip 前景方向随主题（深色主题 = 暗色块 + 浅色前景）。
    // 优先用 React 侧显式传入的 darkMode（保证切主题当帧即重绘），
    // 未传时回退读 DOM（兼容旧调用方）。
    const darkMode =
        args.darkMode ??
        (typeof document !== "undefined" &&
            document.documentElement.dataset.theme === "dark");

    ctx.clearRect(0, 0, args.width, args.height);

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
        const borderColor =
            (typeof document !== "undefined"
                ? getComputedStyle(document.documentElement).getPropertyValue("--qt-border").trim()
                : "") || "rgba(148, 163, 184, 0.22)";
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

    for (const clip of args.clips) {
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
        const visualStyle = buildTimelineClipVisualStyle({
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
        ctx.save();
        ctx.globalAlpha = visualStyle.mutedAlpha;

        // 圆角半径按 Clip 实际尺寸收敛：极短 / 极矮的 Clip 不能把圆角画爆。
        const radius = Math.max(
            0,
            Math.min(CLIP_CORNER_RADIUS_PX, clipWidth / 2, clipHeight / 2),
        );
        const borderRect = () => {
            ctx.beginPath();
            ctx.roundRect(
                clipLeft + 0.5,
                clipTop + 0.5,
                Math.max(0, clipWidth - 1),
                Math.max(0, clipHeight - 1),
                radius,
            );
        };

        // 主体填色统一裁剪到圆角矩形内：内部各段纯色平涂，四角被一起收圆。
        // 简洁风（参考 Ableton / REAPER）：不用渐变、高光、发光 —— 平涂的
        // 边界最清晰，密集轨道里不糊，渲染也最便宜。
        ctx.save();
        ctx.beginPath();
        ctx.roundRect(clipLeft, clipTop, clipWidth, clipHeight, radius);
        ctx.clip();

        // 前导重叠区（被同轨前一个 clip 压住的部分）宽度：上 clip 在该区
        // 半透，让下 clip 的色块/波形透出，避免两层不透明色块叠加成脏色。
        const leadingOverlapPx = Math.max(
            0,
            Math.min(clipWidth - 1, clip.leadingOverlapPx ?? 0),
        );
        const overlapStart = clipLeft + leadingOverlapPx;

        // header：色块头部带，与 body 同色稍压深。前导重叠区也用半透。
        if (leadingOverlapPx > 0.5) {
            ctx.fillStyle = visualStyle.headerFill;
            ctx.globalAlpha *= 0.55;
            ctx.fillRect(clipLeft, clipTop, leadingOverlapPx, headerHeight);
            ctx.globalAlpha = visualStyle.mutedAlpha;
            ctx.fillStyle = visualStyle.headerFill;
            ctx.fillRect(overlapStart, clipTop, clipWidth - leadingOverlapPx, headerHeight);
        } else {
            ctx.fillStyle = visualStyle.headerFill;
            ctx.fillRect(clipLeft, clipTop, clipWidth, headerHeight);
        }

        // body：色块主体。前导重叠区半透。
        if (leadingOverlapPx > 0.5) {
            ctx.fillStyle = visualStyle.bodyFill;
            ctx.globalAlpha *= 0.55;
            ctx.fillRect(clipLeft, bodyTop, leadingOverlapPx, bodyHeight);
            ctx.globalAlpha = visualStyle.mutedAlpha;
            ctx.fillStyle = visualStyle.bodyFill;
            ctx.fillRect(overlapStart, bodyTop, clipWidth - leadingOverlapPx, bodyHeight);
        } else {
            ctx.fillStyle = visualStyle.bodyFill;
            ctx.fillRect(clipLeft, bodyTop, clipWidth, bodyHeight);
        }

        // body 中段（两条渐变之间）不再叠加黑色遮罩：此前 0.18→0.50 的加深
        // 让波形区域的背景发闷显脏（用户反馈"更干净一点"）。渐变区域自身的
        // 压暗由下方 fadeIn/fadeOut 的独立 fillRect 负责。

        // header/body 分隔线：亮色块上的细深线，仅做分区提示。
        ctx.fillStyle = "rgba(0, 0, 0, 0.14)";
        ctx.fillRect(clipLeft, clipTop + headerHeight, clipWidth, 1);

        ctx.restore();

        // 编组激活 = 深金描边 + 外圈（编组语义，非选中语义）。
        // 选中不画边框 —— 由色块提亮表达（Ableton 式，见 style 的 selected 分支）；
        // 未选中画极淡的深色收边，帮助在深背景上定界。
        if (isGroupActive) {
            ctx.strokeStyle = "rgba(146, 104, 10, 0.8)";
            ctx.lineWidth = 1;
            borderRect();
            ctx.stroke();
            ctx.strokeRect(
                clipLeft - 1.5,
                clipTop - 1.5,
                Math.max(0, clipWidth + 3),
                Math.max(0, clipHeight + 3),
            );
        }
        // 描边：选中 = 白色 2px；未选中 = 淡收边 1px（值随选中态切换）。
        ctx.strokeStyle = visualStyle.borderStroke;
        ctx.lineWidth = visualStyle.borderLineWidth;
        borderRect();
        ctx.stroke();

        if (visualStyle.showGainKnob) {
            const knobCenterX = clipLeft + visualStyle.gainKnobCenterOffsetX;
            const knobCenterY = clipTop + visualStyle.gainKnobCenterOffsetY;
            ctx.fillStyle = visualStyle.gainKnobFill;
            ctx.strokeStyle = visualStyle.gainKnobStroke;
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.arc(knobCenterX, knobCenterY, visualStyle.gainKnobRadius, 0, Math.PI * 2);
            ctx.fill();
            ctx.stroke();
            ctx.beginPath();
            ctx.fillStyle = visualStyle.gainKnobCoreFill;
            ctx.arc(knobCenterX, knobCenterY, 1.7, 0, Math.PI * 2);
            ctx.fill();
            const angle = ((visualStyle.gainKnobAngleDeg - 90) * Math.PI) / 180;
            const indicatorOuterX =
                knobCenterX + Math.cos(angle) * (visualStyle.gainKnobRadius - 1.1);
            const indicatorOuterY =
                knobCenterY + Math.sin(angle) * (visualStyle.gainKnobRadius - 1.1);
            const indicatorInnerX = knobCenterX + Math.cos(angle) * 1.6;
            const indicatorInnerY = knobCenterY + Math.sin(angle) * 1.6;
            ctx.beginPath();
            ctx.strokeStyle = visualStyle.gainKnobIndicator;
            ctx.lineWidth = 1.2;
            ctx.moveTo(indicatorInnerX, indicatorInnerY);
            ctx.lineTo(indicatorOuterX, indicatorOuterY);
            ctx.stroke();
        }

        if (visualStyle.showChainBadge) {
            const badgeX = clipLeft + visualStyle.chainBadgeOffsetX;
            const badgeY = clipTop + visualStyle.chainBadgeOffsetY;
            const badgeW = visualStyle.chainBadgeWidth;
            const badgeH = visualStyle.chainBadgeHeight;
            const badgeR = visualStyle.chainBadgeRadius;
            const badgeCx = badgeX + badgeW / 2;
            const badgeCy = badgeY + badgeH / 2;

            ctx.beginPath();
            ctx.roundRect(badgeX, badgeY, badgeW, badgeH, badgeR);
            ctx.fillStyle = visualStyle.chainBadgeFill;
            ctx.fill();
            ctx.strokeStyle = visualStyle.chainBadgeStroke;
            ctx.lineWidth = 1;
            ctx.stroke();

            // Simple chain-link icon: two overlapping circles with a connecting bar
            ctx.strokeStyle = visualStyle.chainBadgeTextFill;
            ctx.lineWidth = 1.5;
            ctx.lineCap = "round";
            const leftCx = badgeCx - 3;
            const rightCx = badgeCx + 3;
            const linkR = 2.5;
            // Left link
            ctx.beginPath();
            ctx.ellipse(leftCx, badgeCy - 0.5, linkR, linkR * 0.7, 0, 0, Math.PI * 2);
            ctx.stroke();
            // Right link
            ctx.beginPath();
            ctx.ellipse(rightCx, badgeCy + 0.5, linkR, linkR * 0.7, 0, 0, Math.PI * 2);
            ctx.stroke();
            // Connecting bar
            ctx.beginPath();
            ctx.moveTo(leftCx + linkR * 0.5, badgeCy - 1);
            ctx.lineTo(rightCx - linkR * 0.5, badgeCy + 0);
            ctx.stroke();

            // Draw diagonal slash when group is disabled
            if (isGroupDisabled) {
                ctx.beginPath();
                ctx.strokeStyle = visualStyle.chainBadgeStroke;
                ctx.lineWidth = 1.5;
                ctx.moveTo(badgeX + 3, badgeY + 2);
                ctx.lineTo(badgeX + badgeW - 3, badgeY + badgeH - 2);
                ctx.stroke();
            }
        }

        if (visualStyle.showMuteBadge) {
            const buttonX = clipLeft + visualStyle.muteBadgeOffsetX;
            const buttonY = clipTop + visualStyle.muteBadgeOffsetY;
            const buttonWidth = visualStyle.muteBadgeWidth;
            const buttonHeight = visualStyle.muteBadgeHeight;
            const buttonRadius = visualStyle.muteBadgeRadius;
            ctx.beginPath();
            ctx.roundRect(buttonX, buttonY, buttonWidth, buttonHeight, buttonRadius);
            ctx.fillStyle = visualStyle.muteBadgeFill;
            ctx.fill();
            ctx.strokeStyle = visualStyle.muteBadgeStroke;
            ctx.lineWidth = 1;
            ctx.stroke();
            ctx.fillStyle = visualStyle.muteBadgeTextFill;
            ctx.font = `bold 9px ${fontFamily}`;
            ctx.textBaseline = "middle";
            ctx.textAlign = "center";
            ctx.fillText(
                visualStyle.muteBadgeLabel,
                buttonX + buttonWidth / 2,
                buttonY + buttonHeight / 2 + 0.5,
            );
            ctx.textAlign = "start";
        }

        if (visualStyle.showFormantBadge) {
            const buttonX = clipLeft + visualStyle.formantBadgeOffsetX;
            const buttonY = clipTop + visualStyle.formantBadgeOffsetY;
            const buttonWidth = visualStyle.formantBadgeWidth;
            const buttonHeight = visualStyle.formantBadgeHeight;
            const buttonRadius = visualStyle.formantBadgeRadius;
            ctx.beginPath();
            ctx.roundRect(buttonX, buttonY, buttonWidth, buttonHeight, buttonRadius);
            ctx.fillStyle = visualStyle.formantBadgeFill;
            ctx.fill();
            ctx.strokeStyle = visualStyle.formantBadgeStroke;
            ctx.lineWidth = 1;
            ctx.stroke();
            ctx.fillStyle = visualStyle.formantBadgeTextFill;
            ctx.font = `bold 9px ${fontFamily}`;
            ctx.textBaseline = "middle";
            ctx.textAlign = "center";
            ctx.fillText(
                visualStyle.formantBadgeLabel,
                buttonX + buttonWidth / 2,
                buttonY + buttonHeight / 2 + 0.5,
            );
            ctx.textAlign = "start";
        }

        if (visualStyle.showGainLabel) {
            ctx.fillStyle = visualStyle.textFill;
            ctx.font = `10px ${fontFamily}`;
            ctx.textBaseline = "middle";
            const metrics = ctx.measureText(visualStyle.gainLabel);
            const gainX = clipLeft + clipWidth - metrics.width - 6;
            if (visualStyle.showPlaybackRate) {
                const rateMetrics = ctx.measureText(visualStyle.playbackRateLabel);
                const rateX = gainX - rateMetrics.width - 8;
                ctx.fillText(visualStyle.playbackRateLabel, rateX, clipTop + 9);
            }
            ctx.fillText(visualStyle.gainLabel, gainX, clipTop + 9);
        }

        if (!clip.isRenaming && visualStyle.showName && visualStyle.displayName.length > 0) {
            const textStartX = clipLeft + visualStyle.leadingControlsWidth;
            const textEndX = visualStyle.showGainLabel
                ? clipLeft + clipWidth - visualStyle.trailingReservePx + 4
                : clipLeft + clipWidth - 8;
            const availableWidth = Math.max(0, textEndX - textStartX);
            if (availableWidth > 12) {
                ctx.save();
                ctx.beginPath();
                ctx.rect(textStartX, clipTop, availableWidth, headerHeight);
                ctx.clip();
                ctx.fillStyle = visualStyle.textFill;
                ctx.font = `12px ${fontFamily}`;
                ctx.textBaseline = "middle";
                ctx.fillText(visualStyle.displayName, textStartX, clipTop + 9);
                ctx.restore();
            }
        }

        if (clip.fadeInPx > 0) {
            // 淡入区压暗：音量从 0 爬升，视觉上"还没到全响"就该更暗。
            // （0.45 在短渐变时像突兀的黑柱，降到 0.32。）
            ctx.fillStyle = "rgba(0, 0, 0, 0.32)";
            ctx.fillRect(clipLeft, bodyTop, Math.min(clipWidth, clip.fadeInPx), bodyHeight);
            // 渐变曲线 = 半透明白：在压暗区与彩色块上都稳定可见（REAPER 式）。
            ctx.strokeStyle = "rgba(255, 255, 255, 0.65)";
            ctx.lineWidth = 1.2;
            drawFadeCurveStroke(ctx, {
                leftPx: clipLeft,
                topPx: bodyTop,
                widthPx: Math.min(clipWidth, clip.fadeInPx),
                heightPx: bodyHeight,
                shape: clip.fadeInShape,
                dir: clip.fadeInDir,
                mode: "in",
            });
        }
        if (clip.fadeOutPx > 0) {
            ctx.fillStyle = "rgba(0, 0, 0, 0.32)";
            ctx.fillRect(
                clipLeft + clipWidth - Math.min(clipWidth, clip.fadeOutPx),
                bodyTop,
                Math.min(clipWidth, clip.fadeOutPx),
                bodyHeight,
            );
            ctx.strokeStyle = "rgba(255, 255, 255, 0.65)";
            ctx.lineWidth = 1.2;
            drawFadeCurveStroke(ctx, {
                leftPx: clipLeft + clipWidth - Math.min(clipWidth, clip.fadeOutPx),
                topPx: bodyTop,
                widthPx: Math.min(clipWidth, clip.fadeOutPx),
                heightPx: bodyHeight,
                shape: clip.fadeOutShape,
                dir: clip.fadeOutDir,
                mode: "out",
            });
        }

        // ── SnapOffset（吸附偏移）三角标记 ────────────────────────
        // 左下角等腰直角三角形（直角在左下，◣）。**左侧竖直边严格对齐
        // 偏移位置**（与波形内橙色竖虚线同 x）—— 不做宽度回退钳制；
        // 三角靠近/越过 Clip 末尾的部分按 Clip 矩形裁剪。offset=0 时
        // 半透明贴在起点作为可发现性提示。
        if (clipWidth >= 12 && clipHeight >= 14) {
            const offsetPx = Math.max(0, Number(clip.snapOffsetPx) || 0);
            const triX = clipLeft + offsetPx;
            const triYBottom = clipTop + clipHeight;
            const size = SNAP_OFFSET_HANDLE_SIZE_PX;
            ctx.save();
            ctx.beginPath();
            ctx.rect(clipLeft, clipTop, clipWidth, clipHeight);
            ctx.clip();
            ctx.globalAlpha *= offsetPx > 1e-9 ? 0.95 : 0.55;
            ctx.beginPath();
            ctx.moveTo(triX, triYBottom - size);
            ctx.lineTo(triX, triYBottom);
            ctx.lineTo(triX + size, triYBottom);
            ctx.closePath();
            // 填充色随 clip 体感亮度取深/浅（timelineCanvasStyle 派生），
            // 写死的黄色在绿/黄轨道色块上会隐身。
            ctx.fillStyle = visualStyle.snapOffsetTriFill;
            ctx.fill();
            ctx.strokeStyle = visualStyle.snapOffsetTriStroke;
            ctx.lineWidth = 1;
            ctx.stroke();
            ctx.restore();
        }

        ctx.restore();
    }
}
