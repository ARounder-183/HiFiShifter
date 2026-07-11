/**
 * timelineCanvasRenderer.ts - 主时间轴 canvas 的 Clip 绘制入口。
 *
 * 主要内容：
 * - drawTimelineCanvas：清屏 + 逐 clip 绘制 header / body / accent bar /
 *   控件 badge / 名称 / fade 曲线。
 *
 * 与其他模块的关系：
 * - 由 TimelineCanvasViewport 在 rAF 中调用。
 * - 颜色和几何参数全部来自 buildTimelineClipVisualStyle，本模块不再做颜色再加工。
 *
 * 维护说明：
 * - 2026-06-30 重做为 Logic Pro 扁平风格：
 *   * 取消 body 的硬黑分隔线，改为半透明白色细线（headerSeparatorFill）。
 *   * 新增左侧 accent bar（满饱和 trackColor 派生），承担色相识别。
 *   * 选中态用 1px 内描边（饱和色），非选中态描边极淡（视觉降噪）。
 *   * fade 区不再叠生硬的 0.05 白矩形，改为 fade 曲线下半透明三角阴影。
 */
import {
    buildTimelineClipVisualStyle,
    computeTimelineFadeShadeRange,
    resolveFontFamily,
} from "./timelineCanvasStyle.js";
import { fadeCurveGain } from "../paths.js";

function drawFadeCurveStroke(
    ctx: CanvasRenderingContext2D,
    args: {
        leftPx: number;
        topPx: number;
        widthPx: number;
        heightPx: number;
        curve: "linear" | "sine" | "exponential" | "logarithmic" | "scurve";
        mode: "in" | "out";
    },
): void {
    const widthPx = Math.max(1, args.widthPx);
    const heightPx = Math.max(1, args.heightPx);
    const steps = Math.max(12, Math.min(48, Math.round(widthPx / 8)));
    ctx.beginPath();
    for (let index = 0; index < steps; index += 1) {
        const t = index / Math.max(1, steps - 1);
        const x = args.leftPx + t * widthPx;
        const gain =
            args.mode === "in" ? fadeCurveGain(t, args.curve) : fadeCurveGain(1 - t, args.curve);
        const y = args.topPx + heightPx * (1 - gain);
        if (index === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }
    }
    ctx.stroke();
}

/**
 * 在 fade 区域绘制半透明阴影：fade 曲线之下的部分用低 alpha 覆盖，
 * 视觉上能直观感知淡入/淡出的能量包络，比矩形覆盖更精致也更接近 Logic 风格。
 */
function fillFadeShade(
    ctx: CanvasRenderingContext2D,
    args: {
        leftPx: number;
        topPx: number;
        widthPx: number;
        heightPx: number;
        curve: "linear" | "sine" | "exponential" | "logarithmic" | "scurve";
        mode: "in" | "out";
    },
): void {
    const widthPx = Math.max(1, args.widthPx);
    const heightPx = Math.max(1, args.heightPx);
    const steps = Math.max(12, Math.min(48, Math.round(widthPx / 8)));
    ctx.beginPath();
    ctx.moveTo(args.leftPx, args.topPx + heightPx);
    for (let index = 0; index < steps; index += 1) {
        const t = index / Math.max(1, steps - 1);
        const x = args.leftPx + t * widthPx;
        const gain =
            args.mode === "in" ? fadeCurveGain(t, args.curve) : fadeCurveGain(1 - t, args.curve);
        const y = args.topPx + heightPx * (1 - gain);
        ctx.lineTo(x, y);
    }
    ctx.lineTo(args.leftPx + widthPx, args.topPx + heightPx);
    ctx.closePath();
    ctx.fill();
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
            fadeInCurve: "linear" | "sine" | "exponential" | "logarithmic" | "scurve";
            fadeOutCurve: "linear" | "sine" | "exponential" | "logarithmic" | "scurve";
            selected: boolean;
            muted: boolean;
            gain: number;
            groupId?: string;
            playbackRate?: number;
            name: string;
            isMidiClip?: boolean;
            trackColor?: string;
        }>;
        fontFamily?: string;
        activeGroupIds?: Set<string>;
        disabledGroupIds?: string[];
    },
): void {
    const fontFamily = args.fontFamily || resolveFontFamily();

    ctx.clearRect(0, 0, args.width, args.height);

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
            playbackRate: clip.playbackRate ?? 1,
            name: clip.name,
            fontFamily,
            isPitchAdjustment: clip.isMidiClip,
            groupId: clip.groupId,
            isGroupActive,
            isGroupDisabled,
        });
        const fadeShadeRange = computeTimelineFadeShadeRange({
            widthPx: clipWidth,
            fadeInPx: clip.fadeInPx,
            fadeOutPx: clip.fadeOutPx,
        });

        ctx.save();
        ctx.globalAlpha = visualStyle.mutedAlpha;

        // ── 1. body / header 底色（一次性铺满，不透明，直角扁平）──
        ctx.fillStyle = visualStyle.headerFill;
        ctx.fillRect(clipLeft, clipTop, clipWidth, headerHeight);
        ctx.fillStyle = visualStyle.bodyFill;
        ctx.fillRect(clipLeft, bodyTop, clipWidth, bodyHeight);

        // ── 2. 左侧 accent bar：满饱和 trackColor 细条，承担色相识别 ──
        const accentWidth = Math.min(visualStyle.accentBarWidthPx, clipWidth);
        if (accentWidth > 0) {
            ctx.fillStyle = visualStyle.accentBarFill;
            ctx.fillRect(clipLeft, clipTop, accentWidth, clipHeight);
        }

        // ── 3. fade 区柔和阴影（从底部向 fade 曲线内部填半透明黑）──
        if (fadeShadeRange) {
            ctx.fillStyle = "rgba(0, 0, 0, 0.14)";
            ctx.fillRect(
                clipLeft + fadeShadeRange.startPx,
                bodyTop,
                Math.max(1, fadeShadeRange.endPx - fadeShadeRange.startPx),
                bodyHeight,
            );
        }

        // ── 4. header 与 body 的分隔线：极淡白色，仅提示分层 ──
        ctx.fillStyle = visualStyle.headerSeparatorFill;
        ctx.fillRect(clipLeft, clipTop + headerHeight, clipWidth, 1);

        // ── 5. 整体描边：选中态饱和色 1px 内描边，非选中态淡黑细线 ──
        ctx.strokeStyle = visualStyle.borderStroke;
        ctx.lineWidth = 1;
        // 整 clip 描边（含 header），半像素对齐避免模糊
        ctx.strokeRect(
            clipLeft + 0.5,
            clipTop + 0.5,
            Math.max(0, clipWidth - 1),
            Math.max(0, clipHeight - 1),
        );

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

        if (visualStyle.showName && visualStyle.displayName.length > 0) {
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

        // ── 6. fade in/out 阴影 + 曲线（半透明三角形 + 1px 白线）──
        if (clip.fadeInPx > 0) {
            ctx.fillStyle = "rgba(0, 0, 0, 0.28)";
            fillFadeShade(ctx, {
                leftPx: clipLeft,
                topPx: bodyTop,
                widthPx: Math.min(clipWidth, clip.fadeInPx),
                heightPx: bodyHeight,
                curve: clip.fadeInCurve,
                mode: "in",
            });
            ctx.strokeStyle = "rgba(255, 255, 255, 0.7)";
            ctx.lineWidth = 1;
            drawFadeCurveStroke(ctx, {
                leftPx: clipLeft,
                topPx: bodyTop + 1,
                widthPx: Math.min(clipWidth, clip.fadeInPx),
                heightPx: Math.max(1, bodyHeight - 2),
                curve: clip.fadeInCurve,
                mode: "in",
            });
        }
        if (clip.fadeOutPx > 0) {
            ctx.fillStyle = "rgba(0, 0, 0, 0.28)";
            fillFadeShade(ctx, {
                leftPx: clipLeft + clipWidth - Math.min(clipWidth, clip.fadeOutPx),
                topPx: bodyTop,
                widthPx: Math.min(clipWidth, clip.fadeOutPx),
                heightPx: bodyHeight,
                curve: clip.fadeOutCurve,
                mode: "out",
            });
            ctx.strokeStyle = "rgba(255, 255, 255, 0.7)";
            ctx.lineWidth = 1;
            drawFadeCurveStroke(ctx, {
                leftPx: clipLeft + clipWidth - Math.min(clipWidth, clip.fadeOutPx),
                topPx: bodyTop + 1,
                widthPx: Math.min(clipWidth, clip.fadeOutPx),
                heightPx: Math.max(1, bodyHeight - 2),
                curve: clip.fadeOutCurve,
                mode: "out",
            });
        }

        ctx.restore();
    }
}
