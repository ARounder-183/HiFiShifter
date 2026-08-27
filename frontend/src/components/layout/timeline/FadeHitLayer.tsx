/**
 * FadeHitLayer — 淡入淡出"画线即控件"的 DOM 命中层。
 *
 * 由 buildFadeHitTargets 生成的两类命中区域：
 *   1. 包络线命中块（沿包络线采样的小方块）；
 *   2. 区域边缘竖线命中条（淡化区域最外侧竖线）。
 *
 * 悬停信息：所有与编辑淡变相关的命中目标都通过项目统一 ToolTips
 * （data-tooltip → AppTooltipProvider）展示该侧完整信息块（类型 / 长度 /
 * 曲率），文本由 fadeTooltipText.buildSingleFadeInfoText 拼装；长度按时间轴
 * 主/副时间单位以**相对时长**格式化（零基点，无工程原点偏移）。
 *
 * 交互：
 * - 左键拖包络线 = 调长度；按住 modifier.fadeCurvatureDrag（默认 Alt）拖 = 调曲率；
 *   拖拽中的实时浮标由 useEditDrag 发布（同一 .app-tooltip 样式）。
 * - modifier.fadeShapeCycleClick + 左键点击包络线 = 循环切换类型并重置曲率。
 */
import React from "react";
import { isNoneBinding } from "../../../features/keybindings/keybindingsSlice";
import type { Keybinding } from "../../../features/keybindings/types";
import { buildFadeHitTargets } from "./fadeHitTargets";
import {
    buildSingleFadeInfoContent,
    buildSingleFadeInfoText,
    publishFadeRichTooltip,
    type FadeLabelLookup,
    type FadeLengthFormatContext,
} from "./fadeTooltipText";
import { requestOpenFadeContextMenu, requestResetFadeCurvature } from "./fadeContextMenuBus";
import { noteFadeLinePointerDown } from "./hooks/fadeLineClickGesture";

export const FadeHitLayer = React.memo(function FadeHitLayer({
    clipLeftPx,
    clipWidthPx,
    bodyTop,
    bodyHeight,
    fadeInPx,
    fadeOutPx,
    fadeInShape,
    fadeInDir,
    fadeOutShape,
    fadeOutDir,
    effectiveFadeInSec,
    effectiveFadeOutSec,
    formatCtx,
    t,
    clipXFrom,
    clipXTo,
    zIndex = 40,
    shapeCycleKb,
    clipId,
    onShapeCycleClick,
    onFadeInPointerDown,
    onFadeOutPointerDown,
}: {
    clipLeftPx: number;
    clipWidthPx: number;
    bodyTop: number;
    bodyHeight: number;
    fadeInPx: number;
    fadeOutPx: number;
    fadeInShape: number;
    fadeInDir: number;
    fadeOutShape: number;
    fadeOutDir: number;
    /** 有效淡化长度（秒）：自动交叉淡化覆盖手动值后的真实编辑对象。 */
    effectiveFadeInSec: number;
    effectiveFadeOutSec: number;
    /** 相对时长时间上下文（主/副单位等），供长度行格式化。 */
    formatCtx: FadeLengthFormatContext;
    /** i18n 文案查询（由宿主组件提供）。 */
    t: (key: string) => string;
    clipXFrom?: number;
    clipXTo?: number;
    zIndex?: number;
    /** 形状循环键绑定；null / "无" 绑定时禁用点击切换。 */
    shapeCycleKb: Keybinding | null;
    /** 所属 clip id（右键菜单载荷用）。 */
    clipId: string;
    /** 修饰键下左键点击包络线 → 循环切换该侧曲线类型。 */
    onShapeCycleClick?: (side: "in" | "out") => void;
    onFadeInPointerDown: (e: React.PointerEvent<HTMLDivElement>) => void;
    onFadeOutPointerDown: (e: React.PointerEvent<HTMLDivElement>) => void;
}) {
    const targets = buildFadeHitTargets({
        clipLeftPx,
        clipWidthPx,
        bodyTop,
        bodyHeight,
        fadeInPx,
        fadeOutPx,
        fadeInShape,
        fadeInDir,
        fadeOutShape,
        fadeOutDir,
        clipXFrom,
        clipXTo,
    });

    if (targets.length === 0) return null;

    // 预构建每侧的富内容节点（同侧命中块共享同一引用；注册表按元素分键）。
    const richIn = buildSingleFadeInfoContent({
        isOut: false,
        shape: fadeInShape,
        dir: fadeInDir,
        lengthSec: effectiveFadeInSec,
        formatCtx,
        t: t as unknown as FadeLabelLookup,
    });
    const richOut = buildSingleFadeInfoContent({
        isOut: true,
        shape: fadeOutShape,
        dir: fadeOutDir,
        lengthSec: effectiveFadeOutSec,
        formatCtx,
        t: t as unknown as FadeLabelLookup,
    });
    // 回调 ref：命中块挂载/更新时把富内容注册进 AppTooltipProvider。
    const publishRef =
        (content: React.ReactNode) =>
        (element: HTMLDivElement | null): void => {
            publishFadeRichTooltip(element, content);
        };

    return (
        <>
            {targets.map((target, index) => {
                const isLine = target.kind === "line";
                const side: "in" | "out" = target.type === "fade_in" ? "in" : "out";
                // data-tooltip 保留纯文本版本（无 JS 场景回退）；悬停实际展示
                // 富内容（首行为内联曲线图标）。
                const tooltip = buildSingleFadeInfoText({
                    isOut: target.type === "fade_out",
                    shape: target.type === "fade_in" ? fadeInShape : fadeOutShape,
                    dir: target.type === "fade_in" ? fadeInDir : fadeOutDir,
                    lengthSec:
                        target.type === "fade_in"
                            ? effectiveFadeInSec
                            : effectiveFadeOutSec,
                    formatCtx,
                    t: t as unknown as FadeLabelLookup,
                });
                return (
                    <div
                        key={`${target.kind}-${target.type}-${index}`}
                        ref={publishRef(
                            target.type === "fade_in" ? richIn : richOut,
                        )}
                        className="absolute"
                        style={{
                            left: target.left,
                            top: target.top,
                            width: target.width,
                            height: target.height,
                            zIndex,
                            cursor:
                                target.type === "fade_in" ? "nwse-resize" : "nesw-resize",
                        }}
                        data-hs-fade-hit={target.type}
                        data-hs-fade-y={String(target.top)}
                        data-hs-fade-line={String(target.kind === "line")}
                        data-hs-clip-id={clipId}
                        data-tooltip={tooltip}
                        onContextMenu={(e) => {
                            // 右键落在交叉点抓手（更高层的 OverlapEditLayer）
                            // 上时不在这里弹菜单，由抓手自身分发双侧载荷。
                            const grab = (e.target as Element | null)?.closest?.(
                                "[data-hs-crossfade-grip]",
                            );
                            if (grab) return;
                            e.preventDefault();
                            e.stopPropagation();
                            requestOpenFadeContextMenu({
                                clientX: e.clientX,
                                clientY: e.clientY,
                                primary: {
                                    clipId:
                                        e.currentTarget.dataset.hsClipId ?? "",
                                    isOut: target.type === "fade_out",
                                    shape:
                                        target.type === "fade_in"
                                            ? fadeInShape
                                            : fadeOutShape,
                                    dir:
                                        target.type === "fade_in"
                                            ? fadeInDir
                                            : fadeOutDir,
                                    lengthSec:
                                        target.type === "fade_in"
                                            ? effectiveFadeInSec
                                            : effectiveFadeOutSec,
                                },
                                secondary: null,
                            });
                        }}
                        onPointerDown={(e) => {
                            if (e.button !== 0) return;
                            // 形状循环点击仅对包络线命中生效（边缘竖线是长度语义）。
                            // 循环键与长度拖拽共用时无法在按下瞬间区分意图：
                            // 延后判定 —— 拖动超阈值 = 长度拖拽；未拖动松开 = 循环。
                            const cycleHeld =
                                onShapeCycleClick &&
                                shapeCycleKb != null &&
                                !isNoneBinding(shapeCycleKb) &&
                                cycleModifierHeld(shapeCycleKb, e.nativeEvent);
                            if (isLine && cycleHeld) {
                                e.preventDefault();
                                e.stopPropagation();
                                const startX = e.clientX;
                                const startY = e.clientY;
                                const pointerId = e.pointerId;
                                const el = e.currentTarget;
                                let dragStarted = false;
                                const onMove = (ev: PointerEvent) => {
                                    if (ev.pointerId !== pointerId || dragStarted) return;
                                    const dx = ev.clientX - startX;
                                    const dy = ev.clientY - startY;
                                    if (dx * dx + dy * dy < 9) return;
                                    dragStarted = true;
                                    // 意图 = 长度拖拽：交给原有淡变拖拽起手。
                                    (
                                        target.type === "fade_in"
                                            ? onFadeInPointerDown
                                            : onFadeOutPointerDown
                                    )({
                                        button: 0,
                                        pointerId,
                                        clientX: ev.clientX,
                                        clientY: ev.clientY,
                                        currentTarget: el,
                                        nativeEvent: ev,
                                        altKey: ev.altKey,
                                        ctrlKey: ev.ctrlKey,
                                        metaKey: ev.metaKey,
                                        shiftKey: ev.shiftKey,
                                        preventDefault() {},
                                        stopPropagation() {},
                                    } as unknown as React.PointerEvent<HTMLDivElement>);
                                };
                                const onUp = (ev: PointerEvent) => {
                                    window.removeEventListener("pointermove", onMove, true);
                                    window.removeEventListener("pointerup", onUp, true);
                                    window.removeEventListener("pointercancel", onUp, true);
                                    if (!dragStarted && ev.pointerId === pointerId) {
                                        // 未拖动 = 点击：循环切换到下一个形状。
                                        onShapeCycleClick(side);
                                    }
                                };
                                window.addEventListener("pointermove", onMove, true);
                                window.addEventListener("pointerup", onUp, true);
                                window.addEventListener("pointercancel", onUp, true);
                                return;
                            }
                            // 双击包络线 = 重置该曲线曲率到当前形状默认值
                            // （仅包络线本体；边缘竖线保持长度语义）。
                            // 检测用时间窗 + 目标键（pointerdown 的 detail 在
                            // 部分 WebView 恒为 0，不可靠）。
                            if (
                                isLine &&
                                !cycleHeld &&
                                noteFadeLinePointerDown(`${clipId}:${target.type}`) ===
                                    "double"
                            ) {
                                e.preventDefault();
                                e.stopPropagation();
                                requestResetFadeCurvature({
                                    sides: [{ clipId, isOut: target.type === "fade_out" }],
                                });
                                return;
                            }
                            if (target.type === "fade_in") onFadeInPointerDown(e);
                            else onFadeOutPointerDown(e);
                        }}
                    />
                );
            })}
        </>
    );
});

/**
 * pointerdown 现场判定循环修饰键是否按下。按下瞬间的事件本身是最可靠
 * 的信号源；modifierWatcher 在手势全程持续自愈全局快照供后续帧使用。
 */
function cycleModifierHeld(kb: Keybinding, event: PointerEvent): boolean {
    const requiredCtrl =
        kb.modifierOnly === true && kb.key === "control" ? true : Boolean(kb.ctrl);
    const requiredAlt =
        kb.modifierOnly === true && kb.key === "alt" ? true : Boolean(kb.alt);
    const requiredShift =
        kb.modifierOnly === true && kb.key === "shift" ? true : Boolean(kb.shift);
    return (
        (!requiredCtrl || event.ctrlKey || event.metaKey) &&
        (!requiredAlt || event.altKey) &&
        (!requiredShift || event.shiftKey)
    );
}
