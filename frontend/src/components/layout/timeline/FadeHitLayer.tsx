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
    buildSingleFadeInfoText,
    type FadeLabelLookup,
    type FadeLengthFormatContext,
} from "./fadeTooltipText";

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

    return (
        <>
            {targets.map((target, index) => {
                const isLine = target.kind === "line";
                const side: "in" | "out" = target.type === "fade_in" ? "in" : "out";
                // 信息浮标对所有淡变编辑目标生效（包络线 + 区域边缘竖线）。
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
                        data-tooltip={tooltip}
                        onPointerDown={(e) => {
                            if (e.button !== 0) return;
                            // 形状循环点击仅对包络线命中生效（边缘竖线是长度语义）。
                            if (
                                isLine &&
                                onShapeCycleClick &&
                                shapeCycleKb != null &&
                                !isNoneBinding(shapeCycleKb) &&
                                cycleModifierHeld(shapeCycleKb, e.nativeEvent)
                            ) {
                                e.preventDefault();
                                e.stopPropagation();
                                onShapeCycleClick(side);
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
