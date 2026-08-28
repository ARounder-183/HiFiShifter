/**
 * FadeContextMenu — 淡入淡出包络专属上下文菜单。
 *
 * 与 Clip 上下文菜单完全独立：**只**包含当前聚焦侧的
 *   1. 七个 REAPER 形状预设（图标 + data-tooltip 名称，选中态高亮）；
 *   2. 曲率滑块（原生 range；支持滚轮步进与 modifier.paramFineAdjust 微调；
 *      onChange 实时提交）。
 *
 * 右键目标为交叉点抓手时同时渲染两列 —— 前者淡出、后者淡入，
 * 各自独立的形状行与滑块。
 *
 * 悬浮 ToolTips 抑制：菜单打开期间设置全局抑制标志
 * （HS_FADE_TOOLTIP_SUPPRESS），AppTooltipProvider 看到它即不再显示
 * 淡变信息浮标，避免与菜单互相遮挡。
 */
import React, { useEffect, useLayoutEffect, useRef } from "react";
import { createPortal } from "react-dom";
import { useI18n } from "../../../i18n/I18nProvider";
import type { MessageKey } from "../../../i18n/messages";
import {
    formatKeybinding,
    isModifierActive,
    isNoneBinding,
    selectKeybinding,
} from "../../../features/keybindings/keybindingsSlice";
import { useAppSelector } from "../../../app/hooks";
import {
    defaultFadeDirFor,
    FADE_PRESETS,
    fadeGainSigned,
    solveNearestCurveDir,
} from "./reaperFade";
import { FadeShapeIcon } from "./FadeShapeIcon";
import type { FadeLabelLookup } from "./fadeTooltipText";

/** 悬浮淡变 ToolTips 全局抑制标志（由本模块导出开关函数）。 */
let fadeTooltipSuppressed = false;

export function setFadeTooltipSuppressed(suppressed: boolean): void {
    fadeTooltipSuppressed = suppressed;
}

export function isFadeTooltipSuppressed(): boolean {
    return fadeTooltipSuppressed;
}

export const FADE_CONTEXT_MENU_ATTR = "data-hs-fade-context-menu";

const SHAPE_LABEL_KEYS: Record<number, MessageKey> = {
    0: "fade_shape_linear",
    1: "fade_shape_fast_start",
    2: "fade_shape_fast_end",
    3: "fade_shape_fast_start_steep",
    4: "fade_shape_fast_end_steep",
    5: "fade_shape_slow_start_end",
    6: "fade_shape_slow_start_end_steep",
};

export type FadeContextSide = {
    clipId: string;
    isOut: boolean;
    shape: number;
    dir: number;
    lengthSec: number;
};

/** 曲率滑块微调步长（dir 单位）。 */
const CURVATURE_WHEEL_STEP = 0.05;
const CURVATURE_FINE_STEP = 0.01;
const CurvatureSlider: React.FC<{
    shape: number;
    dir: number;
    /** 淡出列：预览按淡出取向绘制（时间镜像 + σ 符号归一），与画布一致。 */
    isOut: boolean;
    onChange: (nextDir: number) => void;
}> = ({ shape, dir, isOut, onChange }) => {
    const fineAdjustKb = useAppSelector((state) =>
        selectKeybinding(state, "modifier.paramFineAdjust"),
    );
    const svgRef = useRef<SVGSVGElement | null>(null);
    const draggingRef = useRef(false);
    // 展示用的预览采样点（迷你曲线），随形状与曲率实时重绘。
    // mode='out' 时 fadeGainSigned 内部完成 σ 符号归一与时间镜像，
    // 画出的曲线方向与该侧在 Clip 上看到的完全一致（淡出=左上→右下）。
    const preview = React.useMemo(() => {
        const size = 34;
        const pad = 2;
        const inner = size - pad * 2;
        const steps = 24;
        const pts: string[] = [];
        for (let i = 0; i < steps; i += 1) {
            const p = i / (steps - 1);
            const gain = fadeGainSigned(shape, dir, isOut ? "out" : "in", p);
            pts.push(`${(pad + p * inner).toFixed(2)},${(pad + (1 - gain) * inner).toFixed(2)}`);
        }
        return pts.join(" ");
    }, [shape, dir, isOut]);

    // ── 预览图直接拖拽调曲率（无需修饰键）──────────────────────────
    // 把指针位置投影回曲线空间：x → 进度 t，y → 目标增益，再用
    // solveDirAt 反解 dir。pointer capture 挂在 svg 自身，
    // React 重渲染不会丢失捕获。
    const applyPointerToCurve = (clientX: number, clientY: number): void => {
        const el = svgRef.current;
        if (!el) return;
        const rect = el.getBoundingClientRect();
        const size = rect.width; // 正方形
        const pad = 2;
        const inner = Math.max(1, size - pad * 2);
        const t = Math.min(1, Math.max(0, (clientX - rect.left - pad) / inner));
        const yWithin = Math.min(inner, Math.max(0, clientY - rect.top - pad));
        const targetGain = 1 - yWithin / inner;
        const next = solveNearestCurveDir({
            shape,
            dir,
            mode: isOut ? "out" : "in",
            pointerX01: t,
            pointerY01: targetGain,
            aspectYOverX: 1,
        }).dir;
        onChange(Number(next.toFixed(2)));
    };

    return (
        <div className="flex items-center gap-2 px-2 py-1">
            <svg
                ref={svgRef}
                width={34}
                height={34}
                viewBox="0 0 34 34"
                aria-hidden="true"
                onPointerDown={(e) => {
                    if (e.button !== 0) return;
                    e.preventDefault();
                    e.stopPropagation();
                    draggingRef.current = true;
                    try {
                        (e.currentTarget as SVGSVGElement).setPointerCapture(e.pointerId);
                    } catch {
                        // 捕获失败时仍可通过 move-in-bounds 工作。
                    }
                    applyPointerToCurve(e.clientX, e.clientY);
                }}
                onPointerMove={(e) => {
                    if (!draggingRef.current) return;
                    applyPointerToCurve(e.clientX, e.clientY);
                }}
                onPointerUp={() => {
                    draggingRef.current = false;
                }}
                onPointerCancel={() => {
                    draggingRef.current = false;
                }}
                style={{ cursor: "crosshair", touchAction: "none", flexShrink: 0 }}
            >
                <polyline
                    points={preview}
                    fill="none"
                    stroke="currentColor"
                    strokeWidth={1.5}
                    strokeLinecap="round"
                />
            </svg>
            <input
                type="range"
                min={-1}
                max={1}
                step={0.01}
                value={dir}
                onWheel={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    const fine = isModifierActive(fineAdjustKb, e.nativeEvent);
                    const step = fine ? CURVATURE_FINE_STEP : CURVATURE_WHEEL_STEP;
                    const direction = e.deltaY < 0 ? 1 : -1;
                    const next = Math.max(-1, Math.min(1, dir + direction * step));
                    onChange(Number(next.toFixed(2)));
                }}
                onChange={(e) => onChange(Number(e.currentTarget.value))}
                style={{ flex: 1 }}
            />
            <span className="text-[11px] tabular-nums" style={{ minWidth: 44, textAlign: "right" }}>
                {(dir >= 0 ? "+" : "") + dir.toFixed(2)}
            </span>
        </div>
    );
};

const ShapeRow: React.FC<{
    currentShape: number;
    /** 淡出列：图标水平镜像，方向与该侧画布曲线一致。 */
    isOut?: boolean;
    onSelectShape: (shape: number) => void;
    t: FadeLabelLookup;
}> = ({ currentShape, isOut = false, onSelectShape, t }) => (
    <div className="px-2 py-1.5 flex items-center gap-1">
        {FADE_PRESETS.map((preset) => {
            const key = SHAPE_LABEL_KEYS[preset.shape];
            const selected = Math.trunc(currentShape) === preset.shape;
            return (
                <button
                    key={key}
                    data-tooltip={t(key)}
                    className={`p-0.5 rounded transition-colors leading-none ${
                        selected
                            ? "bg-qt-highlight text-white"
                            : "bg-qt-button hover:bg-qt-button-hover text-qt-text/80"
                    }`}
                    onClick={(e) => {
                        e.stopPropagation();
                        onSelectShape(preset.shape);
                    }}
                >
                    <FadeShapeIcon shape={preset.shape} size={16} mirrored={isOut} />
                </button>
            );
        })}
    </div>
);

const SideColumn: React.FC<{
    side: FadeContextSide;
    isOut: boolean;
    onShapeChange: (clipId: string, isOut: boolean, shape: number) => void;
    onDirChange: (clipId: string, isOut: boolean, dir: number) => void;
    t: FadeLabelLookup;
}> = ({ side, isOut, onShapeChange, onDirChange, t }) => {
    return (
        <div className="min-w-[210px]">
            {/* 形状选择：切换即重置该侧曲率为形状默认值。 */}
            <ShapeRow
                currentShape={side.shape}
                isOut={isOut}
                onSelectShape={(shape) => {
                    onShapeChange(side.clipId, side.isOut, shape);
                }}
                t={t}
            />
            {/* 曲率滑块：实时提交 dir。 */}
            <CurvatureSlider
                shape={Math.trunc(side.shape)}
                dir={side.dir}
                isOut={isOut}
                onChange={(nextDir) => onDirChange(side.clipId, side.isOut, nextDir)}
            />
        </div>
    );
};

export const FadeContextMenu: React.FC<{
    x: number;
    y: number;
    /** 主侧（右键命中的那条包络线）。 */
    primary: FadeContextSide;
    /** 交叉点命中时的第二侧（另一条包络线）；无则不渲染。 */
    secondary?: FadeContextSide | null;
    onClose: () => void;
    onShapeChange: (clipId: string, isOut: boolean, shape: number) => void;
    onDirChange: (clipId: string, isOut: boolean, dir: number) => void;
}> = ({ x, y, primary, secondary, onClose, onShapeChange, onDirChange }) => {
    const { t } = useI18n();
    const menuRef = useRef<HTMLDivElement>(null);
    // 底部提示展示用户实际配置的曲率修饰键（如 "Alt"）。
    const curvatureKb = useAppSelector((state) =>
        selectKeybinding(state, "modifier.fadeCurvatureDrag"),
    );
    const keysText =
        curvatureKb && !isNoneBinding(curvatureKb) ? formatKeybinding(curvatureKb, "") : "";
    const curvatureHint = (t("fade_menu_curvature_hint") as string).replace("{keys}", keysText);

    // 视口夹紧（同 ClipContextMenu 规则）。
    useLayoutEffect(() => {
        const el = menuRef.current;
        if (!el) return;
        const rect = el.getBoundingClientRect();
        if (rect.right > window.innerWidth)
            el.style.left = `${Math.max(0, window.innerWidth - rect.width)}px`;
        if (rect.bottom > window.innerHeight)
            el.style.top = `${Math.max(0, window.innerHeight - rect.height)}px`;
    }, [x, y]);

    // Escape 关闭；点击外部关闭（在 document 捕获阶段，Clip 菜单同款语义）。
    useEffect(() => {
        setFadeTooltipSuppressed(true);
        const onKey = (e: KeyboardEvent) => {
            if (e.key === "Escape") onClose();
        };
        const onPointerDownCapture = (e: PointerEvent) => {
            const target = e.target instanceof Element ? e.target : null;
            if (!target?.closest?.(`[${FADE_CONTEXT_MENU_ATTR}]`)) {
                onClose();
            }
        };
        window.addEventListener("keydown", onKey);
        document.addEventListener("pointerdown", onPointerDownCapture, true);
        return () => {
            setFadeTooltipSuppressed(false);
            window.removeEventListener("keydown", onKey);
            document.removeEventListener("pointerdown", onPointerDownCapture, true);
        };
    }, [onClose]);

    const labelFor = (side: FadeContextSide) => (side.isOut ? t("fade_out") : t("fade_in"));

    return createPortal(
        <div
            ref={menuRef}
            role="menu"
            {...{ [FADE_CONTEXT_MENU_ATTR]: "1" }}
            data-hs-floating-menu="1"
            data-hs-context-menu="1"
            className="fixed z-[999] min-w-[220px] rounded border border-qt-border bg-qt-window text-qt-text shadow-lg py-1"
            style={{ left: x, top: y }}
            onContextMenu={(e) => e.preventDefault()}
            onPointerDown={(e) => e.stopPropagation()}
        >
            {secondary ? (
                // 交叉点：双列 —— 先前者淡出、后后者淡入。
                <>
                    <div className="px-2 py-1 text-[10px] text-qt-text/50 select-none">
                        {labelFor(primary)}
                    </div>
                    <SideColumn
                        side={primary}
                        isOut={primary.isOut}
                        onShapeChange={onShapeChange}
                        onDirChange={onDirChange}
                        t={(key) => t(key as MessageKey)}
                    />
                    <div className="my-1 border-t border-qt-divider" />
                    <div className="px-2 py-1 text-[10px] text-qt-text/50 select-none">
                        {labelFor(secondary)}
                    </div>
                    <SideColumn
                        side={secondary}
                        isOut={secondary.isOut}
                        onShapeChange={onShapeChange}
                        onDirChange={onDirChange}
                        t={(key) => t(key as MessageKey)}
                    />
                </>
            ) : (
                <>
                    <div className="px-2 py-1 text-[10px] text-qt-text/50 select-none">
                        {labelFor(primary)}
                    </div>
                    <SideColumn
                        side={primary}
                        isOut={primary.isOut}
                        onShapeChange={onShapeChange}
                        onDirChange={onDirChange}
                        t={(key) => t(key as MessageKey)}
                    />
                </>
            )}
            {/* 形状切换重置曲率的语义提示（与 Clip 菜单一致的行为说明）。 */}
            <div className="px-2 pt-1 pb-0.5 text-[9px] text-qt-text/40 select-none">
                {curvatureHint}
            </div>
        </div>,
        document.body,
    );
};

/** 形状选择（带默认曲率重置）的统一提交工具，供宿主复用。 */
export function applyFadeShapeWithReset(
    submit: (patch: { shape: number; dir: number }) => void,
    shape: number,
    isOut: boolean,
): void {
    submit({ shape, dir: defaultFadeDirFor(shape, isOut) });
}

/** 供 AppTooltipProvider 挂载时读取的抑制查询源（模块级单例）。 */
export const fadeToolTipSuppress = {
    get isSuppressed(): boolean {
        return isFadeTooltipSuppressed();
    },
};
