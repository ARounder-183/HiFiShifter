// hs-interaction-exempt: 淡变曲率滑块位于自绘 SVG 面板内（与拖动/预览联动），已有滚轮 + 精细调整接线；原语不适用。
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
/* eslint-disable react-refresh/only-export-components -- 文件同时导出组件与 Hook/常量（刷新边界按文件粒度接受） */
import React, { useEffect, useLayoutEffect, useRef } from "react";
import { createPortal } from "react-dom";
import { useMenuKeyboard } from "../../../ui/useMenuKeyboard";
import { useNonPassiveWheel } from "../../../utils/useNonPassiveWheel";
import { registerDragAbort } from "../../../utils/gestureFocusGuard";
import { useI18n } from "../../../i18n/I18nProvider";
import { canEditFadeS, canSelectHostFadeShape } from "../../../services/hostCapabilities";
import type { MessageKey } from "../../../i18n/messages";
import {
    formatKeybindingList,
    isModifierActive,
    selectKeybinding,
    selectKeybindings,
} from "../../../features/keybindings/keybindingsSlice";
import { useAppSelector } from "../../../app/hooks";
import {
    defaultFadeDirFor,
    FADE_PRESETS,
    fadeGainSigned,
    solveNearestCurveAxes,
    solveNearestCurveDir,
} from "./reaperFade";
import { hostFadeGainForAxes } from "./hostFadeDisplay";
import type { HostFadeMetadata } from "../../../types/api";
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
    /** S 轴（REAPER ≥7.81 独有）；legacy 宿主与独立 App 恒为 0。 */
    s: number;
    lengthSec: number;
    /** 宿主淡化读数（插件模式）：预览与曲率投影要按画布那套曲线算。 */
    hostFades?: HostFadeMetadata;
};

/** 曲率滑块微调步长（dir 单位）。 */
const CURVATURE_WHEEL_STEP = 0.05;
const CURVATURE_FINE_STEP = 0.01;
const CurvatureSlider: React.FC<{
    shape: number;
    dir: number;
    /** S 轴当前值（continuous 宿主）。 */
    s: number;
    /** 该侧是不是 continuous 宿主（决定要不要摆第二根滑杆、投影是否二维）。 */
    continuous: boolean;
    /** 淡出列：预览按淡出取向绘制（时间镜像 + σ 符号归一），与画布一致。 */
    isOut: boolean;
    onChange: (nextDir: number) => void;
    onSChange: (nextS: number) => void;
}> = ({ shape, dir, s, continuous, isOut, onChange, onSChange }) => {
    const fineAdjustKb = useAppSelector((state) =>
        selectKeybinding(state, "modifier.paramFineAdjust"),
    );
    const svgRef = useRef<SVGSVGElement | null>(null);
    const draggingRef = useRef(false);
    /*
     * 曲率滑块的滚轮步进必须走**原生非被动**监听。
     *
     * 【为什么不能写在 JSX 的 onWheel 里】React 17+ 在 root 上把 `wheel` 注册为
     * passive，合成事件里的 `preventDefault()` 是空操作（浏览器打干预警告），
     * 于是滚轮**同时**改了曲率、又滚了底下的面板。`useNonPassiveWheel` 的文件头
     * 记录了这条；与 `PianoRollPanel` 的边缘平滑度滑块同因同解。
     */
    const attachCurvatureWheel = useNonPassiveWheel<HTMLInputElement>((e) => {
        e.preventDefault();
        e.stopPropagation();
        // 原生事件本身就是事件对象，修饰键直接读它（不是 `e.nativeEvent`）。
        const fine = isModifierActive(fineAdjustKb, e);
        const step = fine ? CURVATURE_FINE_STEP : CURVATURE_WHEEL_STEP;
        const direction = e.deltaY < 0 ? 1 : -1;
        const next = Math.max(-1, Math.min(1, dir + direction * step));
        onChange(Number(next.toFixed(2)));
    });
    const attachSWheel = useNonPassiveWheel<HTMLInputElement>((e) => {
        e.preventDefault();
        e.stopPropagation();
        const fine = isModifierActive(fineAdjustKb, e);
        const step = fine ? CURVATURE_FINE_STEP : CURVATURE_WHEEL_STEP;
        const direction = e.deltaY < 0 ? 1 : -1;
        const next = Math.max(-1, Math.min(1, s + direction * step));
        onSChange(Number(next.toFixed(2)));
    });
    /** 失焦守卫注销函数（拖拽期间非空；blur/抬起/取消/卸载时清理）。 */
    const unregisterAbortRef = useRef<(() => void) | null>(null);
    // 卸载兜底：菜单被外部关闭时（如点击外部）不能残留失焦注册。
    useEffect(() => {
        return () => {
            unregisterAbortRef.current?.();
            unregisterAbortRef.current = null;
        };
    }, []);
    // 展示用的预览采样点（迷你曲线），随形状与曲率实时重绘。
    // mode='out' 时 fadeGainSigned 内部完成 σ 符号归一与时间镜像，
    // 画出的曲线方向与该侧在 Clip 上看到的完全一致（淡出=左上→右下）。
    // 插件（新轴宿主）上走 `hostFadeGainForAxes`：预览必须是画布那条曲线，
    // 否则菜单里的小图与时间线上的包络不是同一条。
    const mode = isOut ? ("out" as const) : ("in" as const);
    // continuous 宿主用**本地** S（滑杆/拖拽刚改过的值）画预览，而不是宿主回读的
    // 旧值 —— 否则刚拖动 S 滑杆时小图不动，要等一次宿主往返才跟上。
    const hostS = continuous ? s : null;
    const preview = React.useMemo(() => {
        const size = 34;
        const pad = 2;
        const inner = size - pad * 2;
        const steps = 24;
        const pts: string[] = [];
        for (let i = 0; i < steps; i += 1) {
            const p = i / (steps - 1);
            const gain =
                hostS === null
                    ? fadeGainSigned(shape, dir, mode, p)
                    : hostFadeGainForAxes(mode, dir, hostS, p);
            pts.push(`${(pad + p * inner).toFixed(2)},${(pad + (1 - gain) * inner).toFixed(2)}`);
        }
        return pts.join(" ");
    }, [shape, dir, mode, hostS]);

    // ── 预览图直接拖拽调曲率（无需修饰键）──────────────────────────
    // 把指针位置投影回曲线空间：x → 进度 t，y → 目标增益，再取曲率族上的最近点。
    // pointer capture 挂在 svg 自身，React 重渲染不会丢失捕获。
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
        if (continuous) {
            // continuous：曲线是 (curvature, S) 二维的，必须二维求解，否则 S 族
            // 曲线上拖动永远只动 curvature（"拖了不跟手"）。两个分量一起提交。
            const solved = solveNearestCurveAxes({
                mode,
                curvature: dir,
                s,
                pointerX01: t,
                pointerY01: targetGain,
                aspectYOverX: 1,
                gainAt: (at, c, sv) => hostFadeGainForAxes(mode, c, sv, at),
            });
            onChange(Number(solved.curvature.toFixed(2)));
            onSChange(Number(solved.s.toFixed(2)));
            return;
        }
        const next = solveNearestCurveDir({
            shape,
            dir,
            mode,
            pointerX01: t,
            pointerY01: targetGain,
            aspectYOverX: 1,
        }).dir;
        onChange(Number(next.toFixed(2)));
    };

    return (
        <div className="flex flex-col gap-1">
            <div className="hs-menu__body flex items-center gap-2">
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
                        // 失焦取消：切屏期间 pointerup/pointercancel 不送达本窗口
                        //（svg 上的 pointer capture 不会在窗口外释放时派发事件），
                        // blur 必须复位 draggingRef —— 否则切回后任意鼠标移动都会
                        // 持续改写曲率。
                        unregisterAbortRef.current?.();
                        unregisterAbortRef.current = registerDragAbort(() => {
                            draggingRef.current = false;
                            unregisterAbortRef.current?.();
                            unregisterAbortRef.current = null;
                        });
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
                        unregisterAbortRef.current?.();
                        unregisterAbortRef.current = null;
                    }}
                    onPointerCancel={() => {
                        draggingRef.current = false;
                        unregisterAbortRef.current?.();
                        unregisterAbortRef.current = null;
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
                    ref={attachCurvatureWheel}
                    type="range"
                    className="qt-range"
                    min={-1}
                    max={1}
                    step={0.01}
                    value={dir}
                    onChange={(e) => onChange(Number(e.currentTarget.value))}
                    style={{ flex: 1 }}
                />
                <span
                    className="text-qt-xs tabular-nums"
                    style={{ minWidth: 44, textAlign: "right" }}
                >
                    {(dir >= 0 ? "+" : "") + dir.toFixed(2)}
                </span>
            </div>
            {/* S 轴滑杆：只有 REAPER ≥7.81 有这根轴。 */}
            {continuous ? (
                <div className="hs-menu__body flex items-center gap-2">
                    <span
                        className="text-qt-xs text-qt-text/70"
                        style={{ width: 34, textAlign: "center", flexShrink: 0 }}
                    >
                        S
                    </span>
                    <input
                        ref={attachSWheel}
                        type="range"
                        className="qt-range"
                        min={-1}
                        max={1}
                        step={0.01}
                        value={s}
                        onChange={(e) => onSChange(Number(e.currentTarget.value))}
                        style={{ flex: 1 }}
                    />
                    <span
                        className="text-qt-xs tabular-nums"
                        style={{ minWidth: 44, textAlign: "right" }}
                    >
                        {(s >= 0 ? "+" : "") + s.toFixed(2)}
                    </span>
                </div>
            ) : null}
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
    <div className="hs-menu__body flex items-center gap-1">
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
    onSChange: (clipId: string, isOut: boolean, s: number) => void;
    t: FadeLabelLookup;
}> = ({ side, isOut, onShapeChange, onDirChange, onSChange, t }) => {
    // 【新轴宿主上为什么也摆预设】REAPER ≥7.81 由 curvature/S 两个连续轴决定形状，
    // "预设 → (curvature, S)"的映射是实测出来的（`hostFadeAxes.ts`），七个预设各自
    // 对应一组确定坐标，所以按钮照摆、点了由 Rust 侧写两个分量。
    // 只有"宿主版本读不出来"时才退回曲率滑杆 —— 那时轴语义未知，不猜。
    const shapeSelectable = canSelectHostFadeShape();
    // S 轴只有 REAPER ≥7.81 有；旧轴宿主与独立 App 没有这根轴，不摆第二根滑杆。
    const continuous = side.hostFades?.curve_mode === "reaper_new" && canEditFadeS();
    return (
        <div className="min-w-[210px]">
            {/* 形状选择：切换即重置该侧曲率为形状默认值。 */}
            {shapeSelectable ? (
                <ShapeRow
                    currentShape={side.shape}
                    isOut={isOut}
                    onSelectShape={(shape) => {
                        onShapeChange(side.clipId, side.isOut, shape);
                    }}
                    t={t}
                />
            ) : (
                <div className="mb-1 text-qt-xs text-qt-text/70">
                    {t("fade_shape_axes_owned_by_host")}
                </div>
            )}
            {/* 曲率滑块：实时提交 dir；continuous 宿主下并列一根 S 滑杆。 */}
            <CurvatureSlider
                shape={Math.trunc(side.shape)}
                dir={side.dir}
                s={side.s}
                continuous={continuous}
                isOut={isOut}
                onChange={(nextDir) => onDirChange(side.clipId, side.isOut, nextDir)}
                onSChange={(nextS) => onSChange(side.clipId, side.isOut, nextS)}
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
    onSChange: (clipId: string, isOut: boolean, s: number) => void;
}> = ({ x, y, primary, secondary, onClose, onShapeChange, onDirChange, onSChange }) => {
    const { t } = useI18n();
    const menuRef = useRef<HTMLDivElement>(null);
    useMenuKeyboard(menuRef);
    // 底部提示展示用户实际配置的曲率修饰键（如 "Alt"）。
    const curvatureKb = useAppSelector((state) =>
        selectKeybindings(state, "modifier.fadeCurvatureDrag"),
    );
    const keysText = formatKeybindingList(curvatureKb, "");
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
            // 与其它右键菜单共用同一个表面与条目样式（`hs-menu*`，见 index.css）。
            // `--no-scroll`：内容是一张紧凑面板（每侧一行形状 + 一个曲率滑杆），
            // 不滚动 —— 与迁移前一致，也避免滚动容器干扰曲率预览的指针捕获。
            className="hs-menu hs-menu--no-scroll"
            style={{ left: x, top: y }}
            onContextMenu={(e) => e.preventDefault()}
            onPointerDown={(e) => e.stopPropagation()}
        >
            {secondary ? (
                // 交叉点：双列 —— 先前者淡出、后后者淡入。
                <>
                    <div className="hs-menu__label">{labelFor(primary)}</div>
                    <SideColumn
                        side={primary}
                        isOut={primary.isOut}
                        onShapeChange={onShapeChange}
                        onDirChange={onDirChange}
                        onSChange={onSChange}
                        t={(key) => t(key as MessageKey)}
                    />
                    <div className="hs-menu__separator" role="separator" />
                    <div className="hs-menu__label">{labelFor(secondary)}</div>
                    <SideColumn
                        side={secondary}
                        isOut={secondary.isOut}
                        onShapeChange={onShapeChange}
                        onDirChange={onDirChange}
                        onSChange={onSChange}
                        t={(key) => t(key as MessageKey)}
                    />
                </>
            ) : (
                <>
                    <div className="hs-menu__label">{labelFor(primary)}</div>
                    <SideColumn
                        side={primary}
                        isOut={primary.isOut}
                        onShapeChange={onShapeChange}
                        onDirChange={onDirChange}
                        onSChange={onSChange}
                        t={(key) => t(key as MessageKey)}
                    />
                </>
            )}
            {/* 形状切换重置曲率的语义提示（与 Clip 菜单一致的行为说明）。 */}
            <div className="hs-menu__hint">{curvatureHint}</div>
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
