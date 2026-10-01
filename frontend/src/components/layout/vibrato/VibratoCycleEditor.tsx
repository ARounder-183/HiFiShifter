/**
 * 手绘单周期编辑器。
 *
 * 【为什么需要它】`table` 波形此前只能由「从选区提取」产生 —— 用户想手捏一个
 * 周期没有入口。这里补上：横轴 = 一个周期（64 格），纵轴 = −1..1，带中线与四分
 * 线网格；按住拖动即以画笔落点写入，松手完成一次笔画。
 *
 * 【两种手势，互补】左键 = 逐格画笔（改形状）；**右键拖拽 = 整体变换**
 * （水平 → 相位旋转，垂直 → 幅度缩放），不改变形状、只调整它摆放的位置与幅度。
 * 逐格画完整条曲线之后"形状对了但整体偏了半个周期"是常态，没有整体手段就只能
 * 靠平滑（会改形）或复位（丢掉整个手绘）来近似。见 `vibratoCycleEdit.ts` 的
 * `transformCycleTable` 及其上方注释（为什么不是纵向平移）。
 *
 * 【与渲染的关系】写进草稿的是 `cycle: { kind: "table", table }` —— 预览、试听、
 * 拖拽、应用全管线无差别支持（这是统一周期模型当初的承诺）。首尾相接由渲染端
 * 保证（`sampleCycle` 把表当循环），编辑器只把左右缘各画一格高亮作视觉提示。
 */

import { useEffect, useRef, useState } from "react";
import { Flex } from "@radix-ui/themes";

import { isModifierActive } from "../../../features/keybindings/keybindingsSlice";
import type { Keybinding } from "../../../features/keybindings/types";
import { advanceFineAxisDrag, createFineAxisDragState } from "../../../utils/fineAxisDrag";
import type { FineAxisDragState } from "../../../utils/fineAxisDrag";
import { readDevicePixelRatio } from "../../../utils/devicePixelLine";
import { AppButton, useRepeatPress } from "../../../ui";
import {
    cycleEditorPoint,
    cycleEditorY,
    cycleRightDragTransform,
    paintCycleBin,
    paintCycleSegment,
    smoothCycleTable,
    transformCycleTable,
} from "./vibratoCycleEdit";

export interface VibratoCycleEditorProps {
    table: number[];
    disabled?: boolean;
    onChange: (table: number[]) => void;
    /** 复位到进入手绘前的形状；省略则不显示该按钮。 */
    onReset?: () => void;
    /**
     * 复位到**当前选中的参数式形状**（按钮文案里带上形状名）。
     *
     * 与 `onReset` 的区别：那个回到"你进来之前的样子"（可能是一张提取/手绘出来的表），
     * 这个把表重新采样成下拉框里当前那个形状 —— 手绘画歪了想从头来，或者想把提取出来
     * 的波形换成规整的正弦，都靠它。
     */
    onResetToShape?: () => void;
    smoothLabel: string;
    resetLabel: string;
    resetShapeLabel?: string;
    ariaLabel: string;
    /**
     * 右键整体变换的读数标签（相位 / 幅度）。
     *
     * 【为什么需要读数】右键没有悬停反馈，"精细调整"修饰键又让灵敏度变得不可见；
     * 没有数字，用户无从判断"精细"到底有多细。只在手势期间显示。
     */
    readoutLabels?: { phase: string; scale: string };
    /**
     * 「精细调整」修饰键（默认 `Ctrl` / macOS `Command`），作用于右键整体变换。
     *
     * 由宿主解析（与预览画布同一套：宿主手里已经有 `paramFineAdjustKb`）——
     * 叶子组件不自己读 keybindings 切片，免得把这个小组件变成又一个 store 消费者。
     */
    fineAdjustKb?: Keybinding;
}

/** 画布高度（CSS 像素）。 */
const EDITOR_HEIGHT = 120;

/**
 * 右键手势的启动阈值（CSS 像素）。
 *
 * 位移不足阈值前不写草稿：右键在画布上还有一个常见用途是"点一下看看光标在哪"，
 * 若按下即写入，一次没有位移的右键单击也会产生一次等值重算与重绘。
 */
const TRANSFORM_DRAG_THRESHOLD_PX = 3;

/**
 * 读数里的百分比：留一位小数，整数则省掉 `.0`。
 *
 * 【为什么不能一律取整】"精细调整"生效时缩放可能从 `100.0%` 只走到 `100.4%`；
 * 一律取整会把它显示成两个一样的 `100%`，用户以为精细调整没生效 —— 而读数存在的
 * 首要理由正是让精细调整变得可见。整数又省掉小数，纯粹是为了不吵。
 */
function formatPercent(value: number): string {
    const rounded = Math.round(value * 10) / 10;
    return `${Number.isInteger(rounded) ? rounded.toFixed(0) : rounded.toFixed(1)}%`;
}

/**
 * 指针捕获只是增强（手势的仲裁本来就靠 `gestureActiveRef`，不依赖捕获）。
 * jsdom 与部分合成事件环境没有实现这几个方法，缺了不该让手势直接抛错。
 */
function capturePointer(element: Element, pointerId: number) {
    (element as Element & { setPointerCapture?: (id: number) => void }).setPointerCapture?.(
        pointerId,
    );
}

function releasePointer(element: Element, pointerId: number) {
    const target = element as Element & {
        hasPointerCapture?: (id: number) => boolean;
        releasePointerCapture?: (id: number) => void;
    };
    if (target.hasPointerCapture?.(pointerId)) target.releasePointerCapture?.(pointerId);
}

function tokenColor(name: string, fallback: string): string {
    if (typeof window === "undefined") return fallback;
    const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    return value || fallback;
}

export function VibratoCycleEditor({
    table,
    disabled = false,
    onChange,
    onReset,
    onResetToShape,
    smoothLabel,
    resetLabel,
    resetShapeLabel,
    ariaLabel,
    readoutLabels,
    fineAdjustKb,
}: VibratoCycleEditorProps) {
    const canvasRef = useRef<HTMLCanvasElement | null>(null);
    const containerRef = useRef<HTMLDivElement | null>(null);
    const drawingRef = useRef(false);
    const lastPointRef = useRef<{ bin: number; value: number } | null>(null);
    /**
     * 当前手势占用的指针。
     *
     * 【为什么需要它】左键画笔与右键整体变换共用同一块画布，必须互斥：右键拖到
     * 一半又按下左键，会把变换的起点快照冲掉，松手后草稿里是一张被画笔写过的表。
     * `active` 为真即"正忙"，新的 `pointerdown` 一律忽略。
     *
     * `pointerId` 单独放一个 ref 而不是用 `null` 兼任"空闲"标记：某些环境
     * （jsdom、部分合成事件）给不出 `pointerId`，把它存成 `undefined` 会让"非 `null`
     * 即忙碌"的判断永久为真 —— 一次手势之后画布就再也不响应了。用布尔量表达
     * "忙不忙"，`pointerId` 只作过滤，缺省时退化为不做过滤。
     */
    const gestureActiveRef = useRef(false);
    const gesturePointerRef = useRef<number | null>(null);
    /**
     * 右键整体变换的手势状态。
     *
     * `startTable` 是**按下那一刻**的表：整段手势都基于它重算（而不是对上一帧的
     * 结果继续变换）。旋转是线性插值，反复插值会让曲线逐渐变平、峰值衰减；
     * 缩放连乘也会让误差滚雪球。基于快照，"拖出去再拖回来"才能逐位复原。
     */
    const transformRef = useRef<{
        startTable: number[];
        anchorX: number;
        anchorY: number;
        /** 两轴各一份累计器：修饰键能在拖拽途中按下 / 松开，位移须按增量缩放。 */
        fineX: FineAxisDragState;
        fineY: FineAxisDragState;
    } | null>(null);
    /** 手势读数（相位百分比 / 幅度百分比）；仅在右键手势期间非空。 */
    const [readout, setReadout] = useState<{ phasePct: number; scalePct: number } | null>(null);
    /** 右键手势是否进行中 —— 只用来切换光标（右键没有悬停态，按下才谈得上反馈）。 */
    const [transforming, setTransforming] = useState(false);

    const fineActiveFor = (event: {
        ctrlKey: boolean;
        shiftKey: boolean;
        altKey: boolean;
        metaKey?: boolean;
    }) => (fineAdjustKb ? isModifierActive(fineAdjustKb, event) : false);

    /*
     * 最新的表与回调：长按重复时每一拍都要基于**上一拍的结果**继续平滑。
     * 若闭包捕获的是按下那一刻的 `table`，连按十次也只会把同一份表平滑十遍 ——
     * 看起来像"长按没反应"。写入放在 effect 里（不是渲染期），与 `useFrameCommitter`
     * 的处理一致。
     */
    const latestRef = useRef({ table, onChange });
    useEffect(() => {
        latestRef.current = { table, onChange };
    });

    /** 短按平滑一次；按住则连续平滑（与键盘自动重复同一套手感）。 */
    const smoothPress = useRepeatPress({
        disabled,
        onTrigger: () => {
            const { table: current, onChange: apply } = latestRef.current;
            apply(smoothCycleTable(current));
        },
    });

    useEffect(() => {
        const canvas = canvasRef.current;
        const container = containerRef.current;
        if (!canvas || !container) return;

        const draw = () => {
            const width = container.clientWidth;
            if (width <= 0) return;
            const height = EDITOR_HEIGHT;
            const dpr = readDevicePixelRatio();
            canvas.width = Math.round(width * dpr);
            canvas.height = Math.round(height * dpr);
            canvas.style.width = `${width}px`;
            canvas.style.height = `${height}px`;

            const ctx = canvas.getContext("2d");
            if (!ctx) return;
            ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
            ctx.clearRect(0, 0, width, height);

            const accent = tokenColor("--qt-accent", "#6aa9ff");
            const muted = tokenColor("--qt-text-muted", "#8a8a8a");
            const divider = tokenColor("--qt-divider", "#3a3a3a");

            // 值 → y 的换算与命中测试共用 `cycleEditorY`（内缩量同源，见其注释）：
            // 各写一套就会让"看到的峰顶"与"能画到 1.0 的那一行"错开。
            const valueToY = (value: number) => cycleEditorY(value, height);

            // 网格：中线 + 四分线（横），四分位置（纵）。
            ctx.strokeStyle = divider;
            ctx.lineWidth = 1;
            for (const frac of [0.25, 0.5, 0.75]) {
                const y = height * frac;
                ctx.globalAlpha = frac === 0.5 ? 0.9 : 0.4;
                ctx.beginPath();
                ctx.moveTo(0, y + 0.5);
                ctx.lineTo(width, y + 0.5);
                ctx.stroke();
            }
            ctx.globalAlpha = 0.4;
            for (const frac of [0.25, 0.5, 0.75]) {
                const x = width * frac;
                ctx.beginPath();
                ctx.moveTo(x + 0.5, 0);
                ctx.lineTo(x + 0.5, height);
                ctx.stroke();
            }
            ctx.globalAlpha = 1;

            // 首尾各一格高亮：提示"这里与对面相接"。
            if (table.length > 0) {
                const cellWidth = width / table.length;
                ctx.globalAlpha = 0.12;
                ctx.fillStyle = accent;
                ctx.fillRect(0, 0, Math.max(1, cellWidth), height);
                ctx.fillRect(width - Math.max(1, cellWidth), 0, Math.max(1, cellWidth), height);
                ctx.globalAlpha = 1;
            }

            // 波形：画 n+1 个点（末点 = 首点），让周期在视觉上闭合。
            const n = table.length;
            if (n >= 2) {
                ctx.strokeStyle = disabled ? muted : accent;
                ctx.lineWidth = 1.5;
                ctx.lineJoin = "round";
                ctx.beginPath();
                for (let i = 0; i <= n; i += 1) {
                    const value = table[i % n] ?? 0;
                    const x = (i / n) * width;
                    const y = valueToY(value);
                    if (i === 0) ctx.moveTo(x, y);
                    else ctx.lineTo(x, y);
                }
                ctx.stroke();
            }
        };

        draw();
        const observer = new ResizeObserver(draw);
        observer.observe(container);
        return () => observer.disconnect();
    }, [table, disabled]);

    const pointAt = (event: { clientX: number; clientY: number }) => {
        const container = containerRef.current;
        if (!container) return null;
        const rect = container.getBoundingClientRect();
        return cycleEditorPoint(
            event.clientX - rect.left,
            event.clientY - rect.top,
            rect.width,
            EDITOR_HEIGHT,
            table.length,
        );
    };

    /** 本事件是否属于当前手势（`pointerId` 不可得时一律放行）。 */
    const belongsToGesture = (event: { pointerId: number }) =>
        gesturePointerRef.current === null || event.pointerId === gesturePointerRef.current;

    const beginGesture = (pointerId: number) => {
        gestureActiveRef.current = true;
        gesturePointerRef.current = typeof pointerId === "number" ? pointerId : null;
    };

    const handlePointerDown = (event: React.PointerEvent<HTMLDivElement>) => {
        if (disabled || gestureActiveRef.current) return;

        // 右键 = 整体变换：登记起点快照与锚点，整段手势基于快照重算（见 transformRef）。
        if (event.button === 2) {
            if (!containerRef.current) return;
            beginGesture(event.pointerId);
            const fine = fineActiveFor(event);
            transformRef.current = {
                startTable: table.slice(),
                anchorX: event.clientX,
                anchorY: event.clientY,
                fineX: createFineAxisDragState(0, fine),
                fineY: createFineAxisDragState(0, fine),
            };
            setTransforming(true);
            capturePointer(event.currentTarget, event.pointerId);
            return;
        }
        if (event.button !== 0) return;

        const point = pointAt(event);
        if (!point) return;
        beginGesture(event.pointerId);
        drawingRef.current = true;
        lastPointRef.current = point;
        capturePointer(event.currentTarget, event.pointerId);
        onChange(paintCycleBin(table, point.bin, point.value));
    };

    /** 右键整体变换的一帧：位移 → 变换量 → 对快照重算 → 上报。 */
    const applyTransformMove = (
        event: React.PointerEvent<HTMLDivElement>,
        gesture: NonNullable<typeof transformRef.current>,
    ) => {
        const rawDx = event.clientX - gesture.anchorX;
        const rawDy = event.clientY - gesture.anchorY;
        // 阈值内不动草稿：纯右键单击（或手抖）不该产生一次无意义的写入与重绘。
        if (
            Math.abs(rawDx) < TRANSFORM_DRAG_THRESHOLD_PX &&
            Math.abs(rawDy) < TRANSFORM_DRAG_THRESHOLD_PX
        ) {
            return;
        }
        const fine = fineActiveFor(event);
        const dx = advanceFineAxisDrag(gesture.fineX, rawDx, fine);
        const dy = advanceFineAxisDrag(gesture.fineY, rawDy, fine);
        const bins = Math.max(1, gesture.startTable.length);
        const spec = cycleRightDragTransform(
            dx,
            dy,
            containerRef.current?.clientWidth ?? 0,
            EDITOR_HEIGHT,
            bins,
        );
        onChange(transformCycleTable(gesture.startTable, spec.rotateBins, spec.scale));
        // 旋转是周期的：读"当前落在哪儿"（一个周期内的百分比），而不是越拖越大的累计量。
        const turns = spec.rotateBins / bins;
        setReadout({
            phasePct: (((turns % 1) + 1) % 1) * 100,
            scalePct: spec.scale * 100,
        });
    };

    const handlePointerMove = (event: React.PointerEvent<HTMLDivElement>) => {
        if (disabled || !gestureActiveRef.current || !belongsToGesture(event)) return;

        const gesture = transformRef.current;
        if (gesture) {
            applyTransformMove(event, gesture);
            return;
        }
        if (!drawingRef.current) return;
        const last = lastPointRef.current;
        if (!last) return;
        // 用合并事件补齐快笔：一次 pointermove 里可能有多个采样点。
        const native = event.nativeEvent;
        const points =
            typeof native.getCoalescedEvents === "function"
                ? native.getCoalescedEvents()
                : [native];
        let next = table;
        let cursor = last;
        for (const sample of points.length > 0 ? points : [native]) {
            const point = pointAt(sample);
            if (!point) continue;
            next = paintCycleSegment(next, cursor, point);
            cursor = point;
        }
        lastPointRef.current = cursor;
        onChange(next);
    };

    const endGesture = (event: React.PointerEvent<HTMLDivElement>) => {
        if (!gestureActiveRef.current || !belongsToGesture(event)) return;
        gestureActiveRef.current = false;
        gesturePointerRef.current = null;
        transformRef.current = null;
        drawingRef.current = false;
        lastPointRef.current = null;
        setTransforming(false);
        setReadout(null);
        releasePointer(event.currentTarget, event.pointerId);
    };

    /**
     * 吃掉原生右键菜单。
     *
     * 【为什么无条件】右键在这块画布上专用于整体变换，本来就不挂菜单。只在"手势已
     * 成立"时才 `preventDefault` 是不够的：各平台触发时机不同（Windows 在 `pointerup`，
     * X11/macOS 在 `pointerdown`），一次**没有位移的纯右键单击**仍会把系统菜单弹出来 ——
     * 而那一下在用户看来正是"我想整体拖一下"的起手。
     */
    const handleContextMenu = (event: React.MouseEvent<HTMLDivElement>) => {
        event.preventDefault();
    };

    const readoutText =
        readout && readoutLabels
            ? `${readoutLabels.phase} ${formatPercent(readout.phasePct)} · ${readoutLabels.scale} ${formatPercent(readout.scalePct)}`
            : null;

    return (
        <div data-testid="vibrato-cycle-editor">
            <div
                ref={containerRef}
                className={`w-full ${transforming ? "cursor-move" : "cursor-crosshair"}`}
                style={{ height: EDITOR_HEIGHT, touchAction: disabled ? undefined : "none" }}
                onPointerDown={handlePointerDown}
                onPointerMove={handlePointerMove}
                onPointerUp={endGesture}
                onPointerCancel={endGesture}
                onContextMenu={handleContextMenu}
            >
                <canvas
                    ref={canvasRef}
                    role="img"
                    aria-label={ariaLabel}
                    className="block w-full rounded border border-qt-border"
                    style={{ height: EDITOR_HEIGHT }}
                />
            </div>
            <Flex align="center" gap="2" mt="2" wrap="wrap">
                <AppButton
                    size="sm"
                    emphasis="soft"
                    disabled={disabled}
                    // 短按平滑一次，按住连续平滑（`useRepeatPress`）。
                    {...smoothPress}
                >
                    {smoothLabel}
                </AppButton>
                {onReset ? (
                    <AppButton size="sm" emphasis="soft" disabled={disabled} onClick={onReset}>
                        {resetLabel}
                    </AppButton>
                ) : null}
                {onResetToShape && resetShapeLabel ? (
                    <AppButton
                        size="sm"
                        emphasis="soft"
                        disabled={disabled}
                        onClick={onResetToShape}
                    >
                        {resetShapeLabel}
                    </AppButton>
                ) : null}
                {readoutText ? (
                    <span
                        data-testid="vibrato-cycle-readout"
                        className="ml-auto whitespace-nowrap text-qt-xs text-qt-text-muted"
                    >
                        {readoutText}
                    </span>
                ) : null}
            </Flex>
        </div>
    );
}
