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
import { clearCanvasPhysical, rasterize } from "../renderKernel/canvasRaster";
import { subscribeDevicePixelRatio } from "../../../hooks/useDevicePixelRatio";
import { coalescedEventsOf, pointerKindOf } from "../../../utils/penInput";
import type { ContactReadoutMode } from "../../../services/api/settings";
import { AppButton, useRepeatPress } from "../../../ui";
import {
    cycleEditorPoint,
    cycleEditorY,
    cycleRightDragTransform,
    paintCycleBinWeighted,
    paintCycleSegmentWeighted,
    smoothCycleTable,
    transformCycleTable,
    type CyclePaintPoint,
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
    /**
     * 压感画笔（把笔的压力换算成"这一点写多深"）。
     *
     * 与 `fineAdjustKb` 同因：由宿主解析后注入，叶子组件不自己读 store ——
     * 否则这个小组件会变成又一个 store 消费者，而它的单测也就不再需要 Provider。
     *
     * 省略时权重恒为 1，即**完全覆盖**（鼠标的既有行为逐位不变）。压感设备上
     * 轻按只把该格向目标值推一部分，可以来回涂叠 —— 与绘画软件的画笔同源。
     */
    pressurePaint?: {
        /** 手势开始：重置压感标定。 */
        begin(): void;
        /** 给定采样点的写入权重（1 = 完全覆盖）。 */
        weightFor(event: { pointerType?: string | null; pressure?: number }): number;
    };
    /**
     * 接触读数的显示时机（`penInput.contactReadout`）。
     *
     * 【为什么需要】手指落下时**恰好盖住**它正在改的那一格 —— 用户看不见自己在
     * 画什么。数位笔没有这个问题（笔尖细、有悬停），所以默认只对触摸开启。
     *
     * 由宿主解析后注入（与本组件其余设备相关能力同因：叶子组件不读 store）。
     */
    contactReadout?: ContactReadoutMode;
}

/** 画布高度（CSS 像素）。 */
const EDITOR_HEIGHT = 120;

/**
 * 右键手势的启动阈值（CSS 像素）。
 *
 * 位移不足阈值前不写草稿：右键在画布上还有一个常见用途是"点一下看看光标在哪"，
 * 若按下即写入，一次没有位移的右键单击也会产生一次等值重算与重绘。
 */
/** 右键整体变换的位移阈值（CSS 像素）：纯右键单击不该写入草稿。 */
const TRANSFORM_DRAG_THRESHOLD_PX = 3;

/** 键盘改值的粗步长（表值域是 `[-1,1]`，0.1 约等于 5% 的满幅）。 */
const KEY_VALUE_STEP = 0.1;
/** 按住 Shift 的精细步长。 */
const KEY_VALUE_FINE_STEP = 0.02;

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
    pressurePaint,
    contactReadout,
}: VibratoCycleEditorProps) {
    const canvasRef = useRef<HTMLCanvasElement | null>(null);
    const containerRef = useRef<HTMLDivElement | null>(null);
    const drawingRef = useRef(false);
    /**
     * 上一个落点。
     *
     * 【为什么带 `weight`】压感画笔的"写入权重"要沿线段插值（见
     * `paintCycleSegmentWeighted`），因此必须记住上一点的权重；只存 bin/value 会让
     * 一笔之内出现"一段深一段浅"的台阶。鼠标的权重恒为 1，行为与从前逐位一致。
     */
    const lastPointRef = useRef<CyclePaintPoint | null>(null);
    /** 权重来源；缺省恒为 1（完全覆盖），即无压感设备的既有行为。 */
    const weightFor = (event: { pointerType?: string | null; pressure?: number }): number =>
        pressurePaint ? pressurePaint.weightFor(event) : 1;

    /*
     * ── 键盘编辑 ──────────────────────────────────────────────────────
     *
     * 【为什么必须有】这块画布此前唯一的输入方式是"拖"（`role="img"`，没有键盘
     * 路径）。任何丢失拖拽能力的环境 —— 触屏 ergonomics 差到不实用、辅助设备、
     * 远程桌面传不住拖拽 —— 都因此完全无法编辑波形。
     *
     * 语义取"方向键 = 移动笔位 / 改值"这套最标准的做法（与 Radix 滑块的箭头行为
     * 同源），Escape 撤销整段键盘编辑。之所以不做"Enter 才写入"的两步式：方向键
     * 即时写入更流畅，也少一次需要学习的操作 —— 而 Escape 已经提供了回退。
     */
    const [keyboardBin, setKeyboardBin] = useState(0);
    const [keyboardActive, setKeyboardActive] = useState(false);
    /** 获得焦点那一刻的表：Escape 整体回退到它。 */
    const keyboardBaselineRef = useRef<number[] | null>(null);
    /** 键盘光标所在格（取模到合法范围）与那一格的当前值。 */
    const binCount = Math.max(1, table.length);
    const keyboardIndex = ((keyboardBin % binCount) + binCount) % binCount;
    const keyboardValue = table[keyboardIndex] ?? 0;

    /** 该设备的落笔是否要显示接触读数。 */
    const shouldShowContactReadout = (pointerType?: string | null): boolean => {
        if (contactReadout === "always") return true;
        if (contactReadout === "touchOnly") return pointerKindOf(pointerType) === "touch";
        return false;
    };

    const handleKeyDown = (event: React.KeyboardEvent<HTMLCanvasElement>) => {
        if (disabled) return;
        const bins = Math.max(1, table.length);
        let handled = true;
        switch (event.key) {
            case "ArrowLeft":
                setKeyboardBin((bin) => (bin - 1 + bins) % bins);
                break;
            case "ArrowRight":
                setKeyboardBin((bin) => (bin + 1) % bins);
                break;
            case "ArrowUp":
            case "ArrowDown": {
                const step = event.shiftKey ? KEY_VALUE_FINE_STEP : KEY_VALUE_STEP;
                const direction = event.key === "ArrowUp" ? 1 : -1;
                const current = table[((keyboardBin % bins) + bins) % bins] ?? 0;
                // 权重 1 = 直接覆盖（与鼠标落笔逐位一致）。
                onChange(paintCycleBinWeighted(table, keyboardBin, current + direction * step, 1));
                break;
            }
            case "Escape": {
                const baseline = keyboardBaselineRef.current;
                if (baseline) {
                    onChange(baseline);
                    // 回退后以当前状态为新基线，连按两次不会"回退到更早的版本"。
                    keyboardBaselineRef.current = baseline.slice();
                }
                break;
            }
            default:
                handled = false;
        }
        if (handled) {
            // 阻止冒泡：否则方向键会同时触发全局的播放头 seek。
            event.preventDefault();
            event.stopPropagation();
        }
    };
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
    /**
     * 落笔时的接触读数（格号 + 写入值）。
     *
     * 【为什么与上面的 `readout` 分开】那个是右键整体变换的读数（相位 / 幅度），
     * 这个是画笔的落点读数；两者互斥（同一时刻只有一种手势），但语义不同，
     * 混成一个类型会让两边都要做无意义的判别。
     */
    const [paintReadout, setPaintReadout] = useState<{ bin: number; value: number } | null>(null);
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
            const cssWidth = container.clientWidth;
            if (cssWidth <= 0) return;
            const height = EDITOR_HEIGHT;
            const dpr = readDevicePixelRatio();
            // 统一走 `rasterize` 契约：物理尺寸 = `round(css × dpr)`，CSS 尺寸回算为
            // `physical / dpr`（布局盒与 backing store 严格 1:1，否则合成器重采样
            // 会让整块编辑器发虚）。清屏必须按物理尺寸做，不能沿用
            // `clearRect(0,0,cssW,cssH)`（round 向上取整时底部会残留 0~0.5 物理行）。
            const target = rasterize(canvas, cssWidth, height, dpr);
            const ctx = canvas.getContext("2d");
            if (!ctx) return;
            ctx.setTransform(target.dpr, 0, 0, target.dpr, 0, 0);
            clearCanvasPhysical(ctx, target);

            // 绘制坐标系 = 回算后的 CSS 尺寸（与画布样式逐值相等）。
            const width = target.cssWidthPx;
            /** 1 物理像素横线的中心 y：落在设备像素的半格上，线体恰好覆盖一整行。 */
            const hairlineY = (cssY: number): number => (Math.round(cssY * dpr) + 0.5) / dpr;
            /** 1 物理像素竖线的中心 x（同理）。 */
            const hairlineX = (cssX: number): number => (Math.round(cssX * dpr) + 0.5) / dpr;
            /** 1 物理像素的线宽（CSS 单位）。 */
            const hairlineWidth = 1 / dpr;

            const accent = tokenColor("--qt-accent", "#6aa9ff");
            const muted = tokenColor("--qt-text-muted", "#8a8a8a");
            const divider = tokenColor("--qt-divider", "#3a3a3a");

            // 值 → y 的换算与命中测试共用 `cycleEditorY`（内缩量同源，见其注释）：
            // 各写一套就会让"看到的峰顶"与"能画到 1.0 的那一行"错开。
            const valueToY = (value: number) => cycleEditorY(value, height);

            // 网格：中线 + 四分线（横），四分位置（纵）。
            ctx.strokeStyle = divider;
            ctx.lineWidth = hairlineWidth;
            for (const frac of [0.25, 0.5, 0.75]) {
                const y = height * frac;
                ctx.globalAlpha = frac === 0.5 ? 0.9 : 0.4;
                ctx.beginPath();
                ctx.moveTo(0, hairlineY(y));
                ctx.lineTo(width, hairlineY(y));
                ctx.stroke();
            }
            ctx.globalAlpha = 0.4;
            for (const frac of [0.25, 0.5, 0.75]) {
                const x = width * frac;
                ctx.beginPath();
                ctx.moveTo(hairlineX(x), 0);
                ctx.lineTo(hairlineX(x), height);
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

            /*
             * 键盘笔位标记。
             *
             * 只有用键盘时才画：鼠标 / 笔用户不需要它，而常驻的竖线会与"首尾相接"
             * 的左右缘高亮混在一起。没有这个标记，键盘用户按方向键改的是哪一格
             * 完全不可见 —— 而"看不见改了什么"比"改不了"更糟。
             */
            if (keyboardActive && !disabled && n >= 1) {
                const bin = ((keyboardBin % n) + n) % n;
                const x = ((bin + 0.5) / n) * width;
                const value = table[bin] ?? 0;
                ctx.strokeStyle = accent;
                ctx.lineWidth = hairlineWidth;
                ctx.globalAlpha = 0.75;
                ctx.beginPath();
                ctx.moveTo(hairlineX(x), 0);
                ctx.lineTo(hairlineX(x), height);
                ctx.stroke();
                // 当前值上的实心点：一眼看出"这一格被改到哪儿了"。
                ctx.globalAlpha = 1;
                ctx.fillStyle = accent;
                ctx.beginPath();
                ctx.arc(x, valueToY(value), 3, 0, Math.PI * 2);
                ctx.fill();
            }
        };

        draw();
        const observer = new ResizeObserver(draw);
        observer.observe(container);
        // dpr 变化同样要重画：画布物理尺寸与半像素吸附都依赖 dpr，而 ResizeObserver
        // 观察的是 CSS 布局盒 —— 纯 dpr 变化（浏览器缩放 / 换显示器）不会触发它。
        const unsubscribeDpr = subscribeDevicePixelRatio(draw);
        return () => {
            observer.disconnect();
            unsubscribeDpr();
        };
    }, [table, disabled, keyboardActive, keyboardBin]);

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
        // 手势开始：重置压感标定（每段手势重新观察，不固化上一次的极端值）。
        pressurePaint?.begin();
        drawingRef.current = true;
        const weighted: CyclePaintPoint = {
            ...point,
            weight: weightFor(event.nativeEvent),
        };
        lastPointRef.current = weighted;
        if (shouldShowContactReadout(event.nativeEvent.pointerType)) {
            setPaintReadout({ bin: weighted.bin, value: weighted.value });
        }
        capturePointer(event.currentTarget, event.pointerId);
        onChange(paintCycleBinWeighted(table, weighted.bin, weighted.value, weighted.weight ?? 1));
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
        // 取采样点的入口统一走 `penInput.coalescedEventsOf`（它已处理"合成事件 /
        // 旧 WebView 没有 getCoalescedEvents"的回退）—— 这里此前内联了一份同样的
        // 判断，绕开了那个唯一出处，异常路径因此与别处不一致。
        const points = coalescedEventsOf(event.nativeEvent);
        let next = table;
        let cursor = last;
        for (const sample of points) {
            const point = pointAt(sample);
            if (!point) continue;
            /*
             * 逐采样点取权重：一次 pointermove 里合并了多个采样点，而压力在这些
             * 采样点之间是变化的 —— 只按最后一个采样点算，笔画会丢掉压力的变化。
             * 无压感设备返回 1，退化为原来的"逐格覆盖"。
             */
            next = paintCycleSegmentWeighted(next, cursor, {
                ...point,
                weight: weightFor(sample),
            });
            cursor = point;
        }
        lastPointRef.current = cursor;
        if (shouldShowContactReadout(event.nativeEvent.pointerType)) {
            setPaintReadout({ bin: cursor.bin, value: cursor.value });
        }
        onChange(next);
    };

    const endGesture = (event: React.PointerEvent<HTMLDivElement>) => {
        if (!gestureActiveRef.current || !belongsToGesture(event)) return;
        gestureActiveRef.current = false;
        gesturePointerRef.current = null;
        transformRef.current = null;
        drawingRef.current = false;
        lastPointRef.current = null;
        setPaintReadout(null);
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
            : paintReadout
              ? `${paintReadout.bin + 1}/${binCount} · ${formatPercent(paintReadout.value * 100)}`
              : null;

    return (
        <div data-testid="vibrato-cycle-editor">
            <div
                ref={containerRef}
                className={`w-full overflow-hidden rounded ${transforming ? "cursor-move" : "cursor-crosshair"}`}
                style={{
                    height: EDITOR_HEIGHT,
                    touchAction: disabled ? undefined : "none",
                    // 边框用**内阴影**而不是 `border`：`border` 在 `box-sizing:
                    // border-box` 下会吃掉 2px 内容宽，使画布的内容盒比绘制坐标系
                    // 窄 2px —— 整幅图被水平压缩、命中位置也随之偏移。内阴影不占
                    // 布局，画布内容盒因此与绘制坐标系严格一致。
                    boxShadow: "inset 0 0 0 1px var(--qt-border)",
                }}
                onPointerDown={handlePointerDown}
                onPointerMove={handlePointerMove}
                onPointerUp={endGesture}
                onPointerCancel={endGesture}
                onContextMenu={handleContextMenu}
            >
                <canvas
                    ref={canvasRef}
                    /*
                     * `role="slider"` 而不是 `"img"`：键盘光标停在一格上，而那一格的
                     * 幅度就是一个 `[-1, 1]` 的标量 —— 这正是 slider 的语义，屏幕
                     * 阅读器因此能朗读"第几格、当前值多少"（见 `aria-valuetext`）。
                     *
                     * 顺带的好处：`useKeybindings` 的 `ARROW_OWNING_SELECTOR` 收录了
                     * `[role="slider"]`，方向键因此**自动**豁免全局绑定 —— 不必再手写
                     * 一层 stopPropagation 去和播放头 seek 抢按键。
                     */
                    role="slider"
                    tabIndex={disabled ? -1 : 0}
                    aria-label={ariaLabel}
                    aria-valuemin={-1}
                    aria-valuemax={1}
                    aria-valuenow={keyboardValue}
                    aria-valuetext={`${keyboardIndex + 1}/${binCount}`}
                    aria-disabled={disabled || undefined}
                    onKeyDown={handleKeyDown}
                    onFocus={() => {
                        // 以"获得焦点那一刻"为基线：Escape 回到这里。
                        keyboardBaselineRef.current = table.slice();
                        setKeyboardActive(true);
                    }}
                    onBlur={() => {
                        keyboardBaselineRef.current = null;
                        setKeyboardActive(false);
                    }}
                    className="block w-full h-full"
                />
            </div>
            <Flex align="center" gap="2" mt="2" wrap="wrap">
                <AppButton
                    size="sm"
                    emphasis="soft"
                    disabled={disabled}
                    // 短按平滑一次，按住连续平滑（`useRepeatPress`）。
                    // `hs-touch-none`：按住重复靠"指针始终不离开元素"成立，触摸上
                    // 必须收回手势所有权，否则手指一漂就被浏览器当成滚动并
                    // pointercancel，重复中断。见 `useRepeatPress.ts` 的说明。
                    className="hs-touch-none"
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
