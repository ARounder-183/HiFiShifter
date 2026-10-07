/**
 * 颤音预设的波形预览画布。
 *
 * 【为什么必须有】没有它，用户在预设编辑器里是在**盲调**：深度、速率、渐入、
 * 不规则度这几个参数两两耦合，只看数字无法预判听感。这里把
 * `buildVibratoCurve` 的真实输出画出来，参数一改立刻重画。
 *
 * 纵轴按预设**自身**的幅度定标（见 `previewScaleCents`）：按参数值域定标的话，
 * 30 cents 的颤音会是一条直线。真实幅度由旁边的读数表达。
 *
 * 【可编辑】传入 `handles` 与手势回调后，画布从"只读渲染"升级为"所见即所编"：
 * 左右手柄拖动改渐入 / 渐出，主体拖动改相位与深度。手势只上报**区域与像素位移**，
 * 具体的字段换算由 `vibratoPreviewGestures.ts` 的纯函数完成 —— 画布不持有草稿。
 */

import { useEffect, useRef } from "react";
import { readDevicePixelRatio } from "../../../utils/devicePixelLine";
import { clearCanvasPhysical, rasterize } from "../renderKernel/canvasRaster";
import { subscribeDevicePixelRatio } from "../../../hooks/useDevicePixelRatio";
import { coalescedEventsOf } from "../../../utils/penInput";
import { profileFor, scaledHitRadius } from "../../../utils/inputProfile";
import {
    accumulatePinchSteps,
    createPinchStepState,
    subscribePinch,
} from "../../../utils/pinchGesture";
import { previewScaleCents, type VibratoPreviewSamples } from "./vibratoDialogLogic";
import { finiteRuns, strokeFinitePolyline } from "./vibratoPreviewDraw";
import {
    cursorForZone,
    HANDLE_HIT_PX,
    hitTestPreviewZone,
    type PreviewHandleLayout,
    type PreviewZone,
} from "./vibratoPreviewGestures";

/** 画布几何：绘制与手势换算共用同一套标尺。 */
export interface VibratoPreviewGeometry {
    /** CSS 像素宽度。 */
    width: number;
    /** CSS 像素高度。 */
    height: number;
    /** 纵向每像素对应的 cents（与绘制同一套换算）。 */
    centsPerPx: number;
}

/** 手势开始时上报的几何 + 按下瞬间的指针状态。 */
export interface VibratoPreviewGestureInfo extends VibratoPreviewGeometry {
    /**
     * 按下瞬间的指针状态。
     *
     * 【为什么起点就要给】「精细调整」按增量缩放位移（见 `utils/fineAxisDrag.ts`），
     * 起手时修饰键是否已按下决定了首帧走哪个比例 —— 起手就按住时不该按"刚按下"
     * 的过渡比例处理。
     */
    modifiers: VibratoPreviewInputState;
}

/**
 * 一次指针事件的完整状态快照。
 *
 * 【为什么不止是修饰键】设备适配后，手感还取决于**是哪台设备在按**：
 * 数位笔的压感决定本帧走多快（`utils/pressureCurve.ts`），触摸的前 12px 要走
 * 精细斜坡（`utils/inputProfile.ts`）。这两者都只能从指针事件本身读到，
 * 因此随修饰键一起上报，而不是让画布自己去猜。
 *
 * 全部字段都可缺省：合成事件（测试桩 / 键盘重放的 pointermove）没有它们，
 * 缺省即"按鼠标处理"，与 `penInput.ts` 的宽松回退同口径。
 */
export interface VibratoPreviewInputState {
    ctrlKey: boolean;
    shiftKey: boolean;
    altKey: boolean;
    metaKey: boolean;
    /** `PointerEvent.pointerType`；缺省表示合成事件。 */
    pointerType?: string | null;
    /** `PointerEvent.pressure`；鼠标 / 触摸恒为 0.5。 */
    pressure?: number;
    /** `PointerEvent.tiltX`（度）；不报倾斜的设备恒为 0。 */
    tiltX?: number;
}

/** 从指针事件读出完整状态（修饰键 + 设备通道）。 */
function readInputState(event: {
    ctrlKey: boolean;
    shiftKey: boolean;
    altKey: boolean;
    metaKey: boolean;
    pointerType?: string;
    pressure?: number;
    tiltX?: number;
}): VibratoPreviewInputState {
    return {
        ctrlKey: event.ctrlKey,
        shiftKey: event.shiftKey,
        altKey: event.altKey,
        metaKey: event.metaKey,
        pointerType: event.pointerType,
        pressure: event.pressure,
        tiltX: event.tiltX,
    };
}

export interface VibratoPreviewCanvasProps {
    samples: VibratoPreviewSamples;
    /** CSS 像素高度。 */
    height?: number;
    ariaLabel?: string;
    /**
     * 纵轴半幅（cents）。由调用方**一次性拟合**并保持稳定时，波形高度就等于深度，
     * 用户能直接判断幅度大小；省略时按峰值自适应（只读预览够用，但深度一变整幅
     * 就被重新缩放，看不出大小）。
     */
    halfCents?: number;
    /** 手柄的归一化横向位置（0..1）。提供时画出手柄并接受手势。 */
    handles?: PreviewHandleLayout;
    /** 手势开始。 */
    onGestureStart?: (zone: PreviewZone, info: VibratoPreviewGestureInfo) => void;
    /** 手势移动：自起点累计的像素位移，以及当前的指针状态。 */
    onGestureMove?: (deltaX: number, deltaY: number, modifiers: VibratoPreviewInputState) => void;
    /** 手势结束。 */
    onGestureEnd?: () => void;
    /**
     * 触控板捏合调深度（`deltaCents` 为正 = 加深）。
     *
     * 【为什么是深度而不是"缩放纵轴"】本画布在编辑时就是**另一组滑杆**（见文件头），
     * 而纵轴是自动拟合的显示量 —— 缩放它不改变任何参数，只会让波形变大变小。深度
     * 才是用户真正想调的那个量。
     *
     * 【灵敏度与拖拽同源】一格捏合按"相当于纵向拖 20px"折算，再乘本画布的
     * `centsPerPx` —— 于是"捏一格"与"往上拖 20px"改出的深度完全一致，两套输入
     * 共用同一条换算，不会出现"捏合比拖拽灵敏得多"的分裂手感。
     */
    onPinchDepth?: (deltaCents: number) => void;
}

/** 读取语义色令牌；缺失（测试 / 非浏览器环境）时回退到中性色。 */
function tokenColor(name: string, fallback: string): string {
    if (typeof window === "undefined") return fallback;
    const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    return value || fallback;
}

/** 手柄方块边长（CSS 像素）。 */
const HANDLE_SIZE = 7;

/**
 * 一格捏合折算成多少像素的**纵向拖拽**。
 *
 * 【为什么是"拖拽像素"而不是直接写一个 cents 值】本画布的纵轴是按峰值自动拟合的
 * （`centsPerPx` 随之变化），写死一个 cents 步长会在不同深度下给出截然不同的手感。
 * 折算成像素后乘本画布自己的标尺，"捏一格"与"往上拖 20px"就永远改出同样的深度。
 */
const PINCH_DRAG_PX_PER_NOTCH = 20;

/** 轴标签的紧凑写法：整数不带小数点，非整数最多一位。 */
function formatAxisCents(value: number): string {
    return String(Math.round(value * 10) / 10);
}

export function VibratoPreviewCanvas({
    samples,
    height = 120,
    ariaLabel,
    halfCents: explicitHalfCents,
    handles,
    onGestureStart,
    onGestureMove,
    onGestureEnd,
    onPinchDepth,
}: VibratoPreviewCanvasProps) {
    const canvasRef = useRef<HTMLCanvasElement | null>(null);
    const containerRef = useRef<HTMLDivElement | null>(null);
    /** 最近一次绘制的几何：手势换算要用与绘制**同一套**标尺。 */
    const geometryRef = useRef<VibratoPreviewGeometry>({ width: 0, height, centsPerPx: 1 });
    /** 手势起点（未按下时为 null）。 */
    const gestureRef = useRef<{ zone: PreviewZone; x: number; y: number } | null>(null);
    /**
     * 待提交的一帧位移（合并事件批处理用）。
     *
     * 【为什么需要】笔的采样率常见 133–266Hz（高端 500Hz+），远高于渲染帧率。
     * 若每个 pointermove 都走一次 `onGestureMove` → `patch` → `setDraft` + 全画布
     * 重绘，一秒内会做数百次 React 渲染 —— 而其中绝大多数中间帧在下一帧就被覆盖。
     * 只保留"本帧最后一个采样点"，与钢琴卷帘线工具的既有做法同源
     * （`usePianoRollInteractions` 的 `pendingLineEvent`）。
     *
     * 【为什么可以只留最后一个】预览手势是**基于起点的端点映射**（`applyPreviewGesture`
     * 从按下快照重算），中间帧不是轨迹的一部分，丢掉不损失信息。
     */
    const pendingMoveRef = useRef<{
        x: number;
        y: number;
        modifiers: VibratoPreviewInputState;
    } | null>(null);
    /** 已排程的 rAF 句柄；`null` 表示当前没有待提交帧。 */
    const rafRef = useRef<number | null>(null);
    /** 最近一次设置的悬停光标，避免每个 move 都写一次 style（笔悬停会高频触发）。 */
    const hoverCursorRef = useRef<string>("");
    /**
     * 最新的 `onGestureMove`。
     *
     * 【为什么用 ref 而不是直接闭包】rAF 回调在**下一帧**执行，那时本次渲染的闭包
     * 可能已经过期；而调用方每次渲染都会重建内联箭头函数。经 ref 转发保证提交的
     * 永远是当前那一份，与 `useFrameCommit` 转发 commit 的做法同源。
     *
     * 写入发生在 effect 里（不是渲染期）：React Compiler 的引用规则会拒绝
     * "渲染期把读 ref 的闭包传给函数"这一类写法，`useFrameCommit.ts:44-49` 记录了
     * 同一条约束。
     */
    const onGestureMoveRef = useRef(onGestureMove);
    useEffect(() => {
        onGestureMoveRef.current = onGestureMove;
    });
    const interactive = Boolean(handles && (onGestureStart || onGestureMove));

    // 卸载时丢弃挂起帧：此刻提交会打到已卸载的父组件上。
    useEffect(
        () => () => {
            if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
            rafRef.current = null;
            pendingMoveRef.current = null;
        },
        [],
    );

    /**
     * 捏合调深度：一格按"纵向拖 `PINCH_DRAG_PX_PER_NOTCH` 像素"折算。
     *
     * 【为什么订阅总线而不是自己挂 wheel 监听】全局"禁用浏览器缩放"的守卫只有一处
     * （`App.tsx` 的 capture 监听），它已经拿到了全部捏合事件；各表面再各挂一个
     * capture 监听会重复，触发顺序也不确定。总线让 App 做唯一的识别点。
     */
    const onPinchDepthRef = useRef(onPinchDepth);
    useEffect(() => {
        onPinchDepthRef.current = onPinchDepth;
    });
    /** 捏合的整步累积器（跨渲染保留，否则每次都从零开始凑不满一格）。 */
    const pinchStepRef = useRef(createPinchStepState());
    useEffect(() => {
        if (!interactive) return;
        return subscribePinch((event) => {
            const handler = onPinchDepthRef.current;
            const container = containerRef.current;
            if (!handler || !container) return;
            // 同一窗口可能有多个可捏合表面：只处理落在本画布上的。
            const rect = container.getBoundingClientRect();
            if (
                event.clientX < rect.left ||
                event.clientX > rect.right ||
                event.clientY < rect.top ||
                event.clientY > rect.bottom
            ) {
                return;
            }
            const steps = accumulatePinchSteps(
                pinchStepRef.current,
                event.delta,
                performance.now(),
            );
            if (steps === 0) return;
            const centsPerPx = geometryRef.current.centsPerPx;
            if (!Number.isFinite(centsPerPx) || centsPerPx <= 0) return;
            handler(steps * PINCH_DRAG_PX_PER_NOTCH * centsPerPx);
        });
    }, [interactive]);

    useEffect(() => {
        const canvas = canvasRef.current;
        const container = containerRef.current;
        if (!canvas || !container) return;

        const draw = () => {
            const cssWidth = container.clientWidth;
            if (cssWidth <= 0) return;
            const dpr = readDevicePixelRatio();
            // 统一走 `rasterize` 契约：物理尺寸 = `round(css × dpr)`，CSS 尺寸回算为
            // `physical / dpr`。回算保证布局盒与 backing store 严格 1:1 —— 否则浏览器
            // 会把 physical 个像素铺到 `css × dpr` 个像素的盒上做非整数倍重采样，整块
            // 预览发虚（且随宽度奇偶变化）。清屏同理必须按物理尺寸做，不能沿用
            // `clearRect(0,0,cssW,cssH)`（round 向上取整时底部会残留 0~0.5 物理行）。
            const target = rasterize(canvas, cssWidth, height, dpr);
            const ctx = canvas.getContext("2d");
            if (!ctx) return;
            ctx.setTransform(target.dpr, 0, 0, target.dpr, 0, 0);
            clearCanvasPhysical(ctx, target);

            // 绘制坐标系 = 回算后的 CSS 尺寸（与画布样式逐值相等）。
            const width = target.cssWidthPx;
            const drawHeight = target.cssHeightPx;
            /**
             * 1 物理像素横线的中心 y：落在设备像素的半格上，线体恰好覆盖一整行。
             *
             * 旧的 `y + 0.5` 只在 dpr=1 下成立；分数 dpr 下 0.5 CSS px 不是半格，
             * 线会跨在两行设备像素之间按位置忽粗忽细。
             */
            const hairlineY = (cssY: number): number => (Math.round(cssY * dpr) + 0.5) / dpr;
            /** 1 物理像素的线宽（CSS 单位）。 */
            const hairlineWidth = 1 / dpr;

            const accent = tokenColor("--qt-accent", "#6aa9ff");
            const muted = tokenColor("--qt-text-muted", "#8a8a8a");
            const divider = tokenColor("--qt-divider", "#3a3a3a");
            const panel = tokenColor("--qt-panel", "#1e1e1e");

            const midY = drawHeight / 2;
            const verticalReach = drawHeight / 2 - 6;
            // 显式纵轴（半幅 cents）优先：它由调用方**一次性拟合**，编辑期间保持不变，
            // 于是波形高度直接等于深度，用户能直观判断大小。省略时回落到按峰值自适应
            // （缩略图 / 无手势的只读预览用得上）。
            const halfCents =
                explicitHalfCents != null && explicitHalfCents > 0
                    ? explicitHalfCents
                    : previewScaleCents(samples.peakCents);
            const toY = (cents: number) => midY - (cents / halfCents) * verticalReach;
            geometryRef.current = {
                width,
                height: drawHeight,
                centsPerPx: halfCents / Math.max(1, verticalReach),
            };

            const count = Math.max(2, samples.wave.length);
            const toX = (index: number) => (index / (count - 1)) * width;

            /*
             * 背景：原参数线（弱化虚线）。
             *
             * 【与前景同一套标尺】两条线都相对**同一个中心**取值，因此共用上面那套
             * cents 标尺：按「适应」换比例尺时两条一起缩放，而不是只有前景动。这正是
             * "新参数线相对原参数线差多少"能读出来的前提。
             */
            const contour = samples.contour;
            if (contour && contour.length >= 2) {
                ctx.save();
                ctx.globalAlpha = 0.45;
                ctx.strokeStyle = muted;
                ctx.lineWidth = hairlineWidth;
                ctx.setLineDash([4, 3]);
                strokeFinitePolyline(ctx, contour, toX, toY);
                ctx.restore();
            }

            // 刻度网格 + 读数：没有它，"波形占画布多少"仍然只是相对量；有了它，
            // 用户能直接把波峰高度读成 cents。上下边缘各标一次，中间画到 1/4 的细线。
            ctx.strokeStyle = divider;
            ctx.lineWidth = hairlineWidth;
            ctx.font = "10px sans-serif";
            ctx.textBaseline = "middle";
            for (const fraction of [1, 0.5]) {
                for (const sign of [1, -1]) {
                    const cents = sign * halfCents * fraction;
                    const y = toY(cents);
                    ctx.globalAlpha = fraction === 1 ? 0.55 : 0.28;
                    ctx.beginPath();
                    ctx.moveTo(0, hairlineY(y));
                    ctx.lineTo(width, hairlineY(y));
                    ctx.stroke();
                    if (fraction === 1) {
                        ctx.globalAlpha = 0.75;
                        ctx.fillStyle = muted;
                        // 标签贴在线的内侧，避免被裁掉；单位由下方读数承担。
                        const label = `${sign > 0 ? "+" : "−"}${formatAxisCents(halfCents)}`;
                        ctx.fillText(label, 3, y + (sign > 0 ? 7 : -7));
                    }
                }
            }
            ctx.globalAlpha = 1;
            ctx.fillStyle = muted;

            // 零基线：波形围绕它摆动，视觉上是"音高中心"。
            ctx.strokeStyle = divider;
            ctx.lineWidth = hairlineWidth;
            ctx.beginPath();
            ctx.moveTo(0, hairlineY(midY));
            ctx.lineTo(width, hairlineY(midY));
            ctx.stroke();

            /*
             * 包络的两个取样器（带子的填充与渐入 / 渐出手柄共用）。
             *
             * 带子围绕**它所在的那条基线**摆动：省略中心线时是 0（管理器预览的用法），
             * 套用页签给的是那条基线本身（通常不在 0 上）。
             */
            const envelopeAt = (i: number) =>
                samples.envelope.length
                    ? samples.envelope[Math.min(i, samples.envelope.length - 1)]
                    : 0;
            const centerAt = (i: number) => {
                const center = samples.envelopeCenter;
                if (!center || center.length === 0) return 0;
                const value = center[Math.min(i, center.length - 1)];
                return Number.isFinite(value) ? value : 0;
            };

            // 包络带：±envelope 的填充区间，渐入 / 渐强 / 渐出入画就靠它。
            // 按连续段分别填充 —— 未检测帧处留空，而不是横着连成一条带。
            if (samples.envelope.length > 0) {
                ctx.globalAlpha = 0.16;
                ctx.fillStyle = muted;
                for (const [from, to] of finiteRuns(samples.envelope, count)) {
                    ctx.beginPath();
                    for (let i = from; i <= to; i += 1) {
                        ctx.lineTo(toX(i), toY(centerAt(i) + envelopeAt(i)));
                    }
                    for (let i = to; i >= from; i -= 1) {
                        ctx.lineTo(toX(i), toY(centerAt(i) - envelopeAt(i)));
                    }
                    ctx.closePath();
                    ctx.fill();
                }
                ctx.globalAlpha = 1;
            }

            // 波形本体（相对它所围绕的那条曲线的偏移量）。
            ctx.strokeStyle = accent;
            ctx.lineWidth = 1.5;
            ctx.lineJoin = "round";
            strokeFinitePolyline(ctx, samples.wave, toX, toY);

            // 渐入 / 渐出手柄：小方块落在包络斜坡的起止处，语言与时间轴 clip 的
            // fade 手柄一致 —— 用户已经学会在那里拖。
            if (handles) {
                const drawHandle = (frac: number) => {
                    const index = Math.round(frac * (count - 1));
                    const envValue = samples.envelope.length
                        ? samples.envelope[Math.min(index, samples.envelope.length - 1)]
                        : 0;
                    const x = toX(index);
                    // 手柄落在**包络带的上缘**：带子围绕 `envelopeCenter` 摆动时，
                    // 只按 envelope 取值会把它画到带子外面去（套用页签的带子中心是
                    // 基线，通常不在 0 上）。
                    const y = toY(centerAt(index) + envValue);
                    const half = HANDLE_SIZE / 2;
                    ctx.fillStyle = panel;
                    ctx.strokeStyle = accent;
                    ctx.lineWidth = 1.5;
                    ctx.beginPath();
                    ctx.rect(x - half, y - half, HANDLE_SIZE, HANDLE_SIZE);
                    ctx.fill();
                    ctx.stroke();
                };
                drawHandle(handles.attackFrac);
                drawHandle(handles.releaseFrac);
            }
        };

        draw();

        // 宽度随停靠面板 / 对话框变化，需要跟着重画（画布不会自己缩放）。
        const observer = new ResizeObserver(draw);
        observer.observe(container);
        // dpr 变化同样要重画：画布物理尺寸与半像素吸附都依赖 dpr，而 ResizeObserver
        // 观察的是 CSS 布局盒 —— 纯 dpr 变化（浏览器缩放 / 换显示器）不会触发它。
        const unsubscribeDpr = subscribeDevicePixelRatio(draw);
        return () => {
            observer.disconnect();
            unsubscribeDpr();
        };
    }, [samples, height, handles, explicitHalfCents]);

    const localPoint = (event: React.PointerEvent<HTMLDivElement>) => {
        const rect = event.currentTarget.getBoundingClientRect();
        return { x: event.clientX - rect.left, y: event.clientY - rect.top };
    };

    /** 手柄的命中半径：按设备剖面缩放（手指 9mm 的接触面抓不住 9px 的方块）。 */
    const hitRadiusFor = (event: { pointerType?: string | null }) =>
        scaledHitRadius(HANDLE_HIT_PX, profileFor(event));

    /**
     * 把待提交的一帧位移落到手势回调。
     *
     * 【为什么单独抽出来】它同时被 rAF 与 `endGesture` 调用 —— 后者必须在松手时
     * **同步**补一次，否则最后一帧（也是最终值所在的那一帧）会被取消掉，
     * 表现为"松手后停在上一帧的位置"。
     */
    const flushPendingMove = () => {
        if (rafRef.current !== null) {
            cancelAnimationFrame(rafRef.current);
            rafRef.current = null;
        }
        const pending = pendingMoveRef.current;
        pendingMoveRef.current = null;
        const gesture = gestureRef.current;
        if (!pending || !gesture) return;
        onGestureMoveRef.current?.(pending.x - gesture.x, pending.y - gesture.y, pending.modifiers);
    };

    const handlePointerDown = (event: React.PointerEvent<HTMLDivElement>) => {
        if (!interactive || !handles || event.button !== 0) return;
        const { x, y } = localPoint(event);
        const zone = hitTestPreviewZone(
            x,
            geometryRef.current.width,
            handles,
            hitRadiusFor(event.nativeEvent),
        );
        gestureRef.current = { zone, x, y };
        pendingMoveRef.current = null;
        event.currentTarget.setPointerCapture(event.pointerId);
        onGestureStart?.(zone, { ...geometryRef.current, modifiers: readInputState(event) });
    };

    const handlePointerMove = (event: React.PointerEvent<HTMLDivElement>) => {
        if (!interactive || !handles) return;
        const gesture = gestureRef.current;
        if (!gesture) {
            // 未按下时只更新光标，给出"这里能拖"的提示。
            // 只在区域真的变了时才写 style：笔悬停会以 100Hz+ 上报 pointermove，
            // 每次都写一遍是无谓的布局失效。
            const next = cursorForZone(
                hitTestPreviewZone(
                    event.clientX - event.currentTarget.getBoundingClientRect().left,
                    geometryRef.current.width,
                    handles,
                    hitRadiusFor(event.nativeEvent),
                ),
            );
            if (next !== hoverCursorRef.current) {
                hoverCursorRef.current = next;
                event.currentTarget.style.cursor = next;
            }
            return;
        }
        // 合并事件只留最后一个采样点（端点映射，中间帧无信息量），并推迟到
        // 下一帧统一提交 —— 笔的高采样率不该变成同等数量的 React 渲染。
        const samples = coalescedEventsOf(event.nativeEvent);
        const latest = samples[samples.length - 1];
        const rect = event.currentTarget.getBoundingClientRect();
        pendingMoveRef.current = {
            x: latest.clientX - rect.left,
            y: latest.clientY - rect.top,
            modifiers: readInputState(event),
        };
        if (rafRef.current === null) {
            rafRef.current = requestAnimationFrame(() => {
                rafRef.current = null;
                const pending = pendingMoveRef.current;
                pendingMoveRef.current = null;
                const active = gestureRef.current;
                if (!pending || !active) return;
                onGestureMoveRef.current?.(
                    pending.x - active.x,
                    pending.y - active.y,
                    pending.modifiers,
                );
            });
        }
    };

    const endGesture = (event: React.PointerEvent<HTMLDivElement>) => {
        if (!gestureRef.current) return;
        // 先把挂起的那一帧补上，再清手势 —— 顺序反了会丢掉松手前的最后一个采样点。
        flushPendingMove();
        gestureRef.current = null;
        if (event.currentTarget.hasPointerCapture(event.pointerId)) {
            event.currentTarget.releasePointerCapture(event.pointerId);
        }
        onGestureEnd?.();
    };

    return (
        <div
            ref={containerRef}
            className="w-full"
            style={{ height, touchAction: interactive ? "none" : undefined }}
            // 纵轴半幅（cents）：暴露出来便于测试断言"标尺是拟合档位而不是跟着当前值走"。
            data-axis-cents={explicitHalfCents ?? undefined}
            onPointerDown={handlePointerDown}
            onPointerMove={handlePointerMove}
            onPointerUp={endGesture}
            onPointerCancel={endGesture}
            onPointerLeave={(event) => {
                if (!gestureRef.current) {
                    hoverCursorRef.current = "";
                    event.currentTarget.style.cursor = "";
                }
            }}
            data-testid={interactive ? "vibrato-preview-interactive" : undefined}
        >
            <canvas
                ref={canvasRef}
                role="img"
                aria-label={ariaLabel}
                className="block w-full"
                style={{ height }}
            />
        </div>
    );
}
