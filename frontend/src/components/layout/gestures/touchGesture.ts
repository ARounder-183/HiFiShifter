/**
 * 触摸手势：单指平移 + 双指捏合缩放 + 惯性。
 *
 * ## 为什么不用现成的捏合实现（也不照抄常见写法）
 *
 * 常见的捏合实现是**比值律**：`factor = spreadNow / spreadBefore`。它在两根手指近乎
 * 平行时（spread 接近 0）会**爆炸** —— 分母趋零，一帧内缩放几十倍，画面直接跳到
 * 一个不可能的位置。而"两指靠近"恰恰是缩小手势最常见的收尾姿势。
 *
 * 这里用**绝对缩放律**：`factor = e^(k · Δspread)`。它只依赖"这一帧两根手指的距离
 * 变化了多少像素"，与当前距离无关：
 * - 永远不会爆炸（`e^0 = 1`，Δ 有界则 factor 有界）；
 * - 缩放速率与手势速度成正比，手感与手指移动的**物理距离**一致；
 * - 两指缓慢靠近时是平滑的指数衰减，不会突然停住。
 *
 * ## 为什么必须"同帧提交"
 *
 * 缩放的语义是"锚点像素不动"：`scrollLeft' = anchorSec · pxPerSec' - anchorX`。
 * 若 `pxPerSec` 与 `scrollLeft` 分两次提交（例如一个走 dispatch、一个走 ref），
 * 中间那一帧的视口是不自洽的 —— 画面会先缩放、再跳回，用户看到的是**振荡**。
 * 因此本模块只产出"一次调用里同时给出 factor 与锚点"，由调用方在同一个
 * `setViewport` 里提交两者。
 *
 * ## 为什么需要 2→1 手指的冷却
 *
 * 双指捏合结束时，通常是**先抬起一根**。剩下那根手指的位置与捏合中心不同，若立刻
 * 转成平移，画面会"跳一下"再跟着手指走 —— 用户以为手势失控了。150ms 冷却让那次
 * 抬起被完全忽略，只有用户**重新**移动时才进入平移。
 *
 * ## 设计约束
 * 纯状态机 + 纯函数，不碰 DOM / React，可在 node 环境完整单测。
 */

/** 捏合的绝对缩放系数：`factor = e^(k · Δspread)`。 */
export const PINCH_ZOOM_GAIN = 0.0026;

/** 2 指 → 1 指之后，多久才允许进入平移（毫秒）。 */
export const TWO_TO_ONE_COOLDOWN_MS = 150;

/** 单指平移的起手死区（像素）：小于它的抖动不算平移。 */
export const PAN_DEAD_ZONE_PX = INPUT_PROFILES_TOUCH_START_THRESHOLD();

/** 惯性衰减的半衰期（毫秒）。 */
export const INERTIA_HALF_LIFE_MS = 180;

/** 惯性速度低于此值即停止（像素/毫秒）。 */
export const INERTIA_STOP_VELOCITY = 0.02;

/** 手势可作用的轴。 */
export type GestureAxis = "both" | "x" | "y";

/** 一个触点。 */
export interface TouchPoint {
    x: number;
    y: number;
}

export interface TouchGestureHandlers {
    /** 平移开始（已越过死区）。 */
    onPanStart?: () => void;
    /**
     * 平移增量（像素）。
     *
     * 调用方应当**乘在自己的增量上**（见 `utils/axisGain.ts` 的不变量），而不是
     * 重新计算累计量。
     */
    onPan?: (dx: number, dy: number) => void;
    /** 平移结束，附最后的速度（像素/毫秒），供调用方启动惯性。 */
    onPanEnd?: (velocityX: number, velocityY: number) => void;
    /** 捏合开始。 */
    onPinchStart?: () => void;
    /**
     * 捏合。`factor` 是**本帧**的缩放倍率（>1 放大），`anchor` 是两指中点
     * （视口坐标）—— 调用方必须用它在同一帧里同时更新缩放与滚动位置。
     */
    onPinch?: (factor: number, anchorX: number, anchorY: number) => void;
    /** 捏合结束。 */
    onPinchEnd?: () => void;
}

export interface TouchGestureRecognizer {
    /** 一个触点按下。 */
    start(id: number, point: TouchPoint): void;
    /** 触点移动。 */
    move(id: number, point: TouchPoint): void;
    /** 触点抬起。 */
    end(id: number): void;
    /** 全部取消（系统打断 / 手势被上层接管）。 */
    cancel(): void;
    /** 当前处于哪种手势。 */
    readonly mode: "idle" | "pan" | "pinch";
}

/** `inputProfile` 的触摸起手阈值；抽成函数是为了避免模块加载顺序造成的循环依赖。 */
function INPUT_PROFILES_TOUCH_START_THRESHOLD(): number {
    // 8 与 `INPUT_PROFILES.touch.startThresholdPx` 一致。这里不 import 那个模块，
    // 是因为 `inputProfile` → `penInput` → DOM 判断的链路会把纯数学模块拖进浏览器
    // 环境；手势识别的单测需要在 node 里跑。
    return 8;
}

/**
 * 本帧的捏合倍率。
 *
 * @param deltaSpreadPx 两指距离相对上一帧的变化（正 = 分开 = 放大）。
 */
export function pinchZoomFactor(deltaSpreadPx: number): number {
    if (!Number.isFinite(deltaSpreadPx)) return 1;
    // 单帧变化量钳制在 ±400px：再大的值只可能来自"两根手指突然出现/消失"，
    // 而不是真实手势。不钳制的话，一次误采样会让画面瞬间缩放到极限。
    const clamped = Math.max(-400, Math.min(400, deltaSpreadPx));
    return Math.exp(PINCH_ZOOM_GAIN * clamped);
}

/**
 * 惯性位移的衰减。
 *
 * @param velocity 初始速度（像素/毫秒）。
 * @param elapsedMs 已经过去的时间（毫秒）。
 * @returns 从手势结束到此刻的累计位移。
 */
export function inertiaDisplacement(velocity: number, elapsedMs: number): number {
    if (!Number.isFinite(velocity) || elapsedMs <= 0) return 0;
    // 指数衰减的积分：v(t) = v0 · 2^(-t/h) ⇒ ∫ = v0 · h/ln2 · (1 - 2^(-t/h))
    const h = INERTIA_HALF_LIFE_MS;
    const decay = Math.pow(2, -elapsedMs / h);
    return ((velocity * h) / Math.LN2) * (1 - decay);
}

/** 惯性是否已经衰减到可以停止。 */
export function inertiaFinished(velocity: number, elapsedMs: number): boolean {
    if (elapsedMs >= 1200) return true;
    return (
        Math.abs(velocity) * Math.pow(2, -elapsedMs / INERTIA_HALF_LIFE_MS) < INERTIA_STOP_VELOCITY
    );
}

/**
 * 创建一个手势识别器。
 *
 * @param axis 允许的轴。标尺只横向、轨道头只纵向、画布两向。
 * @param handlers 回调。
 * @param now 时钟注入（测试用）。
 */
export function createTouchGestureRecognizer(
    axis: GestureAxis,
    handlers: TouchGestureHandlers,
    now: () => number = () => performance.now(),
): TouchGestureRecognizer {
    const points = new Map<number, TouchPoint>();
    let mode: "idle" | "pan" | "pinch" = "idle";
    /** 平移的"原点"：死区判定与累计量都相对它。 */
    let panOrigin: TouchPoint | null = null;
    let panStarted = false;
    /** 上一帧的速度采样（用于松手时的惯性）。 */
    let lastMoveAt = 0;
    let lastDelta: TouchPoint = { x: 0, y: 0 };
    let velocity: TouchPoint = { x: 0, y: 0 };
    /** 捏合上一帧的两指距离。 */
    let lastSpread: number | null = null;
    /** 2→1 冷却的截止时刻。 */
    let panCooldownUntil = 0;

    const axisX = axis !== "y";
    const axisY = axis !== "x";

    const twoPoints = (): [TouchPoint, TouchPoint] | null => {
        const values = [...points.values()];
        return values.length >= 2 ? [values[0], values[1]] : null;
    };

    const spreadOf = (a: TouchPoint, b: TouchPoint) => Math.hypot(a.x - b.x, a.y - b.y);
    const centerOf = (a: TouchPoint, b: TouchPoint) => ({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 });

    const resetPan = () => {
        panOrigin = null;
        panStarted = false;
        velocity = { x: 0, y: 0 };
        lastDelta = { x: 0, y: 0 };
    };

    return {
        get mode() {
            return mode;
        },
        start(id, point) {
            points.set(id, { ...point });
            if (points.size === 1) {
                // 冷却期内不进入平移：那是一次捏合的收尾，不是新的平移。
                if (now() < panCooldownUntil) {
                    mode = "idle";
                    return;
                }
                mode = "pan";
                panOrigin = { ...point };
                panStarted = false;
                lastMoveAt = now();
                return;
            }
            if (points.size === 2) {
                const pair = twoPoints();
                if (!pair) return;
                mode = "pinch";
                lastSpread = spreadOf(pair[0], pair[1]);
                // 捏合开始时把平移状态清掉：否则松手回落到单指时会带着捏合前的
                // 原点继续平移，画面会跳一段。
                resetPan();
                handlers.onPinchStart?.();
            }
        },
        move(id, point) {
            if (!points.has(id)) return;
            points.set(id, { ...point });

            if (mode === "pinch") {
                const pair = twoPoints();
                if (!pair || lastSpread === null) return;
                const spread = spreadOf(pair[0], pair[1]);
                const factor = pinchZoomFactor(spread - lastSpread);
                lastSpread = spread;
                if (factor !== 1) {
                    const center = centerOf(pair[0], pair[1]);
                    handlers.onPinch?.(factor, center.x, center.y);
                }
                return;
            }

            if (mode !== "pan" || panOrigin === null) return;
            const rawX = point.x - panOrigin.x;
            const rawY = point.y - panOrigin.y;
            if (!panStarted) {
                // 死区：起手的一段抖动不算平移，越过之后才真正开始。
                if (Math.hypot(rawX, rawY) < PAN_DEAD_ZONE_PX) return;
                panStarted = true;
                handlers.onPanStart?.();
            }
            const dx = axisX ? rawX - lastDelta.x : 0;
            const dy = axisY ? rawY - lastDelta.y : 0;
            lastDelta = { x: rawX, y: rawY };

            const at = now();
            const elapsed = Math.max(1, at - lastMoveAt);
            lastMoveAt = at;
            // 速度用指数滑动平均：单帧采样在触摸屏上抖动很大，直接用会让惯性
            // 忽快忽慢。
            velocity = {
                x: velocity.x * 0.6 + (dx / elapsed) * 0.4,
                y: velocity.y * 0.6 + (dy / elapsed) * 0.4,
            };
            if (dx !== 0 || dy !== 0) handlers.onPan?.(dx, dy);
        },
        end(id) {
            if (!points.delete(id)) return;
            if (mode === "pinch") {
                if (points.size >= 2) {
                    // 仍是双指（第三根手指抬起）：重新取基线。
                    const pair = twoPoints();
                    lastSpread = pair ? spreadOf(pair[0], pair[1]) : null;
                    return;
                }
                handlers.onPinchEnd?.();
                mode = "idle";
                lastSpread = null;
                // 2 → 1：进入冷却，剩下那根手指不立刻变成平移。
                panCooldownUntil = now() + TWO_TO_ONE_COOLDOWN_MS;
                resetPan();
                return;
            }
            if (points.size === 0) {
                if (mode === "pan" && panStarted) {
                    handlers.onPanEnd?.(velocity.x, velocity.y);
                }
                mode = "idle";
                resetPan();
            }
        },
        cancel() {
            if (mode === "pan" && panStarted) handlers.onPanEnd?.(0, 0);
            if (mode === "pinch") handlers.onPinchEnd?.();
            points.clear();
            mode = "idle";
            lastSpread = null;
            resetPan();
        },
    };
}
