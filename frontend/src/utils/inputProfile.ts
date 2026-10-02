/**
 * 指针设备的**能力剖面**：把"设备之间的差异"收成一张数据表。
 *
 * 【为什么是数据表而不是 if 分支】
 * 每引入一种设备就在手势代码里加一个 `if (isPen) … else if (isTouch) …`，会让
 * 每个交互面各自长出一套判断，且必然漏掉几处（现状正是如此：`penInput.ts` 的
 * 判定只接进了 6 个调用点）。这里把差异写成常量，交互面统一问"这个设备是什么
 * 剖面"，于是新增设备 = 新增一行，而不是新增一批分支。
 *
 * 【它和 `penInput.ts` 的分工】
 * - `penInput.ts`：**判定**一次事件属于哪类设备（纯布尔 / 归类），无状态。
 * - 本文件：**描述**这类设备有哪些能力、容差该放多大、有没有另一只手可用。
 *   前者是输入，后者是策略。
 *
 * 【与 `fineAxisDrag.ts` 的关系】
 * 本文件只产出"倍率"（`dragGain` 等），不自己做位移缩放 —— 缩放必须由调用方
 * **乘在增量上**（见 `fineAxisDrag.ts` 头部：只按增量缩放，绝不重算累计量）。
 * 本文件是那条规则的**数据来源**，不是它的替代品。
 *
 * 【设计约束】
 * 1. **mouse 剖面必须与引入本文件之前的行为逐字节一致**：所有 scale 为 1、
 *    threshold 为原值。这是零回归的保证，也由单测钉住。
 * 2. **未知设备按鼠标对待**（沿用 `penInput.ts:28-30` 的宽松回退）：合成事件与
 *    测试桩不能因为本文件被意外拦截。
 * 3. 纯数据 + 纯查询，不接触 DOM / React，可在 node 环境完整单测。
 */

import { pointerKindOf, type PointerKind } from "./penInput";

/**
 * 次级手势的长按判定（毫秒）。
 *
 * 【为什么与 `useRepeatPress` 的 320ms 分开】
 * `useRepeatPress` 是**重复触发**（按住不放连续触发），长按次级手势是**一次性
 * 状态转换**（按住 = 进入次级模式）。两者语义不同，不能共用一个值 —— 否则
 * "右键菜单按 320ms 出、画布整体变换要按 400ms"这类不一致会固化下来。
 * 取 400ms 是因为它必须明显长于触摸的起始阈值判定时间（8px），否则轻抖会先
 * 触发拖拽。
 */
export const SECONDARY_LONG_PRESS_MS = 400;

/**
 * 触摸的"前置精细斜坡"长度（CSS 像素）。
 *
 * 触摸既没有修饰键也没有第二根手指可用，无法进入精细模式 —— 这是硬阻断。
 * 做法是：接触后的前 `TOUCH_PRECISION_RAMP_PX` 像素按 `TOUCH_PRECISION_RAMP_GAIN`
 * 走，之后恢复 1.0。"轻推 = 微调、长拖 = 大范围"。
 */
export const TOUCH_PRECISION_RAMP_PX = 12;
/** 斜坡区间内的位移倍率。 */
export const TOUCH_PRECISION_RAMP_GAIN = 0.35;

/**
 * 一台指针设备的能力与手感剖面。
 *
 * 字段分三组：**几何**（容差 / 阈值）、**通道**（有哪些输入维度）、
 * **手感**（增益）。
 */
export interface InputProfile {
    /** 设备归类。`trackpad` 在指针层不可分辨，只能由显式设置叠加（见 §4.2）。 */
    kind: PointerKind;

    // ── 几何 ──────────────────────────────────────────────────────
    /**
     * 命中容差倍率：所有 `*_HIT_PX` 常量拿到基准值后乘它。
     *
     * 【为什么 pen 只有 1.15 而 touch 是 2.4】笔尖在数位板上有物理级坐标精度，
     * 本体精度高于鼠标，只需补偿"手离屏幕远、看不清 7px 方块"；而手指是 9mm
     * 接触面 + 遮挡，是数量级差异。把两者做成同一个倍数会让笔用户觉得目标过大，
     * 相邻手柄互相打架（`hitTestPreviewZone` 的"两个手柄重叠取近的"会随之失真）。
     */
    hitRadiusScale: number;
    /** 判定为"拖拽"而非"点击"的最小位移（CSS 像素）。 */
    startThresholdPx: number;
    /** 次级手势的长按判定；该设备本就有右键时为 `null`。 */
    secondaryLongPressMs: number | null;

    // ── 通道 ──────────────────────────────────────────────────────
    /** 是否存在真实悬停 —— 决定 hover 能否作为主要反馈通道。 */
    hasHover: boolean;
    /** 悬停副作用（按下即 seek / 按下即弹浮窗）是否保留。 */
    allowsHoverSideEffects: boolean;
    /** 是否有可信的压力通道。 */
    hasPressure: boolean;
    /** 是否有可信的倾斜通道。 */
    hasTilt: boolean;
    /**
     * 操作期间是否有另一只手空着 —— 决定"修饰键 = 精细"这套策略是否成立。
     *
     * 握笔时另一只手通常扶着板子，触屏时手指就在屏幕上；这两种设备都必须用手上
     * 已有的自由度（压力、位移斜坡）置换出精细模式，而不能指望键盘。
     */
    freeHand: boolean;

    // ── 手感 ──────────────────────────────────────────────────────
    /** 参与拖拽的表面应声明的 `touch-action`。 */
    touchAction: "none" | "pan-y" | "auto";
    /** 位移增益：设备自身的分辨率 / 加速度差异在这里一遍抵扣。 */
    dragGain: number;
}

/**
 * 各设备的能力剖面。
 *
 * 【mouse 一行是"恒等元"】所有 scale = 1、threshold 取原值、通道按鼠标的实际
 * 能力（有悬停、无压力）。它是零回归的锚点：任何"顺手把设备一起改了"的改动都会
 * 在这里被单测挡下。
 */
export const INPUT_PROFILES: Record<PointerKind, InputProfile> = {
    mouse: {
        kind: "mouse",
        hitRadiusScale: 1,
        startThresholdPx: 3,
        // 鼠标有右键，不需要长按兜底。
        secondaryLongPressMs: null,
        hasHover: true,
        allowsHoverSideEffects: true,
        hasPressure: false,
        hasTilt: false,
        freeHand: true,
        touchAction: "auto",
        dragGain: 1,
    },
    pen: {
        kind: "pen",
        // 只补偿"手离屏幕远"，不是数量级差异（见字段注释）。
        hitRadiusScale: 1.15,
        startThresholdPx: 3,
        // 笔杆键（button 2）已由 `isSecondaryButtonDown` 覆盖，但低端笔没有
        // 笔杆键、且按下时常把笔尖推向橡皮 —— 长按是零成本兜底。
        secondaryLongPressMs: SECONDARY_LONG_PRESS_MS,
        hasHover: true,
        // 笔尖未接触就上报 pointermove，悬停副作用会乱跳（penInput.ts:11-14）。
        allowsHoverSideEffects: false,
        hasPressure: true,
        hasTilt: true,
        // 握笔时另一只手通常扶着板子 —— 不能指望键盘修饰键。
        freeHand: false,
        touchAction: "auto",
        dragGain: 1,
    },
    touch: {
        kind: "touch",
        // 手指接触面约 9mm，基准容差小了一个量级。
        hitRadiusScale: 2.4,
        startThresholdPx: 8,
        secondaryLongPressMs: SECONDARY_LONG_PRESS_MS,
        hasHover: false,
        allowsHoverSideEffects: false,
        hasPressure: false,
        hasTilt: false,
        freeHand: false,
        touchAction: "none",
        // 略低于 1：手指在屏幕上没有光标，缺少"手眼反馈"，同一位移显得更远。
        dragGain: 0.85,
    },
    unknown: {
        // 合成事件 / 测试桩：与 mouse 完全同构，保持宽松回退。
        kind: "unknown",
        hitRadiusScale: 1,
        startThresholdPx: 3,
        secondaryLongPressMs: null,
        hasHover: true,
        allowsHoverSideEffects: true,
        hasPressure: false,
        hasTilt: false,
        freeHand: true,
        touchAction: "auto",
        dragGain: 1,
    },
};

/**
 * 取一个事件所属设备的剖面。
 *
 * 未知 / 缺失 `pointerType` 一律按鼠标（宽松回退），与 `penInput.ts:28-30` 同口径。
 */
export function profileFor(event: { pointerType?: string | null }): InputProfile {
    return INPUT_PROFILES[pointerKindOf(event.pointerType)];
}

/**
 * 用户可显式声明的指针设备。
 *
 * 【为什么需要人工声明】Web 平台在"这块板是触控板还是鼠标"这件事上没有可靠信号：
 * `WheelEvent` 不带 `pointerType`，触控板在指针层就是 `"mouse"`。`auto` 走既有
 * 启发式（行为不变），显式设定后跳过猜测。项目里已有同类先例（键位预设的
 * `trackpad`）。类型定义在本模块，`settings.ts` 再导出 —— 让"有哪些设备"只有
 * 一个出处。
 */
export type PointerDeviceDeclaration = "auto" | "mouse" | "trackpad" | "pen" | "touch";

/**
 * 取剖面的**声明感知**版本：用户显式声明了设备时以声明为准，否则按事件推断。
 *
 * 【触控板为什么只能这样】它不是一种 `pointerType`（指针层就是 `"mouse"`），
 * 因此只能靠声明；声明为触控板时使用 mouse 剖面（它有鼠标的全部能力），
 * 低速补偿另由 `trackpadInertiaGain` 提供。
 *
 * @param declared 设置里的显式声明。
 * @param event 触发本次判定的指针事件。
 */
export function profileForDeclared(
    declared: PointerDeviceDeclaration | null | undefined,
    event: { pointerType?: string | null },
): InputProfile {
    switch (declared) {
        case "mouse":
            return INPUT_PROFILES.mouse;
        case "trackpad":
            return INPUT_PROFILES.mouse;
        case "pen":
            return INPUT_PROFILES.pen;
        case "touch":
            return INPUT_PROFILES.touch;
        default:
            return profileFor(event);
    }
}

/** 按设备剖面缩放一个基准命中半径。 */
export function scaledHitRadius(basePx: number, profile: InputProfile): number {
    return basePx * profile.hitRadiusScale;
}

/**
 * 按设备剖面缩放一个基准起手阈值。
 *
 * 基准以 **mouse 剖面的 `startThresholdPx`** 为准（各交互面现有的 3/4/5px 常量都是
 * 按鼠标调的），因此 mouse 路径上返回原值，零回归。
 */
export function scaledThreshold(basePx: number, profile: InputProfile): number {
    return (basePx * profile.startThresholdPx) / INPUT_PROFILES.mouse.startThresholdPx;
}

/**
 * 触摸的"前置精细斜坡"倍率。
 *
 * 接触后的前 `TOUCH_PRECISION_RAMP_PX` 像素按 `TOUCH_PRECISION_RAMP_GAIN` 走，
 * 之后恢复 1.0。只对触摸生效 —— 鼠标有修饰键、笔有压力通道，都不需要它。
 *
 * 【为什么是纯函数且只吃距离】斜坡必须能随手势进度连续变化，而"只乘增量"的
 * 不变量（见 `fineAxisDrag.ts`）要求这个倍率**每帧可重新计算**。因此它不能有
 * 内部状态，只能由"已经走了多远"决定。
 *
 * @param profile 设备剖面。
 * @param travelledPx 本次手势自起点累计的原始位移（像素）。
 */
export function precisionRampGain(profile: InputProfile, travelledPx: number): number {
    if (profile.kind !== "touch") return 1;
    return Math.abs(travelledPx) < TOUCH_PRECISION_RAMP_PX ? TOUCH_PRECISION_RAMP_GAIN : 1;
}

/**
 * 笔杆倾斜 → 波形偏斜（`skew ∈ [0.02, 0.98]`，0.5 = 不偏）。
 *
 * 【为什么是绝对映射而不是增量】偏斜是一个"摆放姿态"式的量（0.5 = 对称），
 * 与相位、深度不同：它没有"从某个起点走多远"的自然含义，用户看着波形说"往右偏
 * 一点"时心里想的就是一个绝对形状。因此直接由倾斜角度定值。
 *
 * 【为什么默认关】倾斜是三维输入里最不可靠的一轴：大量设备不报，且握笔姿势一变
 * 值就漂。绑一个会漂移的语义比留白更糟，所以只在用户显式打开时生效。
 *
 * 【调用方还要再判一次"形状是否用偏斜"】本函数只回答"倾斜说该偏多少"，不问
 * 当前波形认不认这个字段 —— 那是 `vibratoCycle.shapeUsesSkew` 的事。两者分开
 * 才能让"正弦预设上拖一下"不会把偏斜悄悄改掉。
 */
export function tiltToSkew(tiltX: number): number {
    if (!Number.isFinite(tiltX)) return 0.5;
    // ±90° 映射到 0.02..0.98，中位 0.5。
    const normalized = Math.max(-1, Math.min(1, tiltX / 90));
    return Math.max(0.02, Math.min(0.98, 0.5 + normalized * 0.48));
}

/**
 * 触控板的"低速段补偿"倍率。
 *
 * 【为什么需要】触控板拖拽的 delta 带操作系统级加速度曲线：快速划动时像素/毫米
 * 比很高，慢速微调时反而掉得厉害。对小位移额外放大，抵消"越想精细越拖不动"。
 *
 * 【为什么按设备显式声明而不是嗅探】触控板在指针层就是 `"mouse"`，Web 平台没有
 * 可靠信号（`WheelEvent` 不带 `pointerType`）。因此只有用户在设置里显式声明
 * `inputDevice: "trackpad"` 时才启用 —— 默认的 `auto` 不改变任何既有行为。
 *
 * @param trackpadDeclared 用户是否显式声明了触控板。
 * @param deltaPx 本帧位移（像素）。
 */
export function trackpadInertiaGain(trackpadDeclared: boolean, deltaPx: number): number {
    if (!trackpadDeclared) return 1;
    const magnitude = Math.abs(deltaPx);
    // 只补偿低速段：超过 2px/帧 已是"快划"，交给系统加速度曲线即可。
    if (magnitude >= 2 || magnitude <= 0) return 1;
    return 1.6;
}
