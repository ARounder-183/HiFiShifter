/**
 * 拖拽手势的「设备手感倍率」。
 *
 * 【它把三件独立的事收成一处】
 *   1. **压感**：数位笔轻按 = 精细、重按 = 快速（连续）；
 *   2. **触摸前置斜坡**：手指刚落下的一小段按 0.35 倍走，补上触屏"没有修饰键
 *      也没有第二根手指"造成的精细模式缺口；
 *   3. **触控板低速补偿**：抵消操作系统加速度曲线在慢速段的掉速。
 *
 * 【为什么收进一处】三者的乘积就是"本帧走多快"。散在各调用点算，一定会漏掉几处
 * （这正是 `penInput.ts` 只接进了 6 个调用点的老问题）。而且它们必须**同时**
 * 遵守同一条不变量 —— 只乘在这一帧的增量上（见 `utils/axisGain.ts`）。
 *
 * 【为什么放在 `src/ui` 且读 Redux】与 `useFineAdjustModifier` 同一个理由：
 * `src/ui` 是本应用共生的设计系统层，读一个设置换来的"能力无法被忘记"是划算的。
 *
 * 【与 `useFineAdjustModifier` 的关系】两者互补、可叠加：那个是用户按下的
 * **离散**修饰键（键盘），这个是设备自带的**连续**信号。本模块**不**碰修饰键，
 * 因此调用方仍按原样先过 `advanceFineAxisDrag`。
 *
 * 【设计约束】倍率的计算是纯函数（`pressureCurve` / `inputProfile`），本 hook 只
 * 负责"读设置 + 持有每段手势的状态"。
 */
import { useEffect, useMemo, useRef } from "react";

import { useAppSelector } from "../app/hooks";
import { createAxisGainState2D, advanceAxisGain2D, type AxisGainState2D } from "../utils/axisGain";
import {
    precisionRampGain,
    profileForDeclared,
    tiltToSkew,
    trackpadInertiaGain,
    type InputProfile,
} from "../utils/inputProfile";
import {
    createPressureCalibration,
    observePressure,
    pressureLooksConstant,
    pressureToGain,
    pressureToWeight,
    type PressureCalibration,
    type PressureConfig,
} from "../utils/pressureCurve";

/** 倍率计算所需的指针事件字段（原生与 React 事件都满足）。 */
export interface InputGainEvent {
    pointerType?: string | null;
    pressure?: number;
    /** `PointerEvent.tiltX`（度，`-90..90`）。 */
    tiltX?: number;
}

/**
 * 判定"这台设备根本不报压感"所需的样本数。
 *
 * 太少会被一笔轻描淡写误判，太多则真·无压感设备的错误手感会持续可见。
 * 8 个采样点在任何采样率下都不到一帧的十分之一。
 */
const CONSTANT_PRESSURE_SAMPLE_LIMIT = 8;

export interface InputGainController {
    /** 手势开始：重置压感标定与采样。必须在 pointerdown 时调用。 */
    reset(): void;
    /**
     * 本帧的位移倍率（正实数，1 = 与鼠标同速）。
     *
     * @param event 指针事件快照（读 `pointerType` / `pressure`）。
     * @param travelledPx 相对手势起点的累计位移（触摸斜坡用）。
     * @param deltaPx 本帧位移（触控板低速补偿用）。
     */
    gainFor(event: InputGainEvent, travelledPx: number, deltaPx: number): number;
    /**
     * 手绘笔画这一点的**写入权重**（1 = 完全覆盖，即鼠标的既有行为）。
     *
     * 与 `gainFor` 共用同一份压感标定与"该设备是否真的在报压感"的判定 ——
     * 于是"多用力 = 多写一点"与"多用力 = 走得快一点"是同一个肌肉记忆，
     * 且无压感设备在两条路径上都会退化成 1。
     */
    paintWeightFor(event: InputGainEvent): number;
    /**
     * 笔杆倾斜对应的目标 `skew`；未启用倾斜 / 设备不报倾斜时为 `null`。
     *
     * 返回 `null` 而不是一个默认值是有意的：调用方据此决定"要不要写这个字段"，
     * 而 `0.5` 之类的默认值会让每个不报倾斜的设备在每次拖拽时都改一次偏斜。
     */
    tiltSkewFor(event: InputGainEvent): number | null;
    /**
     * 便利方法：两轴累计位移一次过倍率。
     *
     * 【为什么必须由本模块持有累计状态】"只乘增量"要求记住上一帧的原始累计值；
     * 让调用方自己维护就等于把不变量交给了每个调用点。
     */
    advanceXY(
        state: AxisGainState2D,
        nextX: number,
        nextY: number,
        event: InputGainEvent,
        deltaPx: number,
    ): { x: number; y: number };
}

/** 从设置里取出压感曲线参数（与 `utils/pressureCurve.ts` 的默认值同构）。 */
function pressureConfigFrom(settings: {
    pressureDeadZone: number;
    pressureFloor: number;
    pressureCeiling: number;
    pressureMinGain: number;
    pressureMaxGain: number;
    pressureGamma: number;
}): PressureConfig {
    return {
        deadZone: settings.pressureDeadZone,
        floor: settings.pressureFloor,
        ceiling: settings.pressureCeiling,
        minGain: settings.pressureMinGain,
        maxGain: settings.pressureMaxGain,
        gamma: settings.pressureGamma,
    };
}

export function useInputModifiers(): InputGainController {
    const settings = useAppSelector((state) => state.session.penInput);

    /*
     * 设置经 ref 转发：控制器的方法在事件/帧回调里执行，那时本次渲染的闭包可能
     * 已过期。写入放在 effect 里（不是渲染期），与 `useFrameCommit` 一致。
     */
    const settingsRef = useRef(settings);
    useEffect(() => {
        settingsRef.current = settings;
    });

    /*
     * 每段手势一份的状态。
     *
     * 放在 ref 里而不是 state：它随指针事件高频更新，进 state 会让每个采样点都
     * 触发一次渲染 —— 正是本方案要消除的那类开销。
     */
    const calibrationRef = useRef<PressureCalibration>(createPressureCalibration());
    const samplesRef = useRef<number[]>([]);
    /** 该设备是否仍然可信地提供压感（恒定压力会被判为不可信）。 */
    const trustPressureRef = useRef(true);

    return useMemo<InputGainController>(() => {
        const reset = () => {
            calibrationRef.current = createPressureCalibration();
            samplesRef.current = [];
            trustPressureRef.current = true;
        };

        const gainFor = (
            event: InputGainEvent,
            travelledPx: number,
            deltaPx: number,
        ): number => {
            const current = settingsRef.current;
            const profile: InputProfile = profileForDeclared(current.device, event);

            // ── 压感 ───────────────────────────────────────────────
            const pressureGain = samplePressure(event, profile)
                ? pressureToGain(
                      typeof event.pressure === "number" ? event.pressure : 0,
                      pressureConfigFrom(current),
                      calibrationRef.current,
                  )
                : 1;

            // ── 触摸前置精细斜坡 ────────────────────────────────────
            const rampGain = current.touchPrecisionRamp
                ? precisionRampGain(profile, travelledPx)
                : 1;

            // ── 触控板低速补偿（仅在用户显式声明触控板时启用） ──────
            const inertiaGain = trackpadInertiaGain(current.device === "trackpad", deltaPx);

            return profile.dragGain * pressureGain * rampGain * inertiaGain;
        };

        const advanceXY = (
            state: AxisGainState2D,
            nextX: number,
            nextY: number,
            event: InputGainEvent,
            deltaPx: number,
        ) => {
            const travelled = Math.hypot(nextX, nextY);
            return advanceAxisGain2D(state, nextX, nextY, gainFor(event, travelled, deltaPx));
        };

        /**
         * 记录一次压力采样并判定该设备是否可信。
         *
         * 【为什么抽出来】拖拽倍率与画笔权重都要做同一件事：喂样本、抬标定上界、
         * 在"恒定压力"出现后停止相信它。分开写两遍必然有一遍漏掉某个环节。
         *
         * @returns 是否应当**采用**压力（false = 该设备没在报压感，按 1 处理）。
         */
        const samplePressure = (event: InputGainEvent, profile: InputProfile): boolean => {
            const current = settingsRef.current;
            if (!current.pressureEnabled || !profile.hasPressure) return false;
            if (!trustPressureRef.current) return false;
            const raw = typeof event.pressure === "number" ? event.pressure : 0;
            const samples = samplesRef.current;
            samples.push(raw);
            observePressure(calibrationRef.current, raw);
            if (
                samples.length >= CONSTANT_PRESSURE_SAMPLE_LIMIT &&
                pressureLooksConstant(samples)
            ) {
                /*
                 * 恒定压力 = 这台设备根本没在报压感（或驱动被禁用）。继续按它映射
                 * 会把整段手势钉在一个非 1 的倍率上，用户只觉得"这次拖得莫名其妙
                 * 地慢/快"。本段余下按 1 处理。
                 */
                trustPressureRef.current = false;
                return false;
            }
            return true;
        };

        const paintWeightFor = (event: InputGainEvent): number => {
            const current = settingsRef.current;
            const profile = profileForDeclared(current.device, event);
            if (!samplePressure(event, profile)) return 1;
            return pressureToWeight(
                typeof event.pressure === "number" ? event.pressure : 0,
                pressureConfigFrom(current),
                calibrationRef.current,
            );
        };

        const tiltSkewFor = (event: InputGainEvent): number | null => {
            const current = settingsRef.current;
            if (!current.tiltEnabled) return null;
            const profile = profileForDeclared(current.device, event);
            if (!profile.hasTilt) return null;
            // 不报倾斜的设备会一直给 0（= 竖直握笔），那是"没有这个通道"而不是
            // "用户想要对称"。只认非零值，避免每次拖拽都把偏斜归到 0.5。
            const tiltX = typeof event.tiltX === "number" ? event.tiltX : 0;
            if (!Number.isFinite(tiltX) || tiltX === 0) return null;
            return tiltToSkew(tiltX);
        };

        return { reset, gainFor, advanceXY, paintWeightFor, tiltSkewFor };
    }, []);
}

/** 新建一份两轴累计状态（调用方在 pointerdown 时创建，手势结束即丢弃）。 */
export { createAxisGainState2D };
export type { AxisGainState2D };
