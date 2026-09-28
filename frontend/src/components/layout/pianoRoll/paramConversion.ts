/**
 * 音量（volume）↔ 动态（dyn）曲线互转 —— UI 侧的方向规划与可用性判定。
 *
 * ## 为什么需要互转
 *
 * 已经用音量画了一条包络、后来意识到它其实该是"动态"（或反之）时，
 * 用户不必重画 —— 一次菜单操作即可搬迁。
 *
 * ## 换算在哪里做（重要：不在前端！）
 *
 * 两者虽然同值域（0..4 倍率），但**语义不同量纲**：
 * - `volume` 是乘性增益：最终增益 = 值本身；
 * - `dyn` 是目标电平：最终增益 = 值 / 原声基线（`compute_dyn_gain`）。
 *
 * 因此"纯搬迁 = 无损"的假设是**错误的**（增益会变成 原值/基线 ≠ 原值）。
 * 等效互转需要逐帧基线补偿（`dyn_target = volume × orig`、
 * `volume = target/orig`），而权威基线与"曲线哪些帧有数据"只有后端知道，
 * 所以往返换算、哨兵帧与静音保护帧的处理全部在**后端命令**
 * `convert_mix_param`（`commands/params.rs`）内完成，前端不做任何算术。
 *
 * 本模块只负责：
 * 1. 方向规划（编辑的是哪个参数 → 命令方向与切换目标）；
 * 2. 菜单项可用性（必须有选区）。
 */

import type { ParamName } from "./types";

/** 互转方向（与后端 `MixConversionDirection` 对应）。 */
export type ParamConversionDirection = "volume_to_dyn" | "dyn_to_volume";

/** 一次互转的计划：传给后端命令的方向 + 完成后切换显示的目标参数。 */
export interface ParamConversionPlan {
    /** 互转方向（命令入参 `from`）。 */
    direction: ParamConversionDirection;
    /** 完成后切换显示的参数（= 迁入方，用户立刻看到搬迁结果）。 */
    targetParam: ParamName;
    /** 发起互转的参数（= 迁出方，已在选区内归位）。 */
    sourceParam: ParamName;
}

/**
 * 生成互转计划（纯函数，便于单测与在 UI 中预览）。
 *
 * @param from 当前正在编辑的参数（`volume` 或 `dyn`）。
 * @returns 计划；`from` 不是这两个参数时返回 null（调用方据此隐藏菜单项）。
 */
export function planParamConversion(from: ParamName): ParamConversionPlan | null {
    if (from === "volume") {
        return {
            direction: "volume_to_dyn",
            targetParam: "dyn",
            sourceParam: "volume",
        };
    }
    if (from === "dyn") {
        return {
            direction: "dyn_to_volume",
            targetParam: "volume",
            sourceParam: "dyn",
        };
    }
    return null;
}

/**
 * 判断互转在当前选区下是否可用。
 *
 * 约束来自后端 `MAX_SELECTION_FRAMES` 与"必须有选区"：没有选区时无从搬迁
 * （那等价于"整条曲线互换"，语义上应该用两参数的初始化，而不是一个菜单项）。
 */
export function canConvertParam(args: {
    editParam: ParamName;
    selectionFrameCount: number;
}): boolean {
    const { editParam, selectionFrameCount } = args;
    if (planParamConversion(editParam) === null) return false;
    return selectionFrameCount > 0;
}
