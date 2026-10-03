/**
 * 「设备手感倍率」的轴向累加器。
 *
 * 【它与 `fineAxisDrag.ts` 是同一族，但管的是另一个倍率】那个管"精细调整修饰键"
 * （布尔、用户按下的），这里管**设备带来的倍率**：压感强度、触摸的前置精细斜坡、
 * 触控板的低速补偿。三者都要求同一条不变量 ——
 *
 *   **只缩放这一帧新走的那一段，绝不重算已经累计的总量。**
 *
 * 否则倍率一变，数值就会瞬间跳回某个旧位置（"闪回"），正在进行的拖拽被打断。
 * `fineAxisDrag.ts:1-18` 已经完整论证过这条规则，本模块只是把它推广到连续倍率。
 *
 * 【为什么不直接改 `fineAxisDrag` 接受连续倍率】它被 8 个文件、14 个调用点共用，
 * 且行为被 `fineAxisDrag.test.ts` 与 `VibratoDialog.test.tsx` 钉住（含"修饰键刚
 * 按下那一帧走 0.65"这条特定手感）。改签名会波及全部调用点并动摇既有断言。
 * 组合则完全不动它：设备倍率先过一次本模块，再把结果喂给它 —— 两级串联，
 * 每一级都满足"只乘增量"，因此整体也满足。
 *
 * 【为什么不需要过渡帧】`fineAxisDrag` 的过渡帧（0.65）是为了避免"倍率骤降时
 * 指针突然拽不动"的卡顿感。本模块的倍率要么连续变化（压感），要么只会**变快**
 * （触摸斜坡 0.35 → 1.0）—— 变快永远不会让人以为"拖不动了"，因此不需要过渡。
 *
 * 【设计约束】纯函数 + 纯数据，不接触 DOM / React，可在 node 环境完整单测。
 */

export interface AxisGainState {
    /** 上一次的**原始**累计位移。 */
    raw: number;
    /** 累计的**已缩放**位移，与 `raw` 同量纲、同起点。 */
    adjusted: number;
}

/** 新建一份手势状态（`raw` 与 `adjusted` 必须同起点，否则首帧会凭空产生位移）。 */
export function createAxisGainState(): AxisGainState {
    return { raw: 0, adjusted: 0 };
}

/**
 * 推进一帧，返回**累计的已缩放位移**。
 *
 * @param state 手势期间持续复用的状态（就地更新）。
 * @param nextRaw 本帧的**原始累计**位移（相对手势起点）。
 * @param gain 本帧的倍率；非有限或非正值按 `1` 处理（宁可不动，不可反向）。
 */
export function advanceAxisGain(state: AxisGainState, nextRaw: number, gain: number): number {
    if (!Number.isFinite(nextRaw)) return state.adjusted;
    const safeGain = Number.isFinite(gain) && gain > 0 ? gain : 1;
    const delta = nextRaw - state.raw;
    state.adjusted += delta * safeGain;
    state.raw = nextRaw;
    return state.adjusted;
}

/** 两轴版本（预览画布的深度 / 相位同时受同一个标量倍率影响）。 */
export interface AxisGainState2D {
    x: AxisGainState;
    y: AxisGainState;
}

/** 新建两轴状态。 */
export function createAxisGainState2D(): AxisGainState2D {
    return { x: createAxisGainState(), y: createAxisGainState() };
}

/** 两轴推进：同一个倍率同时作用于两轴。 */
export function advanceAxisGain2D(
    state: AxisGainState2D,
    nextX: number,
    nextY: number,
    gain: number,
): { x: number; y: number } {
    return {
        x: advanceAxisGain(state.x, nextX, gain),
        y: advanceAxisGain(state.y, nextY, gain),
    };
}
