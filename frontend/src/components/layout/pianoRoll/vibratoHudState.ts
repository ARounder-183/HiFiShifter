/**
 * 颤音拖拽 HUD 的载荷与锚点规则（纯函数，可单测）。
 *
 * 【它解决什么】深度 / 速率不只有指针能改：滚轮与方向键同样在改。曲线是**命令式**
 * 重画的（立即生效），而 HUD 读数走 React 状态 —— 一旦某条调参路径因为"手上没有
 * 指针坐标"就跳过上报，气泡里的数字就只能等下一次指针移动才追上，用户看到的是
 * **"读数比曲线慢一拍"**。
 *
 * 键盘事件天生没有指针坐标，因此上报不能要求坐标；规则是"带新坐标就更新锚点，
 * 省略就沿用上一次"。把这条规则写在这里，而不是散在调用点，是为了让"省略即沿用"
 * 可以被单测钉住 —— 这类滞后不会抛错，只会让人觉得工具"钝"。
 */

/** 面板侧 HUD 的载荷（与 `PianoRollPanel` 的 `vibratoDragHud` 状态同形）。 */
export interface VibratoHudState {
    presetId: string;
    depthCents: number;
    rateHz: number;
    adjusted: boolean;
    clientX: number;
    clientY: number;
}

/** HUD 的锚点：指针在画布坐标系里的位置。 */
export interface VibratoHudAnchor {
    clientX: number;
    clientY: number;
}

/** 工作副本里 HUD 需要的字段。 */
export interface VibratoHudSource {
    presetId: string;
    depthCents: number;
    rateHz: number;
    depthAdjusted: boolean;
    rateAdjusted: boolean;
}

/**
 * 锚点更新规则：**带坐标就换，省略就沿用**。
 *
 * @returns 新的锚点；从未拿到过坐标时为 `null`（调用方据此放弃上报，宁可不动，
 *   也不要把气泡甩到 `(0, 0)`）。
 */
export function nextVibratoHudAnchor(
    previous: VibratoHudAnchor | null,
    clientX?: number,
    clientY?: number,
): VibratoHudAnchor | null {
    if (clientX != null && clientY != null) return { clientX, clientY };
    return previous;
}

/**
 * 由工作副本与锚点算出 HUD 载荷。
 *
 * "已调整"是聚合标记：深度或速率任一项被调过就点亮（两者在切换预设时各自独立继承，
 * 见 `vibratoDragAdjust.ts` 的 `switchDragPreset`）。
 *
 * @returns 没有锚点时返回 `null`。
 */
export function vibratoHudState(
    source: VibratoHudSource,
    anchor: VibratoHudAnchor | null,
): VibratoHudState | null {
    if (!anchor) return null;
    return {
        presetId: source.presetId,
        depthCents: source.depthCents,
        rateHz: source.rateHz,
        adjusted: source.depthAdjusted || source.rateAdjusted,
        clientX: anchor.clientX,
        clientY: anchor.clientY,
    };
}
