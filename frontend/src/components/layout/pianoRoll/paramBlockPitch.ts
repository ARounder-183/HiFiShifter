/**
 * 块级音高处理：保留曲线轮廓，只把音高中心归一化到**最近的钢琴键**（半音）。
 *
 * 不能逐帧 `snapToScale`：颤音 / 滑音会被压成阶梯，听感与视觉都变成“直线”。
 * 这里先求该段主簇中心，再把**整段曲线**做固定音程平移，使其中心落在最近
 * 半音上；曲线内部的所有相对起伏原样保留。音阶不参与双击归一化，避免
 * “看起来落在黑键 / 非工程音阶”时被用户误判为没有吸附。
 */
import { snapToSemitone } from "../../../utils/musicalScales";

export function normalizePitchToNearestSemitone(args: {
    readonly values: readonly number[];
    readonly modeCenter: number | null;
}): number[] {
    if (args.values.length === 0) return [];
    const voiced = args.values.filter((value) => Number.isFinite(value) && value > 0);
    const center =
        args.modeCenter !== null && Number.isFinite(args.modeCenter) && args.modeCenter > 0
            ? args.modeCenter
            : voiced.length > 0
              ? voiced.reduce((sum, value) => sum + value, 0) / voiced.length
              : null;
    if (center === null) return args.values.slice();
    const target = snapToSemitone(center);
    const delta = target - center;
    if (Math.abs(delta) < 1e-9) return args.values.slice();
    return args.values.map((value) =>
        Number.isFinite(value) && value > 0 ? Math.max(0.01, value + delta) : value,
    );
}

/**
 * 块内音高纵向缩放：中心固定，所有可编辑帧相对中心按倍率缩放。
 *
 * - `factor > 1` 放大起伏（颤音 / 滑音更明显）；
 * - `factor < 1` 压平起伏（更接近稳定音高）；
 * - 0 哨兵保持 0；可用 `range` 钳制到最小 / 最大 MIDI。
 */
export function scalePitchAroundCenter(args: {
    readonly values: readonly number[];
    readonly center: number | null;
    readonly factor: number;
    readonly min?: number;
    readonly max?: number;
}): number[] {
    const factor = Number.isFinite(args.factor) ? Math.max(0, args.factor) : 1;
    const voiced = args.values.filter((value) => Number.isFinite(value) && value > 0);
    const center =
        args.center !== null && Number.isFinite(args.center) && args.center > 0
            ? args.center
            : voiced.length > 0
              ? voiced.reduce((sum, value) => sum + value, 0) / voiced.length
              : null;
    if (center === null) return args.values.slice();
    const min = args.min ?? 1;
    const max = args.max ?? 127;
    return args.values.map((value) => {
        if (!Number.isFinite(value) || value <= 0) return value;
        const scaled = center + (value - center) * factor;
        return Math.max(min, Math.min(max, scaled));
    });
}

/**
 * 横向滚轮 / Alt 纵向拖拽共用的音高缩放因子。
 *
 * `deltaPx > 0`（向上）放大起伏，反向压平；指数映射让每一次拖动都接近
 * “手感线性”，不会在接近 0 时突然变得毫无响应。
 */
export function pitchScaleFactorFromDrag(deltaPx: number): number {
    const px = Number.isFinite(deltaPx) ? deltaPx : 0;
    return Math.exp(px / 120);
}
