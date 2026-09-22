/**
 * 参数编辑器纵轴标尺的**展示单位**（倍率 ↔ dB）。
 *
 * 【为什么需要这个文件】音量（volume）与动态（dyn）的曲线值是**线性幅值倍率**
 * （1.0 = 数字满量程）。用户读曲线时常常更需要 dB —— 而 1× 恰为 0 dB，两者是
 * 同一个量的两种读法，可以无损互转。把"用哪种单位读"做成用户设置后，刻度标签
 * 与悬浮浮窗必须**同源**地换算，否则同一个位置会出现"刻度写 −6、浮窗写 0.501"
 * 这类自相矛盾的读数。换算与解析收在本模块（纯函数，有单测）避免分叉。
 *
 * 【为什么只是显示单位，不动刻度位置】刻度线的**值**（及其在轴上的位置）描述的是
 * 数据本身，与读法无关；切单位只换标签文本。刻度几何因此完全不重建，只重算文字，
 * 也让"用户切单位"与"数据没变"的直觉一致（曲线不会跳动）。
 *
 * 【与后端的关系】纯前端显示换算，不参与渲染：音频侧始终按线性倍率求值。
 * 唯一的持久化是用户设置里的 `paramAxisUnits` 映射（见 sessionSlice）。
 */

import { isDynParam, VOLUME_PARAM_ID } from "./paramRanges";

/** 纵轴展示单位。 */
export type ParamAxisUnit = "ratio" | "db";

/** 默认单位：倍率（与历史行为一致）。 */
export const DEFAULT_PARAM_AXIS_UNIT: ParamAxisUnit = "ratio";

/** 支持切换展示单位的参数（音量 / 动态）→ 当前单位。 */
export type ParamAxisUnits = Record<string, ParamAxisUnit>;

/**
 * 旧版音量参数 id（`hifigan_volume`）。
 *
 * 与 `clampParamWriteValue` / 面板的标签分支同一组别名：它可能出现在旧工程里，
 * 换算与切换逻辑必须同样认它，否则打开旧工程后音量轴切不了单位。
 */
const LEGACY_VOLUME_PARAM_ID = "hifigan_volume";

/**
 * 该参数的纵轴是否支持「倍率 / dB」切换。
 *
 * 只覆盖**线性幅值倍率**语义的参数：音量与动态。其余参数（音高 / 音分 / 度数 /
 * 共振峰 / 张力…）的单位由参数语义唯一确定，没有第二种读法，点击不应有任何反应。
 *
 * @param param 参数名。
 * @returns 是否支持切换。
 */
export function supportsParamAxisUnit(param: string | null | undefined): boolean {
    if (param == null) return false;
    return param === VOLUME_PARAM_ID || param === LEGACY_VOLUME_PARAM_ID || isDynParam(param);
}

/**
 * 取参数的展示单位（未设置时回退默认值）。
 *
 * @param units 用户设置里的映射（可能为 undefined / 缺项）。
 * @param param 参数名。
 * @returns 该参数的展示单位。
 */
export function resolveParamAxisUnit(
    units: ParamAxisUnits | null | undefined,
    param: string,
): ParamAxisUnit {
    const value = units?.[param];
    return value === "db" ? "db" : DEFAULT_PARAM_AXIS_UNIT;
}

/**
 * 取「切换后的单位」。
 *
 * 【为什么不叫 `toggleParamAxisUnit`】slice 里的同名 reducer 也叫这个名字；方法
 * 简写会在函数体内绑定自己的名字，切片内调用同名 helper 会变成**递归调用自己**。
 * 名字区分开后两边都不必小心翼翼。
 */
export function nextParamAxisUnit(unit: ParamAxisUnit): ParamAxisUnit {
    return unit === "db" ? "ratio" : "db";
}

/**
 * 规范化持久化读回的映射：丢弃非支持参数与非法取值。
 *
 * 【为什么要过滤而不是原样信任】设置文件是用户可以手改的纯 JSON：写入未知参数名
 * 或 `"dB"` 这类大小写/拼写变体时，若原样进入 Redux，后续所有 `=== "db"` 判定
 * 都会静默落到倍率分支 —— 表现为"设置里明明写着 dB，界面却还是倍率"。这里统一
 * 收口成"只保留合法键值"，坏数据等于没有数据。
 *
 * @param raw 读回的原始值（任意 JSON）。
 * @returns 规范化后的映射。
 */
export function normalizeParamAxisUnits(raw: unknown): ParamAxisUnits {
    const out: ParamAxisUnits = {};
    if (raw == null || typeof raw !== "object") return out;
    for (const [key, value] of Object.entries(raw as Record<string, unknown>)) {
        if (!supportsParamAxisUnit(key)) continue;
        if (value !== "ratio" && value !== "db") continue;
        out[key] = value;
    }
    return out;
}

/**
 * 线性幅值倍率 → dB（`20 · log10(ratio)`）。
 *
 * 用 20 倍而不是 10 倍：曲线值是**幅值**倍率而非功率倍率，1× = 0 dB、0.5× =
 * −6.02 dB（半个幅值）正是用户对"音量减半"的预期。
 *
 * @param ratio 线性幅值倍率。
 * @returns dB 值；`ratio ≤ 0`（静音）时为 `-Infinity`。
 */
export function ratioToDb(ratio: number): number {
    if (!Number.isFinite(ratio) || ratio <= 0) return Number.NEGATIVE_INFINITY;
    return 20 * Math.log10(ratio);
}

/**
 * 把倍率值格式化为主标签风格的 dB 文本（**不带 `dB` 后缀**，供刻度标签用）。
 *
 * 规则：保留 1 位小数并去掉无意义的 `.0`（刻度标签要短才能排进 56px 的轴列）；
 * 正值带 `+` 号（与增益刻度的惯例一致，也避免"6"被读成衰减 6 dB）；
 * 静音（倍率 0）写 `-∞`（−∞ dB 无法写成有限数字）。
 *
 * @param ratio 线性幅值倍率。
 * @returns 标签文本。
 */
export function formatDbLabel(ratio: number): string {
    const db = ratioToDb(ratio);
    if (!Number.isFinite(db)) return "-∞";
    const rounded = Math.round(db * 10) / 10;
    if (rounded === 0) return "0";
    const abs = Math.abs(rounded);
    const text = Number.isInteger(abs) ? `${abs}` : abs.toFixed(1);
    return rounded > 0 ? `+${text}` : `-${text}`;
}

/**
 * 把倍率值格式化为**读数**文本（带 `dB` 后缀，供悬浮浮窗用）。
 *
 * 与 {@link formatDbLabel} 分开：刻度要短（56px 轴列），浮窗有空间写清楚"这是
 * dB 而不是倍率"，用户才不会把 −6 误读成倍率。
 *
 * @param ratio 线性幅值倍率。
 * @returns 读数文本（如 `-6 dB`）。
 */
export function formatDbReadout(ratio: number): string {
    return `${formatDbLabel(ratio)} dB`;
}
