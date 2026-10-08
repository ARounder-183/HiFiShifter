// 宿主淡化只读显示：用户允许HiFiShifter自己的示意曲线；普通fade声音仍由宿主负责一次。
import type { HostFadeMetadata } from "../../../types/api";
import { defaultFadeDirFor, fadeGainSigned, SHAPE_LABEL_KEYS } from "./reaperFade";
import { hostFadePresetForAxes } from "./hostFadeAxes";
import { formatTemplate } from "../../../i18n/format";

/** 热路径复用同一函数，非法轴退零，不为每个波形采样创建闭包。 */
function visualAxis(value: number): number {
    return Number.isFinite(value) ? Math.min(1, Math.max(-1, value)) : 0;
}

/** 新轴采用本应用曲线风格，不宣称逐点复现REAPER；独立App及旧轴保持原路径。 */
export function hostFadeDisplay(
    metadata: HostFadeMetadata | undefined,
    _isOut: boolean,
): "legacy" | "hifishifter" {
    if (!metadata || metadata.curve_mode === "legacy") return "legacy";
    return "hifishifter";
}

/** 默认宿主渐变采用圆滑的四分之一正弦示意；只用于GUI，不修改宿主参数或PCM。 */
function defaultHostFadeGain(mode: "in" | "out", progress: number): number {
    if (progress === 0) return mode === "in" ? 0 : 1;
    if (progress === 1) return mode === "in" ? 1 : 0;
    return Math.sin(((mode === "in" ? progress : 1 - progress) * Math.PI) / 2);
}

/** UI求值器供曲线描边与波形共用；只画图，不把宿主fade烘焙到插件PCM。 */
export function visualFadeGain(
    metadata: HostFadeMetadata | undefined,
    shape: number,
    dir: number,
    mode: "in" | "out",
    t: number,
): number {
    if (!metadata || metadata.curve_mode === "legacy" || metadata.curve_mode === "hifishifter")
        return fadeGainSigned(shape, dir, mode, t);
    const progress = Number.isFinite(t) ? Math.min(1, Math.max(0, t)) : 0;
    if (metadata.curve_mode !== "reaper_new") return defaultHostFadeGain(mode, progress);
    const curvature = visualAxis(mode === "out" ? metadata.out_curvature : metadata.in_curvature);
    const s = visualAxis(mode === "out" ? metadata.out_s : metadata.in_s);
    // 落在预设上：用该预设自己的曲线（含它在该侧的重置曲率）。
    // 【为什么匹配用的是夹紧后的值】曲线也按夹紧后的值画，两边用同一对数字才不会
    // 出现"画的是夹紧后的曲线、认的却是另一个预设"。
    const preset = hostFadePresetForAxes(curvature, s);
    if (preset !== null)
        return fadeGainSigned(preset, defaultFadeDirFor(preset, mode === "out"), mode, progress);
    // 本应用既有幂曲率与S族做有界混合；这是HFS自身视觉定义，不是反推的REAPER公式。
    const power = fadeGainSigned(0, curvature, mode, progress);
    if (s === 0) return power;
    const sigmoid = fadeGainSigned(5, visualAxis(curvature + s), mode, progress);
    return power + (sigmoid - power) * Math.abs(s);
}

/**
 * 直接报告宿主两个原始轴；问号表示版本语义不可用，不沿用过时shape名称。
 *
 * 落在七个预设上时额外附上预设名（`c=0.50 S=0.00 (Fast Start)`）—— 那张映射是
 * 实测的（`hostFadeAxes.ts`），所以说得出名字；不在预设上就不猜，只报原始读数。
 *
 * `t` 只用于"轴不可用"这一种状态：`c=`/`S=` 是数值读数，语言中立，不进词表。
 * 品牌名 `HiFiShifter` 同理保持原样。
 */
export function hostFadeLabel(
    metadata: HostFadeMetadata,
    isOut: boolean,
    t: (key: string) => string,
): string {
    if (metadata.curve_mode === "hifishifter") return "HiFiShifter";
    if (metadata.curve_mode === "unknown") return t("fade_info_host_unknown");
    const curvature = isOut ? metadata.out_curvature : metadata.in_curvature;
    const s = isOut ? metadata.out_s : metadata.in_s;
    const reading = `REAPER c=${curvature.toFixed(2)} S=${s.toFixed(2)}`;
    const preset = hostFadePresetForAxes(curvature, s);
    if (preset === null) return reading;
    const name = t(SHAPE_LABEL_KEYS[preset] ?? "fade_shape_linear");
    return formatTemplate(t("common_parenthetical"), { value: reading, note: name });
}
