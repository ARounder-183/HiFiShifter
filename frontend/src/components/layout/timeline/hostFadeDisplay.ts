// 宿主淡化只读显示：用户允许HiFiShifter自己的示意曲线；普通fade声音仍由宿主负责一次。
import type { HostFadeMetadata } from "../../../types/api";
import {
    defaultFadeDirFor,
    FADE_LINEAR,
    FADE_S_SHARP,
    FADE_S_SLIGHT,
    fadeGainSigned,
} from "./reaperFade";
import { hostFadePresetForAxes } from "./hostFadeAxes";

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

/**
 * 新轴宿主的曲线求值，**曲率与 S 由调用方给出**。
 *
 * 【为什么把它单独开出来】画布要按"当前轴值"画，而曲率拖拽要把指针投影到**候选**曲率
 * 上（`solveNearestCurveDir` 的 `gainAt`）。两者必须是同一条曲线，否则指针和画出来的
 * 包络对不上（"拖了不跟手"）。所以只有这一个求值入口，两处都走它。
 *
 * 落在实测表上的坐标画该预设自己的曲线；表外的坐标是"线性形状 + 曲率"与 S 族的**有界
 * 混合**（权重 |S|）—— 这是 HFS 自身的视觉定义，不是反推的 REAPER 公式。
 */
export function hostFadeGainForAxes(
    mode: "in" | "out",
    curvature: number,
    s: number,
    progress: number,
): number {
    const preset = hostFadePresetForAxes(curvature, s);
    if (preset !== null)
        return fadeGainSigned(preset, defaultFadeDirFor(preset, mode === "out"), mode, progress);
    const power = fadeGainSigned(0, curvature, mode, progress);
    if (s === 0) return power;
    const sigmoid = fadeGainSigned(5, visualAxis(curvature + s), mode, progress);
    return power + (sigmoid - power) * Math.abs(s);
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
    const isOut = mode === "out";
    // 曲线也按夹紧后的值画，与"认预设"用的是同一对数字，才不会出现"画的是夹紧后的
    // 曲线、认的却是另一个预设"。
    return hostFadeGainForAxes(
        mode,
        visualAxis(isOut ? metadata.out_curvature : metadata.in_curvature),
        visualAxis(isOut ? metadata.out_s : metadata.in_s),
        progress,
    );
}

/**
 * 界面该把这一侧显示成**哪个形状** —— 与画布画的是同一族曲线。
 *
 * 宿主新轴坐标**恰好**落在实测表上时（`hostFadePresetForAxes`）就是那个预设；不在表上
 * （用户拖过曲率滑杆）时没有可命名的预设，按渲染器实际用的权重报出占优的那一族：
 * `hostFadeGainForAxes` 画的是"线性形状 + 曲率"与 S 族的有界混合（权重 |S|），所以
 * |S| 过半报 S 族、否则报线性。**不**替用户认领一个具体预设 —— 实测表明宿主自己的
 * 粗分类会把 `(0.5, -1)` 认成 S 族（见 `hostFadeAxes.ts` 的说明）。
 *
 * 曲率行照常报 clip 的 `dir`：那正是曲率滑杆持有的、也是拖拽写回宿主轴的同一个数。
 */
export function hostFadeDisplayShape(
    metadata: HostFadeMetadata | undefined,
    isOut: boolean,
    fallbackShape: number,
): number {
    const normalized = Math.trunc(Number.isFinite(fallbackShape) ? fallbackShape : 0);
    const known =
        normalized >= FADE_LINEAR && normalized <= FADE_S_SHARP ? normalized : FADE_LINEAR;
    if (!metadata || metadata.curve_mode !== "reaper_new") return known;
    const preset = hostFadePresetForAxes(
        visualAxis(isOut ? metadata.out_curvature : metadata.in_curvature),
        visualAxis(isOut ? metadata.out_s : metadata.in_s),
    );
    if (preset !== null) return preset;
    const s = visualAxis(isOut ? metadata.out_s : metadata.in_s);
    return Math.abs(s) >= 0.5 ? FADE_S_SLIGHT : FADE_LINEAR;
}
