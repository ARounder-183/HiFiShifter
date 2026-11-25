/**
 * 绘制中的 live 覆盖 → 波形幅度映射可消费的「响度曲线视图」。
 *
 * ## 为什么需要单独一个模块
 *
 * 参数编辑器波形的幅度映射是**延迟取值**的：几何层在为每一列构建包络时会反复
 * 询问当前 live 覆盖（编辑中的音量 / 动态曲线）。列内增益切片把询问次数提到
 * 每列最多 32 次 —— 2112 列的典型窗口下，**一次几何重建约 6.8 万次**。
 *
 * 天真的实现（每次询问都重新投影、再新建一个视图对象）在这条路径上要花掉
 * ~26ms/帧，正是用户报告的「编辑音量/动态时卡顿」的根因。本模块把它收敛为：
 *
 * - **投影按 (覆盖对象, 窗口, 缓冲版本) 缓存**：三者都没变时直接返回上一次的数组
 *   （`projectLiveEdit` 内部已按同样口径缓存，这里再比一次**数组身份**即可，
 *   零分配、零字符串操作）；
 * - **视图对象复用**：稳态下每次询问只做几次身份比较；
 * - **视图持有投影数组本身**，因此缓冲的原地更新对采样**立即可见** —— 波形不会停在
 *   拖动起点。
 *
 * ## 语义
 *
 * 投影数组与**当前参数窗口**对齐（起点与步长取自窗口），而映射要按绝对帧采样，故两者
 * 都必须解析出来。编辑结束后本视图返回 null，调用方回退到整工程快照曲线。
 *
 * ## 与其他模块的关系
 * - 参数归属规则（volume 字面量、dyn 的历史别名）收口在 `useLoudnessCurves`
 *   （`liveOverrideParamMatches`），本模块只负责"按身份缓存 + 惰性失效"。
 * - 投影本身收口在 `strokeBuffer`（绝对帧缓冲），本模块不碰坐标换算。
 * - 消费方：`PianoRollPanel.readLiveOverrideFor` → `makeLoudnessAmplitudeMap`。
 */

import type { LoudnessLiveCurve } from "./PianoRollWaveformSurface";
import { paramScopeParamId } from "./paramWindowKey";
import { projectLiveEdit } from "./strokeBuffer";
import type { LiveEditOverride, LiveProjectionWindow } from "./strokeBuffer";
import { liveOverrideParamMatches } from "./useLoudnessCurves";

/** live 覆盖实体（与 `useLiveParamEditing` 的 ref 同形）。 */
export type LiveOverrideSource = LiveEditOverride;

/** 一份 live 覆盖的读取器（缓存随覆盖对象 / 窗口 / 缓冲版本失效）。 */
export interface LiveOverrideReader {
    /**
     * 取指定参数的 live 曲线视图。
     *
     * @param param 目标参数（volume / dyn）。
     * @param live 当前 live 覆盖；null 表示没有进行中的编辑。
     * @param window 当前参数窗口（投影目标）；null 表示还没有窗口。
     * @returns 可交给幅度映射采样的曲线视图；该参数不在覆盖中 / 无覆盖时为 null。
     */
    read(
        param: "volume" | "dyn",
        live: LiveOverrideSource | null,
        window: LiveProjectionWindow | null,
    ): LoudnessLiveCurve | null;
    /**
     * 该覆盖是否会影响**波形**（即它编辑的是 volume / dyn）。
     *
     * 【为什么需要】波形画的是「可听结果」`源峰值 × clip增益×淡化 × volume(t) × dyn增益(t)`，
     * 与音高 / 共振峰 / 齿度等参数**无关**。面板据此决定要不要在绘制中强制重建波形几何：
     * 绘制音高时逐帧重建两千余列的包络纯属浪费。
     */
    affectsWaveform(live: LiveOverrideSource | null): boolean;
}

/** 一份覆盖的缓存条目。 */
interface CacheEntry {
    source: LiveOverrideSource;
    window: LiveProjectionWindow;
    /** 投影数组（身份即版本，见模块头）。 */
    values: readonly number[];
    /** 交给调用方的视图对象（同一份数据，复用同一个对象）。 */
    view: LoudnessLiveCurve;
}

/**
 * 创建 live 覆盖读取器（每个面板实例一个，跨帧复用其缓存）。
 *
 * @returns 读取器；`read` 在稳态下零分配。
 */
export function createLiveOverrideReader(): LiveOverrideReader {
    let cache: CacheEntry | null = null;

    /** 取当前覆盖的缓存条目（缺失或身份 / 版本变了就重新投影）。 */
    function entryFor(
        live: LiveOverrideSource | null,
        window: LiveProjectionWindow | null,
    ): CacheEntry | null {
        if (live === null || window === null || live.buffer.frameCount === 0) {
            // 覆盖消失（编辑结束 / 空缓冲）：丢弃缓存，避免长期持有已结束编辑的数组引用。
            cache = null;
            return null;
        }
        const values = projectLiveEdit(live, window);
        if (
            cache !== null &&
            cache.source === live &&
            cache.window === window &&
            cache.values === values
        ) {
            return cache;
        }
        cache = {
            source: live,
            window,
            values,
            view: {
                startFrame: window.startFrame,
                stride: window.stride,
                values,
            },
        };
        return cache;
    }

    return {
        read(param, live, window) {
            const entry = entryFor(live, window);
            if (entry === null) return null;
            // 参数归属按作用域判定：同一份覆盖只属于它编辑的那个参数。
            if (!liveOverrideParamMatches(paramScopeParamId(entry.source.scope), param)) {
                return null;
            }
            return entry.view;
        },
        affectsWaveform(live) {
            if (live === null || live.buffer.frameCount === 0) return false;
            const paramId = paramScopeParamId(live.scope);
            return (
                liveOverrideParamMatches(paramId, "volume") ||
                liveOverrideParamMatches(paramId, "dyn")
            );
        },
    };
}
