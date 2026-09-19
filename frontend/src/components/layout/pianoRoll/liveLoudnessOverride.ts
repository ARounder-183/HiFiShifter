/**
 * 绘制中的 live 覆盖 → 波形幅度映射可消费的「响度曲线视图」。
 *
 * ## 为什么需要单独一个模块
 *
 * 参数编辑器波形的幅度映射是**延迟取值**的：几何层在为每一列构建包络时会反复
 * 询问当前 live 覆盖（编辑中的音量 / 动态曲线）。列内增益切片把询问次数提到
 * 每列最多 32 次 —— 2112 列的典型窗口下，**一次几何重建约 6.8 万次**。
 *
 * 天真的实现（每次询问都从 `key` 里 `split("|")` 解析窗口、再新建一个视图
 * 对象）在这条路径上要花掉 ~26ms/帧，正是用户报告的「编辑音量/动态时卡顿」
 * 的根因。本模块把它收敛为：
 *
 * - **解析按覆盖对象身份缓存**：live 覆盖每次更新都写新对象
 *   （`{ key, edit }`，见 `useLiveParamEditing` 的 `applyDenseToLiveEdit`），
 *   因此「对象换了」就是「该重解析」的充要条件，用 `===` 判定（零成本）；
 * - **视图对象复用**：稳态下每次询问只做一次身份比较，零分配、零字符串操作。
 *
 * ## 语义
 *
 * `live.edit` 与**发起编辑时**的 paramView 窗口对齐（起点与步长来自 key），
 * 而映射要按绝对帧采样，故两者都必须解析出来；编辑结束后本视图返回 null，
 * 调用方回退到整工程快照曲线。
 *
 * ## 与其他模块的关系
 * - key 的格式与参数归属规则收口在 `useLoudnessCurves`（`parseLiveOverrideKey`
 *   / `liveOverrideParamMatches`），本模块只负责"按身份缓存 + 惰性失效"。
 * - 消费方：`PianoRollPanel.readLiveOverrideFor` → `makeLoudnessAmplitudeMap`。
 */

import type { LoudnessLiveCurve } from "./PianoRollWaveformSurface";
import { liveOverrideParamMatches, parseLiveOverrideKey } from "./useLoudnessCurves";

/** live 覆盖实体（与 `useLiveParamEditing` 的 ref 同形）。 */
export interface LiveOverrideSource {
    key: string;
    edit: number[];
}

/** 一份 live 覆盖的读取器（缓存随覆盖对象的身份失效）。 */
export interface LiveOverrideReader {
    /**
     * 取指定参数的 live 曲线视图。
     *
     * @param param 目标参数（volume / dyn）。
     * @param live 当前 live 覆盖；null 表示没有进行中的编辑。
     * @returns 可交给幅度映射采样的曲线视图；该参数不在覆盖中 / 无覆盖时为 null。
     */
    read(param: "volume" | "dyn", live: LiveOverrideSource | null): LoudnessLiveCurve | null;
    /**
     * 该覆盖是否会影响**波形**（即它编辑的是 volume / dyn）。
     *
     * 【为什么需要】波形画的是「可听结果」`源峰值 × clip增益×淡化 × volume(t)
     * × dyn增益(t)`，与音高 / 共振峰 / 齿度等参数**无关**。面板据此决定要不要
     * 在绘制中强制重建波形几何：绘制音高时逐帧重建两千余列的包络纯属浪费。
     * 判定与 {@link read} 共用同一份解析缓存，调用成本与 `read` 相同。
     */
    affectsWaveform(live: LiveOverrideSource | null): boolean;
}

/** 一份覆盖的缓存条目（解析结果 + 两份按参数归属的视图）。 */
interface CacheEntry {
    /** 产生本缓存的覆盖对象（身份即版本）。 */
    source: LiveOverrideSource;
    /** key 中的参数 id（归属判定用，避免每次再 split）。 */
    paramId: string;
    /** 曲线视图（两份参数共用同一个对象：同一覆盖只有一条数据）。 */
    view: LoudnessLiveCurve;
}

/**
 * 创建 live 覆盖读取器（每个面板实例一个，跨帧复用其缓存）。
 *
 * @returns 读取器；`read` 在稳态下零分配。
 */
export function createLiveOverrideReader(): LiveOverrideReader {
    let cache: CacheEntry | null = null;

    /**
     * 取当前覆盖的缓存条目（缺失或身份变了就重解析）。
     *
     * @returns 缓存条目；无覆盖 / 空数组时为 null。
     */
    function entryFor(live: LiveOverrideSource | null): CacheEntry | null {
        if (live === null || live.edit.length === 0) {
            // 覆盖消失（编辑结束 / 空数组）：丢弃缓存，避免长期持有已结束
            // 编辑的数组引用（撤销 / 切轨后可能很久不再更新）。
            cache = null;
            return null;
        }
        if (cache === null || cache.source !== live) {
            // 对象身份变了（新一次更新 / 新一次编辑）→ 重解析。
            // 注意 `edit` 的**内容**是原地更新的，因此不能只比数组引用；
            // 覆盖对象每次更新都换新（见 useLiveParamEditing），身份即版本。
            const parts = parseLiveOverrideKey(live.key);
            cache = {
                source: live,
                paramId: parts.paramId,
                view: {
                    startFrame: parts.startFrame,
                    stride: parts.stride,
                    values: live.edit,
                },
            };
        }
        return cache;
    }

    return {
        read(param, live) {
            const entry = entryFor(live);
            if (entry === null) return null;
            // 参数归属仍按 key 判定：同一份覆盖只属于它编辑的那个参数。
            if (!liveOverrideParamMatches(entry.paramId, param)) return null;
            return entry.view;
        },
        affectsWaveform(live) {
            const entry = entryFor(live);
            if (entry === null) return false;
            return (
                liveOverrideParamMatches(entry.paramId, "volume") ||
                liveOverrideParamMatches(entry.paramId, "dyn")
            );
        },
    };
}
