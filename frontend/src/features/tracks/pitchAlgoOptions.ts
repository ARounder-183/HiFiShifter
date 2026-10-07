/**
 * 音高分析算法的**唯一**选项来源。
 *
 * 【为什么必须收成一处】这份列表此前在 `TrackList.tsx`（轨道头下拉）与
 * `PianoRollPanel.tsx`（参数编辑器头部下拉）各写了一遍字面量数组 ——
 * 标签或顺序一旦调整就会静默漂移，且"哪些算法可用"这类能力信息没有任何
 * 地方可以挂。同类的漂移问题在 `ui/useMenuShortcut.ts` 的文件头已有记录
 * （三份快捷键文案实现各自漂移）。
 *
 * 这里只负责「列表长什么样 + 谁该出现」；**标签**由调用方注入（前三个是
 * 专有名词，不翻译；`none` 走 i18n）。
 */

/** 算法 id。顺序即下拉框展示顺序。 */
export const PITCH_ALGO_IDS = ["nsf_hifigan_onnx", "world_dll", "vslib", "none"] as const;

export type PitchAlgoId = (typeof PITCH_ALGO_IDS)[number];

/** 专有名词标签：不翻译，与后端 algo 字符串一一对应。 */
const FIXED_LABELS: Record<Exclude<PitchAlgoId, "none">, string> = {
    nsf_hifigan_onnx: "nsf-hifigan",
    world_dll: "world",
    vslib: "vslib",
};

export interface PitchAlgoOption {
    value: PitchAlgoId;
    label: string;
}

/**
 * 构建算法下拉的选项列表。
 *
 * 【vslib 的隐藏规则】
 * - `vslibAvailable === true` → 正常展示；
 * - 否则（`false`，或状态尚未取到的 `null`）→ 隐藏。
 *
 * 未知态取"隐藏"是刻意的：两个方向都会有短暂的闪烁，但"短暂不显示一个
 * 用不了的选项"比"短暂显示一个用不了的选项"安全 —— 后者最坏是用户已经
 * 点下去了，然后得到静默回退到别的算法的结果。
 *
 * 【为什么当前值即使不可用也要保留】下拉框的 `value` 若不在 `options` 里，
 * 调用方会回退显示 `nsf_hifigan_onnx` —— 那是在**谎报**轨道的真实算法
 * （轨道实际是 vslib、渲染走回退，界面却写着 nsf-hifigan）。保留并标注
 * `（不可用）` 同时解释了"它为什么在这个列表里"与"它为什么不工作"。
 *
 * @param args.noneLabel "无" 的本地化文案。
 * @param args.vslibAvailable vslib 可用性；`null` = 尚未取到后端状态。
 * @param args.formatUnavailable 把算法名包装成"不可用"标注的本地化回调
 *   （词典模板 `algo_unavailable_label`，如 `"{name}（不可用）"`）。
 * @param args.currentValue 当前轨道的算法值。
 */
export function buildPitchAlgoOptions(args: {
    noneLabel: string;
    vslibAvailable: boolean | null;
    formatUnavailable: (label: string) => string;
    currentValue?: string;
}): PitchAlgoOption[] {
    const vslibUsable = args.vslibAvailable === true;
    const keepUnavailableVslib = !vslibUsable && args.currentValue === "vslib";

    const options: PitchAlgoOption[] = [];
    for (const id of PITCH_ALGO_IDS) {
        if (id === "vslib" && !vslibUsable && !keepUnavailableVslib) continue;
        if (id === "none") {
            options.push({ value: id, label: args.noneLabel });
            continue;
        }
        const label = FIXED_LABELS[id];
        options.push({
            value: id,
            label:
                id === "vslib" && !vslibUsable ? args.formatUnavailable(label) : label,
        });
    }
    return options;
}

/**
 * 下拉框的 `value`：把任意后端/工程字符串收敛到本次可展示的选项上。
 *
 * - 值在选项内 → 原样；
 * - 值不在选项内（典型：工程里存着 vslib 但本构建不可用）→ 若被
 *   `buildPitchAlgoOptions` 以"当前值"身份保留，则仍原样返回；
 * - 其余未知值 → 回退到 `nsf_hifigan_onnx`（与 `resolveEffectivePitchAlgo`
 *   以及后端的回退规则一致）。
 */
export function resolvePitchAlgoSelectValue(
    raw: string | undefined,
    options: ReadonlyArray<PitchAlgoOption>,
): PitchAlgoId {
    if (raw && options.some((option) => option.value === raw)) return raw as PitchAlgoId;
    return "nsf_hifigan_onnx";
}

/**
 * 把任意后端/工程字符串收敛到**本构建实际执行的算法**。
 *
 * 与后端 `PitchAnalysisAlgo::effective()` 同一规则：未识别的字符串
 * （`"unknown"`，或更早/更新的版本写入、本构建不认识的算法名）回退到工程默认
 * 算法 `nsf_hifigan_onnx` —— 后端就是这么渲染的。任何"按算法分支"的前端逻辑
 * 都必须先过这里，否则会出现"参数集按 nsf-hifigan 取、排布却按 world 排"
 * 这类界面与声音对不上的问题。
 *
 * 【与 `resolvePitchAlgoSelectValue` 的分工】那个是**下拉框取值**，还要考虑
 * 选项里有没有 vslib（不可用时以"当前值"身份保留）；这个只回答"实际跑哪个
 * 算法"，不看可用性。
 */
export function resolveEffectivePitchAlgo(raw: string | undefined | null): PitchAlgoId {
    return raw && (PITCH_ALGO_IDS as readonly string[]).includes(raw)
        ? (raw as PitchAlgoId)
        : "nsf_hifigan_onnx";
}
