/**
 * 参数取数窗口的键（`v2|{trackId}|{paramId}|{startFrame}|{frameCount}|{stride}`）解析。
 *
 * ## 为什么只剩"解析"
 *
 * 这个模块此前叫 `liveEditWindow`，装着整套"live 覆盖按窗口下标寻址"的逻辑：写入、
 * 回滚、换窗口时的**重锚**（把用户画过的下标搬到新窗口）。那套逻辑的前提是"笔画的坐标
 * 是窗口下标"，而窗口是后端驱动的 —— 于是每次换窗口都得把笔画搬一次，搬不动的帧就被
 * 丢掉，重锚本身还成了"轨迹不画"的又一个来源。
 *
 * 笔画改成**绝对帧缓冲**（见 `strokeBuffer.ts`）之后，重锚不再存在：换窗口只需重新
 * 投影。这个模块因此只剩一件事 —— 从键里取出**作用域**，因为"换参数 / 换轨"与"换窗口"
 * 的语义仍然相反：
 *
 * - 作用域变了 → 合法换窗口，正在画的笔画本就该丢弃；
 * - 只有窗口参数变了 → 什么都不该发生。
 *
 * 键的格式由 `usePianoRollData.computeVisibleRequest` 生成，两侧必须同口径。
 */

export interface ParamWindowKey {
    /** `v2|{trackId}|{paramId}` —— 作用域；变了就是换参数 / 换轨。 */
    readonly scope: string;
    /** 窗口首帧。 */
    readonly startFrame: number;
    /** 窗口帧步长。 */
    readonly stride: number;
}

/**
 * 解析窗口键。
 *
 * 【为什么缺字段要按默认值走】键是字符串拼出来的，历史上存在过字段更少的版本；解析器对
 * 残缺输入必须给出**可用**的默认值而不是 NaN —— 一个 NaN 帧号会让整条曲线投影到画布外，
 * 看起来像"数据没了"，而真正的原因只是键格式变了。
 */
export function parseParamWindowKey(key: string): ParamWindowKey {
    const parts = key.split("|");
    return {
        // [v2, trackId, paramId, startFrame, frameCount, stride]
        scope: `${parts[0] ?? ""}|${parts[1] ?? ""}|${parts[2] ?? ""}`,
        startFrame: Number(parts[3]) || 0,
        stride: Number(parts[5]) || 1,
    };
}

/**
 * 取窗口键的作用域（`v2|trackId|paramId`）。
 *
 * 【为什么非 `v2|` 的键原样返回】历史键格式（以及测试里手写的键）没有前缀，
 * 拆出来的 "a||" 会让两条本来不同的键撞成同一个作用域 —— 那比"按原样比较"更糟。
 */
export function paramWindowScope(key: string): string {
    if (!key.startsWith("v2|")) return key;
    return parseParamWindowKey(key).scope;
}

/** 从作用域里取参数 id（`v2|trackId|paramId` 的最后一段）。 */
export function paramScopeParamId(scope: string): string {
    const separator = scope.lastIndexOf("|");
    return separator < 0 ? scope : scope.slice(separator + 1);
}
