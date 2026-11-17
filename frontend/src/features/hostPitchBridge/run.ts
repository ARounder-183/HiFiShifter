/**
 * 「REAPER 音高互转」的分批与结果汇总（纯逻辑，便于单测）。
 *
 * 【为什么要分批】一条命令最多带 512 个片段（后端 `pitch_bridge_clip_ids` 的硬上限），
 * 而"全部片段"在真实工程里轻松过千。分批本身不难，难的是**分批之后仍是一次操作**：
 * 撤销栈上必须只有一步、报告里必须只有一组数。前者由 `begin_undo_group` 收口
 * （见 `useHostPitchBridge`），后者就是本模块的汇总。
 *
 * 【为什么每批是 128 而不是 512】每条片段都要在 actor 线程上读一次宿主 chunk
 * （导出还要写回 + 让宿主重建 item）。512 条挤在一条命令里，actor 会在一次心跳里
 * 卡住几秒 —— 那期间参数线拖动、播放头轮询全部排队。128 条是"少几轮往返"与
 * "心跳不被卡住"之间的折中，分批的代价只有一次 IPC。
 */

/** 单条命令携带的片段数上限。 */
export const PITCH_BRIDGE_CHUNK = 128;

/** 后端两条命令的共同回执（字段按方向取舍，未用到的缺省）。 */
export interface PitchBridgeResult {
    readonly ok?: boolean;
    readonly clips?: number;
    /** 真的被转换的片段数（跳过的不计）。 */
    readonly clipsConverted?: number;
    readonly framesWritten?: number;
    readonly framesSkipped?: number;
    readonly pointsWritten?: number;
    /** 无法转换的片段 id（没有宿主 take / 没有音高数据 / 未知片段）。 */
    readonly unresolved?: readonly string[];
    readonly error?: string;
}

/** 一次「全部片段」操作的累计结果。 */
export interface PitchBridgeOutcome {
    /** 请求转换的片段数（去重后）。 */
    readonly requested: number;
    readonly converted: number;
    readonly framesWritten: number;
    readonly framesSkipped: number;
    readonly pointsWritten: number;
    readonly unresolved: readonly string[];
}

/** 把 id 列表切成每批至多 `PITCH_BRIDGE_CHUNK` 条。空输入得到空数组（调用方据此跳过）。 */
export function planPitchBridgeBatches(ids: readonly string[]): string[][] {
    const batches: string[][] = [];
    for (let index = 0; index < ids.length; index += PITCH_BRIDGE_CHUNK) {
        batches.push(ids.slice(index, index + PITCH_BRIDGE_CHUNK));
    }
    return batches;
}

/**
 * 去掉重复 id（保持首次出现的位置）。
 *
 * 【为什么必须去重】"当前轨道"与"选区内"在同一批命令里不会重叠，但作用域是**取交集
 * 之外**的三选一，而调用方可能把两个作用域合并后传进来。重复 id 会让同一条片段被写
 * 两次：导入方向是"叠加两次宿主音高"，导出方向是"包络写两遍"—— 前者直接错音。
 */
export function uniqueClipIds(ids: readonly string[]): string[] {
    const seen = new Set<string>();
    const out: string[] = [];
    for (const id of ids) {
        if (seen.has(id)) continue;
        seen.add(id);
        out.push(id);
    }
    return out;
}

export function emptyOutcome(requested: number): PitchBridgeOutcome {
    return {
        requested,
        converted: 0,
        framesWritten: 0,
        framesSkipped: 0,
        pointsWritten: 0,
        unresolved: [],
    };
}

/** 把一条命令的回执累加进总账（缺省字段按 0 处理：老后端不回这些字段）。 */
export function mergeOutcome(
    outcome: PitchBridgeOutcome,
    result: PitchBridgeResult,
): PitchBridgeOutcome {
    const count = (value: number | undefined): number =>
        typeof value === "number" && Number.isFinite(value) && value > 0 ? value : 0;
    return {
        requested: outcome.requested,
        converted: outcome.converted + count(result.clipsConverted),
        framesWritten: outcome.framesWritten + count(result.framesWritten),
        framesSkipped: outcome.framesSkipped + count(result.framesSkipped),
        pointsWritten: outcome.pointsWritten + count(result.pointsWritten),
        unresolved: [...outcome.unresolved, ...(result.unresolved ?? [])],
    };
}

/** 没有被转换的片段数（"跳过了几个"）。 */
export function skippedClips(outcome: PitchBridgeOutcome): number {
    return Math.max(0, outcome.requested - outcome.converted);
}
