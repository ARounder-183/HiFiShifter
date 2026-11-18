/**
 * 「REAPER 音高互转」的三种**作用域**（纯逻辑，便于单测）。
 *
 * 【为什么作用域在前端算】"当前轨道"与"选区内"都是**界面概念**：轨道组来自参数编辑器
 * 正在编辑的那条根轨道，选区来自参数编辑器自己的帧选区。后端只认一份 id 列表 —— 让它
 * 去猜"当前"是哪一个，等于把同一件事的两份定义分开维护，而两份定义迟早会不一致。
 *
 * 【为什么是"与选区重叠"而不是"完整落在选区里"】用户框选的是**想看的那一段**，片段
 * 往往比选区长（或一头露在外面）。按"完整落在里面"取，选区只要没盖满整段就一个片段
 * 都不转，而屏幕上明明画着它们 —— 那不是精确，那是没反应。
 */

/** 作用域名字（与菜单里的三项一一对应）。 */
export type PitchBridgeScope = "track" | "selection" | "all";

/**
 * 作用域解析需要的片段事实。
 *
 * 【为什么不直接用 `ClipInfo`】本模块只做区间运算，结构化输入让测试不必造一份完整
 * 片段（那些字段与作用域无关，改了还会连带这条用例）。
 */
export interface PitchBridgeClipRef {
    readonly id: string;
    readonly trackId: string;
    readonly startSec: number;
    readonly lengthSec: number;
}

/** 三个作用域各自的片段 id（保持时间线顺序）。 */
export interface PitchBridgeScopes {
    readonly track: readonly string[];
    readonly selection: readonly string[];
    readonly all: readonly string[];
}

/** 半开区间 `[startSec, endSec)`。 */
export type PitchBridgeRangeSec = readonly [number, number];

/**
 * 片段是否与任一区间重叠。
 *
 * 非有限几何一律视为"不重叠"：一个 `startSec` 是 NaN 的片段无法判断落点，把它算进
 * 选区只会让"我明明只选了一小段"变成一次范围不明的写入。
 */
function overlaps(clip: PitchBridgeClipRef, ranges: readonly PitchBridgeRangeSec[]): boolean {
    const start = clip.startSec;
    const end = clip.startSec + clip.lengthSec;
    if (!Number.isFinite(start) || !Number.isFinite(end)) return false;
    for (const range of ranges) {
        const from = Math.min(range[0], range[1]);
        const to = Math.max(range[0], range[1]);
        if (!Number.isFinite(from) || !Number.isFinite(to)) continue;
        // 半开区间：端点相接不算重叠（上一个片段结束处正是选区起点时，它不在选区里）。
        if (end > from && start < to) return true;
    }
    return false;
}

/**
 * 解析三种作用域。
 *
 * @param args.clips    时间线上的全部片段（顺序即输出的顺序）。
 * @param args.trackIds 当前根轨道**组**内的轨道 id（根轨道 + 它的子轨道）。
 * @param args.rangesSec 参数选区的帧区间换算成的秒区间；空数组 = 当前没有选区。
 */
export function resolvePitchBridgeScopes(args: {
    readonly clips: readonly PitchBridgeClipRef[];
    readonly trackIds: ReadonlySet<string>;
    readonly rangesSec: readonly PitchBridgeRangeSec[];
}): PitchBridgeScopes {
    const { clips, trackIds, rangesSec } = args;
    return {
        track: clips.filter((clip) => trackIds.has(clip.trackId)).map((clip) => clip.id),
        selection: clips.filter((clip) => overlaps(clip, rangesSec)).map((clip) => clip.id),
        all: clips.map((clip) => clip.id),
    };
}

/** 某个作用域的 id 列表（菜单渲染与执行共用这一个入口）。 */
export function scopeClipIds(
    scopes: PitchBridgeScopes,
    scope: PitchBridgeScope,
): readonly string[] {
    return scopes[scope];
}
