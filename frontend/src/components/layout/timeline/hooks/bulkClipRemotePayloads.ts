export type BulkClipRemoteChange = {
    /**
     * **Item 层**增益（时间轴旋钮 / 行内编辑 / 双击复位）。
     *
     * 【为什么不再有裸 `gain`】它此前同时被两个层使用：时间轴的乐观更新写的是
     * `clip.clipGain`（item 层），而这里发出去的裸 `gain` 被后端读成 **take** 层。
     * 于是「旋钮拖到 +12 dB，再用行内编辑回 0 dB」之后，item 层停在 4.0、take 层变成
     * 1.0 —— 现场那对镜像读数（`gain=4.0 expected 1.0` 与 `gain=1.0 expected 4.0`）
     * 就是这么来的。字段名自己说明层，这类错位在写代码时就看得见。
     */
    itemGain?: number;
    /** **Take 层**增益（take 属性面板）。与 item 层是两个独立的旋钮。 */
    takeGain?: number;
    muted?: boolean;
    /** 尺寸编辑（多选裁短/延长/拉伸的原子批量提交；后端 ClipStatePatch 原生支持）。 */
    startSec?: number;
    lengthSec?: number;
    snapOffsetSec?: number;
    clipPlaybackRate?: number;
    fadeInSec?: number;
    fadeOutSec?: number;
    /** REAPER 形状 id / 曲率：拖拽期间的修改必须随最终提交一并落盘，
     *  否则批量 fulfilled 的整份时间线回灌会把拖拽期修改丢弃（回退/无效）。 */
    fadeInShape?: number;
    fadeInDir?: number;
    fadeOutShape?: number;
    fadeOutDir?: number;
    /** 自动交叉淡化（交叉点拖拽/粘贴后跟随）：与手动 fade 分离。 */
    autoFadeInSec?: number;
    autoFadeOutSec?: number;
    /** 倒放开关（后端 ClipStatePatch 支持，乐观更新同步应用）。 */
    reversed?: boolean;
    /** Loop（循环源）开关。 */
    loopEnabled?: boolean;
    /** 源窗口（派生窗口模型下随 Loop 开关等操作一并归一）。 */
    sourceStartSec?: number;
    sourceEndSec?: number;
    /** 声道模式 0..=4（对齐 REAPER CHANMODE）。 */
    channelMode?: number;
};

/** 时间轴旋钮 / 行内编辑 / 双击复位的**唯一**构造器（item 层）。
 *
 * 【为什么必须走构造器】载荷字段名与"改哪一层"的对应关系是本轮修的那类 bug 的全部
 * 内容。调用点直接写字面量时，`{ gain }` 与 `{ itemGain }` 在代码评审里长得几乎一样，
 * 而它们在宿主侧是两个不同的旋钮。
 */
export function itemGainChange(gain: number): BulkClipRemoteChange {
    return { itemGain: gain };
}

/** take 属性面板的**唯一**构造器（take 层）。 */
export function takeGainChange(gain: number): BulkClipRemoteChange {
    return { takeGain: gain };
}

export function buildBulkClipStateUpdates(args: {
    clipIds: string[];
    changesById: Map<string, BulkClipRemoteChange>;
}): Array<{ clipId: string } & BulkClipRemoteChange> {
    return args.clipIds.flatMap((clipId) => {
        const changes = args.changesById.get(clipId);
        if (!changes) return [];
        return [{ clipId, ...changes }];
    });
}

export function buildDuplicateClipsBulkPayload(args: {
    sourceClipIds: string[];
    deltaSec: number;
    copyLinkedParams: boolean;
    applyAutoCrossfade: boolean;
    trackMode: Record<string, unknown>;
    placeOnSelectedTrack?: boolean;
    renameCopies?: boolean;
}) {
    return {
        sourceClipIds: args.sourceClipIds,
        deltaSec: args.deltaSec,
        copyLinkedParams: args.copyLinkedParams,
        applyAutoCrossfade: args.applyAutoCrossfade,
        selectCreatedClips: true,
        trackMode: args.trackMode,
        ...(args.placeOnSelectedTrack ? { placeOnSelectedTrack: true } : {}),
        ...(args.renameCopies === false ? { renameCopies: false } : {}),
    };
}
