/**
 * 参数编辑器「音节块」的纯派生模型（第一版 mock）。
 *
 * 【为什么块不进工程文件】音节边界最终应由音高分析 / 分词结果给出；第一版先在
 * 前端从现有 Clip + MIDI 音符 / 曲线推导，保证工具、显示、多选、合并这些交互
 * 可以先被真实使用和验证。用户手动合并的选择保存在会话内的 `ParamBlockState`，
 * 不复制参数数组，也不改写 Clip 几何。
 *
 * 【坐标】一律用**工程秒**。参数曲线本身是帧域，但这里只做“这一块覆盖哪段时间”，
 * 绘制时再经 `TimelineAxis` 投影；块高则由当前参数曲线在成员范围内的均值求值。
 *
 * 【作用域】只在当前根轨道组内建块。跨组块没有唯一参数所有者，第一版直接拒绝。
 */

export interface ParamBlockSourceClip {
    readonly id: string;
    readonly trackId: string;
    readonly startSec: number;
    readonly lengthSec: number;
    /** 源窗口与播放速率、方向、循环；缺省为 0 起点、1 倍速、正放、不循环。 */
    readonly sourceStartSec?: number;
    readonly sourceEndSec?: number;
    readonly playbackRate?: number;
    readonly reversed?: boolean;
    readonly loopEnabled?: boolean;
    /** 内容时长（源域秒）；只有 MIDI Clip 的 Loop 周期需要它。 */
    readonly contentDurationSec?: number | null;
    readonly midiNoteData?: ReadonlyArray<{
        readonly startSec: number;
        readonly endSec: number;
        readonly note: number;
    }>;
    /** 可选 MIDI 音符：存在时直接作为音节边界（**工程秒**）。 */
    readonly syllableSpansSec?: ReadonlyArray<{
        readonly startSec: number;
        readonly endSec: number;
    }>;
    /** 后端分析结果的源域音节区间；由 `projectSourceSpansToTimeline` 投影。 */
    readonly sourceSyllableSpansSec?: ReadonlyArray<readonly [number, number]>;
    /** active take id；音节 id 用它保持长时间稳定引用。 */
    readonly takeId?: string;
    /** 为 true 时，无显式音节也按固定时长切成 mock 音节块。 */
    readonly mockSyllables?: boolean;
}

/**
 * MIDI 音符（**源域**秒）→ 工程秒上的音节块时间区间。
 *
 * Note `startSec` / `endSec` 是源文件坐标，不是相对 Clip 起点。之前的
 * `clip.startSec + note.startSec` 只在 `sourceStart=0`、速率 1、正放、
 * 非 Loop 时成立；裁切 / 拉伸 / 倒放 / Loop 都会整体偏时。
 *
 * 这里复用时间轴音高曲线的同一套窗口数学，只把“源域音符”换成“时间线段”，
 * 不另写第二份；Loop 音符可能跨周期，按每个周期投影后再合并相邻区间。
 */
export function projectMidiNoteSpansToTimeline(clip: ParamBlockSourceClip): Array<{
    startSec: number;
    endSec: number;
}> {
    const notes = clip.midiNoteData ?? [];
    if (notes.length === 0) return [];
    const clipStartSec = finite(clip.startSec, 0);
    const lengthSec = Math.max(0, finite(clip.lengthSec, 0));
    if (!(lengthSec > 1e-9)) return [];
    const rate =
        Number.isFinite(clip.playbackRate) && (clip.playbackRate as number) > 1e-6
            ? (clip.playbackRate as number)
            : 1;
    const reversed = clip.reversed === true;
    const sourceStartSec = finite(clip.sourceStartSec ?? 0, 0);
    const sourceEndRaw = Number.isFinite(clip.sourceEndSec)
        ? (clip.sourceEndSec as number)
        : Math.max(0, ...notes.map((note) => finite(note.endSec, 0)));
    /*
     * 非 Loop 的可见窗口结束 = 显式 sourceEnd，缺省才按 length×rate 推导。
     * 不能无条件用推导值：音频 Clip 的 sourceEnd 是权威媒体窗口，用它参与
     * 裁切才能与后端逐帧映射 / 音高线投影一致。
     */
    const srcEnd =
        clip.loopEnabled === true
            ? sourceEndRaw
            : reversed
              ? sourceEndRaw
              : Number.isFinite(clip.sourceEndSec)
                ? Math.min(sourceEndRaw, sourceStartSec + lengthSec * rate)
                : sourceStartSec + lengthSec * rate;
    const winStartSec =
        clip.loopEnabled === true
            ? sourceStartSec
            : reversed
              ? srcEnd - lengthSec * rate
              : sourceStartSec;
    const srcTotalLen = Math.max(0, srcEnd - winStartSec);
    if (!(srcTotalLen > 1e-9)) return [];
    const cycleSec =
        clip.loopEnabled === true
            ? Math.max(1e-9, Math.abs(finite(clip.contentDurationSec ?? 0, 0)) || srcTotalLen)
            : null;
    const spans: Array<{ startSec: number; endSec: number }> = [];
    for (const note of notes) {
        const noteStart = finite(note.startSec, 0);
        const noteEnd = finite(note.endSec, 0);
        if (!(noteEnd > noteStart)) continue;
        if (cycleSec !== null) {
            const anchor = reversed ? srcEnd : sourceStartSec;
            const offset = reversed
                ? modEuclid(anchor - noteEnd, cycleSec)
                : modEuclid(noteStart - anchor, cycleSec);
            const length = Math.min(lengthSec, (noteEnd - noteStart) / rate);
            const cycleTimeline = cycleSec / rate;
            for (
                let startLocal = offset / rate;
                startLocal < lengthSec - 1e-9;
                startLocal += cycleTimeline
            ) {
                const endLocal = Math.min(lengthSec, startLocal + length);
                if (endLocal > startLocal + 1e-9) {
                    spans.push({
                        startSec: clipStartSec + startLocal,
                        endSec: clipStartSec + endLocal,
                    });
                }
            }
            continue;
        }
        if (noteEnd <= winStartSec || noteStart >= srcEnd) continue;
        const relStart = Math.max(0, noteStart - winStartSec);
        const relEnd = Math.min(srcTotalLen, noteEnd - winStartSec);
        if (!(relEnd > relStart)) continue;
        const [effStart, effEnd] = reversed
            ? [Math.max(0, srcTotalLen - relEnd), Math.min(srcTotalLen, srcTotalLen - relStart)]
            : [relStart, relEnd];
        if (!(effEnd > effStart)) continue;
        const startLocal = Math.max(0, effStart / rate);
        const endLocal = Math.min(lengthSec, effEnd / rate);
        if (endLocal > startLocal + 1e-9) {
            spans.push({ startSec: clipStartSec + startLocal, endSec: clipStartSec + endLocal });
        }
    }
    return mergeSortedSpans(spans);
}

/** 源域区间 [start, end] → 工程秒；与 MIDI 投影共用窗口数学。 */
export function projectSourceSpansToTimeline(clip: ParamBlockSourceClip): Array<{
    startSec: number;
    endSec: number;
}> {
    const spans = clip.sourceSyllableSpansSec ?? [];
    if (spans.length === 0) return [];
    /*
     * 窗口写成闭区间显式值，避免调用方传 `endSec=0` 的旧语义（"到源末尾"）
     * 被投影当成空窗口。这里只关系到可见裁剪，不改变源域音节本身。
     */
    const base: ParamBlockSourceClip = {
        ...clip,
        // 音频 Clip 的可见窗口是 `[0, lengthSec)`；未显式给边界时按它推导，
        // 避免在测试 / 旧载荷里把 `sourceStartSec` 缺省成 0 后错误重复循环。
        sourceStartSec: Math.max(0, finite(clip.sourceStartSec ?? 0, 0)),
        sourceEndSec: (() => {
            const start = Math.max(0, finite(clip.sourceStartSec ?? 0, 0));
            const rate =
                Number.isFinite(clip.playbackRate) && (clip.playbackRate as number) > 1e-6
                    ? (clip.playbackRate as number)
                    : 1;
            const explicit = Number.isFinite(clip.sourceEndSec)
                ? Math.max(0, clip.sourceEndSec as number)
                : 0;
            const needed = start + Math.max(0, finite(clip.lengthSec, 0)) * rate;
            return explicit > start + 1e-9 ? Math.max(explicit, needed) : needed;
        })(),
    };
    /*
     * 【不能批量投影后 merge】第 N 个源域音节必须对应第 N 个结果区间。
     * `projectMidiNoteSpansToTimeline` 会合并首尾相接的区间（正确的显示语义），
     * 但那样会让后续 `sourceSyllableSpansSec[index]` 与块 index 错位。
     * 这里逐段投影，保留一一对应；连续段的显示合并交给 UI 层。
     */
    return spans
        .map(
            ([startSec, endSec]) =>
                projectMidiNoteSpansToTimeline({
                    ...base,
                    midiNoteData: [{ startSec, endSec, note: 60 }],
                })[0] ?? { startSec: NaN, endSec: NaN },
        )
        .filter((span) => Number.isFinite(span.startSec) && Number.isFinite(span.endSec));
}

function mergeSortedSpans(
    spans: Array<{ startSec: number; endSec: number }>,
): Array<{ startSec: number; endSec: number }> {
    const sorted = spans.slice().sort((a, b) => a.startSec - b.startSec);
    const merged: Array<{ startSec: number; endSec: number }> = [];
    for (const span of sorted) {
        const previous = merged[merged.length - 1];
        if (previous !== undefined && span.startSec <= previous.endSec + 1e-9) {
            previous.endSec = Math.max(previous.endSec, span.endSec);
        } else {
            merged.push({ ...span });
        }
    }
    return merged;
}

function modEuclid(value: number, modulus: number): number {
    return ((value % modulus) + modulus) % modulus;
}

export interface ParamBlockMember {
    readonly clipId: string;
    readonly takeId?: string;
    readonly syllableId?: string;
    /** 该成员对应的源域区间；持久化合并使用它校核，不用工程秒。 */
    readonly sourceStartSec?: number;
    readonly sourceEndSec?: number;
    /** 没有显式源域区间时（例如旧来源），由 Clip 映射得到此源域位置。 */
    readonly sourceStartForMatchSec?: number;
    readonly sourceEndForMatchSec?: number;
    readonly startSec: number;
    readonly endSec: number;
}

export interface ParamBlock {
    readonly id: string;
    readonly rootTrackId: string;
    readonly kind: "auto" | "merged";
    readonly members: readonly ParamBlockMember[];
    readonly startSec: number;
    readonly endSec: number;
    /**
     * 块的代表参数值（**密度众数中心**，不是算术均值）。
     *
     * 用块内出现最密集的参数区间代表块的高度位置；颤音、短滑音与少量离群点
     * 不会被拉平。字段名保留 `averageValue` 以兼容既有宿主 / 测试载荷。
     */
    readonly averageValue: number | null;
}

/**
 * 参数块的可见矩形。
 *
 * 块高只由当前参数的一行像素高决定（1.5 行高），代表参数值决定纵向中心。
 * `topPx` 是绘制与命中的唯一纵向几何，避免两处各用中心减半高而漂移。
 */
export interface ParamBlockRect {
    readonly id: string;
    readonly startSec: number;
    readonly endSec: number;
    /** 在本轨道组内的显示序号（从 1 起）；标签只展示这个，不伪造未取样的均值。 */
    readonly ordinal: number;
    /** 是否在已取数窗口内取样；未取样的块不画数值标签。 */
    readonly sampled: boolean;
    /** 块顶边（CSS px，视口坐标）；绘制 / 命中共用。 */
    readonly topPx: number;
    /** 块高（CSS px，视口坐标）。 */
    readonly heightPx: number;
    /** 与像素同源的块代表值（供拖动锚点直接读取）。 */
    readonly averageValue: number | null;
    readonly selected: boolean;
    readonly merged: boolean;
}

export interface MergedParamBlock {
    readonly id: string;
    /**
     * 被合并的自动块 id（音节级），不是 Clip id。
     *
     * 【为什么必须是音节级】同一 Clip 内的多个音节块也应能合并；按 Clip id 记录会
     * 在 materialize 时把整块 Clip 一并吞掉，视觉上“合并了两个音节，剩下全消失”。
     * 旧会话记录里的 `clipIds` 仍按 Clip 粒度兼容（见 materialize）。
     */
    readonly blockIds?: readonly string[];
    readonly clipIds?: readonly string[];
}

export interface ParamBlockState {
    readonly mergedBlocks: readonly MergedParamBlock[];
    /**
     * 工程持久化的合并记录（v7）：成员用 Clip / Take / 音节稳定引用，
     * 不随自动数组序号变化。与旧会话 `mergedBlocks` 并存，派生时二者共同生效。
     */
    readonly persistedMerges?: readonly PersistedParamBlockMerge[];
    /**
     * 兼容旧设置项：重叠块现在**无条件**合并，此字段不再参与派生。
     * 保留字段只为旧会话 / 设置载荷可继续解析。
     */
    readonly mergeOverlapsActive?: boolean;
}

export const EMPTY_PARAM_BLOCK_STATE: ParamBlockState = {
    mergedBlocks: [],
    persistedMerges: [],
    mergeOverlapsActive: false,
};

export interface PersistedParamBlockMerge {
    readonly id: string;
    readonly members: readonly {
        readonly clipId: string;
        readonly takeId: string;
        readonly syllableId: string;
        readonly sourceStartSec: number;
        readonly sourceEndSec: number;
    }[];
}

export type BlockMergeResult =
    | { readonly ok: true; readonly state: ParamBlockState }
    | { readonly ok: false; readonly reason: ParamBlockMergeFailureReason };

export type ParamBlockMergeFailureReason =
    "empty" | "different_root" | "not_adjacent" | "already_merged" | "nothing_to_merge";

/** 合并失败原因 → i18n key；交互层用它给出明确反馈。 */
export type ParamBlockMergeMessageKey =
    | "param_blocks_merge_need_two"
    | "param_blocks_merge_not_adjacent"
    | "param_blocks_merge_cross_root"
    | "param_blocks_merge_already"
    | "param_blocks_merge_none";

export function paramBlockMergeMessageKey(
    reason: ParamBlockMergeFailureReason,
): ParamBlockMergeMessageKey {
    switch (reason) {
        case "empty":
            return "param_blocks_merge_need_two";
        case "not_adjacent":
            return "param_blocks_merge_not_adjacent";
        case "different_root":
            return "param_blocks_merge_cross_root";
        case "already_merged":
            return "param_blocks_merge_already";
        case "nothing_to_merge":
            return "param_blocks_merge_none";
    }
}

/** 自动块的最小宽度（秒）：避免极碎的音节在画面上变成一堆看不到的细条。 */
export const MIN_AUTO_BLOCK_SEC = 0.02;
/** mock 音节块的默认时长（秒）：足够宽到能点中、又能看到“多个相邻块”。 */
export const MOCK_SYLLABLE_SEC = 0.55;
/** 块高 = 半音行高（键盘键高）的倍数。纵向缩放时跟着键高一起缩放。 */
export const PARAM_BLOCK_ROW_HEIGHT_MULTIPLIER = 1.5;

function finite(value: number, fallback: number): number {
    return Number.isFinite(value) ? value : fallback;
}

function clipSpan(clip: ParamBlockSourceClip): ParamBlockMember {
    const startSec = Math.max(0, finite(clip.startSec, 0));
    const endSec = Math.max(startSec, startSec + Math.max(0, finite(clip.lengthSec, 0)));
    return { clipId: clip.id, startSec, endSec };
}

/**
 * 由 Clip 派生自动音节块。
 *
 * 优先级：
 * 1. 显式 `syllableSpansSec`（mock / 未来分析结果）；
 * 2. 无显式音节时，每个 Clip 作为一个块（保守，不凭曲线零散跳变乱切）。
 *
 * 显式音节会先裁剪到 Clip 范围、去掉零宽段；全无效时退回整块。
 */
export function autoBlocksForClips(
    rootTrackId: string,
    clips: readonly ParamBlockSourceClip[],
): ParamBlock[] {
    const blocks: ParamBlock[] = [];
    for (const clip of clips) {
        if (clip.trackId.length === 0) continue;
        const member = clipSpan(clip);
        if (!(member.endSec > member.startSec)) continue;
        const spans: Array<{ startSec: number; endSec: number }> = [];
        for (const span of clip.syllableSpansSec ?? []) {
            // `syllableSpansSec` 是**工程秒**（MIDI / 已投影区间）；不能在这里
            // 再套一遍 source window。源域区间走 `sourceSyllableSpansSec`。
            const rawStartSec = finite(span.startSec, member.startSec);
            const rawEndSec = finite(span.endSec, rawStartSec);
            const clipped = {
                startSec: Math.max(rawStartSec, member.startSec),
                endSec: Math.min(rawEndSec, member.endSec),
            };
            if (clipped.endSec - clipped.startSec < MIN_AUTO_BLOCK_SEC) continue;
            const previous = spans[spans.length - 1];
            /*
             * 只合并**真正重叠**的音节。首尾相接（0.5 ~ 0.5）是正常音节边界，
             * 合并它们会把整句压成一个块；重叠才是重复检测 / 同音延音。
             */
            if (previous !== undefined && clipped.startSec < previous.endSec - 1e-9) {
                previous.endSec = Math.max(previous.endSec, clipped.endSec);
            } else {
                spans.push(clipped);
            }
        }
        const effective =
            spans.length > 0
                ? spans
                : clip.mockSyllables === true
                  ? mockSyllableSpans(member)
                  : [member];

        /*
         * 【源域数组存在时，工程秒只用于显示】`syllableSpansSec` 是已投影结果，
         * 可能被视觉层的"首尾相接合并"压成更少的区间；不能再按它的下标去取
         * `sourceSyllableSpansSec`。用源域数组自投射得到一一对应的工程区间，
         * 这是块身份与持久化合并的共同基准。
         */
        const projectedFromSource =
            (clip.sourceSyllableSpansSec?.length ?? 0) > 0
                ? projectSourceSpansToTimeline(clip)
                : [];
        const effectiveSpans =
            projectedFromSource.length > 0
                ? projectedFromSource.map((span, sourceIndex) => ({
                      ...span,
                      // 逐段投影保留一一对应；循环内不再二次投影求下标。
                      sourceIndex,
                  }))
                : effective.map((span, index) => ({ ...span, sourceIndex: index }));
        effectiveSpans.sort((a, b) => a.startSec - b.startSec);
        for (let index = 0; index < effectiveSpans.length; index += 1) {
            const span = effectiveSpans[index];
            const sourceSpan =
                clip.sourceSyllableSpansSec?.[span.sourceIndex >= 0 ? span.sourceIndex : index];
            const hasSourceSpan =
                clip.sourceSyllableSpansSec !== undefined && clip.sourceSyllableSpansSec.length > 0;
            /*
             * 纯测试 / 旧来源只给了工程秒 `syllableSpansSec`（没有源域数组）时，
             * 仍按工程秒校核；真实音频链路总会同时提供 `sourceSyllableSpansSec`。
             * 这样旧载荷可恢复，而真实 Clip 移动不会把工程秒误当源域身份。
             */
            // `span` 已是工程秒；没有源域数组时直接用它校核。
            const fallbackStart = span.startSec;
            const fallbackEnd = span.endSec;
            /* 工程秒只用于稳定 id 的兜底；源域数组存在时，index 必须对齐
             * 原始音节，而不是 `effective`（可能已裁剪 / 合并）。 */
            const sourceStart = hasSourceSpan ? (sourceSpan?.[0] ?? fallbackStart) : fallbackStart;
            const sourceEnd = hasSourceSpan ? (sourceSpan?.[1] ?? fallbackEnd) : fallbackEnd;
            /* 工程秒不能当源域身份：Clip 移动 / 裁切后它就变了。只有
             * `sourceSyllableSpansSec` 是权威源域输入；缺失时不做持久化匹配。 */
            const sourceStartForMatchSec = sourceStart;
            const sourceEndForMatchSec = sourceEnd;
            const memberSyllableId = `${clip.takeId ?? clip.id}:${sourceStart.toFixed(3)}:${sourceEnd.toFixed(3)}`;
            blocks.push({
                id: `auto:${clip.id}:${index}`,
                rootTrackId,
                kind: "auto",
                members: [
                    {
                        clipId: clip.id,
                        takeId: clip.takeId,
                        syllableId: memberSyllableId,
                        sourceStartSec: sourceStart,
                        sourceEndSec: sourceEnd,
                        sourceStartForMatchSec,
                        sourceEndForMatchSec,
                        startSec: span.startSec,
                        endSec: span.endSec,
                    },
                ],
                startSec: span.startSec,
                endSec: span.endSec,
                averageValue: null,
            });
        }
    }
    return blocks;
}

/**
 * mock 音节边界：固定时长切片，最后一段吸收余量。
 *
 * 只在显式请求 mock（音频 Clip 没有 MIDI 音符）时启用；真实工程绝不能凭曲线
 * 零散跳变自动切片，否则用户画一笔就会改变块的数量。
 */
function mockSyllableSpans(member: ParamBlockMember): Array<{ startSec: number; endSec: number }> {
    const spans: Array<{ startSec: number; endSec: number }> = [];
    for (let startSec = member.startSec; startSec < member.endSec - 1e-9;) {
        const endSec = Math.min(member.endSec, startSec + MOCK_SYLLABLE_SEC);
        if (endSec - startSec >= MIN_AUTO_BLOCK_SEC) spans.push({ startSec, endSec });
        startSec = endSec;
    }
    return spans.length > 0 ? spans : [{ startSec: member.startSec, endSec: member.endSec }];
}

/** 块覆盖的所有成员的总时间区间。 */
export function blockBounds(block: ParamBlock): { startSec: number; endSec: number } {
    return { startSec: block.startSec, endSec: block.endSec };
}

/** 两个块是否在时间上首尾相接或重叠。 */
export function blocksAreAdjacent(a: ParamBlock, b: ParamBlock): boolean {
    return a.startSec <= b.endSec + 1e-6 && b.startSec <= a.endSec + 1e-6;
}

/** 是否已存在把这组 Clip 全部覆盖的合并块。 */
function hasSameMergedBlock(state: ParamBlockState, blockIds: readonly string[]): boolean {
    const wanted = new Set(blockIds);
    return state.mergedBlocks.some(
        (merged) =>
            merged.blockIds !== undefined &&
            merged.blockIds.length === wanted.size &&
            merged.blockIds.every((id) => wanted.has(id)),
    );
}

/**
 * 合并选中的块。
 *
 * 只接受同一根轨道组、时间相邻（或重叠）的块；跨组直接拒绝。合并结果是成员
 * Clip 的集合，不复制参数。第一版按成员 Clip 粒度处理：用户多选同一 Clip 内的
 * 两个音节也会合并为一个成员块，后续可扩展为音节级成员。
 */
export function mergeParamBlocks(args: {
    readonly state: ParamBlockState;
    readonly blocks: readonly ParamBlock[];
    readonly selectedBlockIds: readonly string[];
}): BlockMergeResult {
    const selected = args.selectedBlockIds
        .map((id) => args.blocks.find((block) => block.id === id))
        .filter((block): block is ParamBlock => block !== undefined);
    if (selected.length < 2) return { ok: false, reason: "empty" };

    const rootTrackId = selected[0].rootTrackId;
    if (selected.some((block) => block.rootTrackId !== rootTrackId)) {
        return { ok: false, reason: "different_root" };
    }

    const ordered = [...selected].sort((a, b) => a.startSec - b.startSec);
    for (let index = 1; index < ordered.length; index += 1) {
        if (!blocksAreAdjacent(ordered[index - 1], ordered[index])) {
            return { ok: false, reason: "not_adjacent" };
        }
    }

    const blockIds = ordered.map((block) => block.id);
    if (hasSameMergedBlock(args.state, blockIds)) {
        return { ok: false, reason: "already_merged" };
    }

    return {
        ok: true,
        state: {
            ...args.state,
            mergedBlocks: [
                ...args.state.mergedBlocks,
                {
                    id: `merged:${blockIds.join("+")}`,
                    blockIds,
                },
            ],
        },
    };
}

/** 解合并：移除覆盖指定成员集合的合并块。 */
export function unmergeParamBlocks(
    state: ParamBlockState,
    ids: readonly string[],
): ParamBlockState {
    const wanted = new Set(ids);
    const wantedSyllables = new Set(
        state.persistedMerges
            ?.filter((merged) => wanted.has(merged.id))
            .flatMap((merged) => merged.members.map((member) => member.syllableId)) ?? [],
    );
    return {
        ...state,
        mergedBlocks: state.mergedBlocks.filter(
            (merged) =>
                !(merged.blockIds ?? []).every((id) => wanted.has(id)) &&
                !(merged.clipIds ?? []).some((id) => wanted.has(id)),
        ),
        persistedMerges: (state.persistedMerges ?? []).filter(
            (merged) =>
                !wanted.has(merged.id) &&
                !merged.members.some((member) => wantedSyllables.has(member.syllableId)),
        ),
    };
}

/**
 * 合并状态 → 实际块列表。
 *
 * 合并块按成员块的**并集**覆盖整个时间范围；被合并掉的自动块不再单列。
 * 旧记录若只有 `clipIds`，按整块 Clip 兼容（升级前的会话状态）。
 */
export function materializeParamBlocks(args: {
    readonly rootTrackId: string;
    readonly clips: readonly ParamBlockSourceClip[];
    readonly state: ParamBlockState;
}): ParamBlock[] {
    return materializeNonOverlappingParamBlocks(args);
}

/** 自动重叠归并的稳定身份：只由成员自动块 id 决定，不借用用户合并记录的 id。 */
function overlapGroupId(blockIds: readonly string[]): string {
    return `overlap:${[...new Set(blockIds)].sort().join("+")}`;
}

/** 把状态里已登记的合并记录校核成真实块（忽略失效成员）。 */
function materializeMergedRecords(
    args: {
        readonly rootTrackId: string;
        readonly clips: readonly ParamBlockSourceClip[];
        readonly state: ParamBlockState;
    },
    auto: readonly ParamBlock[],
): ParamBlock[] {
    const clipById = new Map(args.clips.map((clip) => [clip.id, clip]));
    const autoById = new Map(auto.map((block) => [block.id, block]));
    const merged: ParamBlock[] = [];
    for (const record of args.state.mergedBlocks) {
        if (record.blockIds !== undefined) {
            const blocks = record.blockIds
                .map((id) => autoById.get(id))
                .filter((block): block is ParamBlock => block !== undefined)
                .sort((a, b) => a.startSec - b.startSec);
            if (blocks.length < 2) continue;
            merged.push({
                id: record.id,
                rootTrackId: args.rootTrackId,
                kind: "merged",
                members: blocks.flatMap((block) => block.members),
                startSec: Math.min(...blocks.map((block) => block.startSec)),
                endSec: Math.max(...blocks.map((block) => block.endSec)),
                averageValue: null,
            });
            continue;
        }
        // 旧会话状态兼容：clipIds 代表整块 Clip。
        const members = (record.clipIds ?? [])
            .map((id) => clipById.get(id))
            .filter((clip): clip is ParamBlockSourceClip => clip !== undefined)
            .map(clipSpan)
            .sort((a, b) => a.startSec - b.startSec);
        if (members.length < 2) continue;
        merged.push({
            id: record.id,
            rootTrackId: args.rootTrackId,
            kind: "merged",
            members,
            startSec: Math.min(...members.map((member) => member.startSec)),
            endSec: Math.max(...members.map((member) => member.endSec)),
            averageValue: null,
        });
    }
    // v7 持久化记录：按稳定 Clip/Take/音节引用找自动块，并校验源区间。
    // 找不到（重分析后边界变了、Clip 被删）就丢弃该记录，绝不捏造块。
    const persistedAuto = new Map(
        auto.flatMap((block) =>
            block.members
                .filter(
                    (member) =>
                        member.syllableId !== undefined &&
                        member.sourceStartForMatchSec !== undefined &&
                        member.sourceEndForMatchSec !== undefined,
                )
                .map(
                    (member) =>
                        [
                            `${member.clipId}|${member.takeId ?? ""}|${member.sourceStartForMatchSec!.toFixed(3)}|${member.sourceEndForMatchSec!.toFixed(3)}`,
                            block,
                        ] as const,
                ),
        ),
    );
    for (const record of args.state.persistedMerges ?? []) {
        /* 已被旧会话 / 手动合并记录占用的自动块不能再组成另一组。 */
        const usedByManual = new Set(
            merged.flatMap((block) => block.members.map((member) => member.syllableId)),
        );
        const matched: ParamBlock[] = [];
        for (const member of record.members) {
            const block = persistedAuto.get(
                `${member.clipId}|${member.takeId}|${member.sourceStartSec.toFixed(3)}|${member.sourceEndSec.toFixed(3)}`,
            );
            if (block === undefined || matched.includes(block)) continue;
            if (
                block.members.some(
                    (m) => m.syllableId !== undefined && usedByManual.has(m.syllableId),
                )
            ) {
                continue;
            }
            const real = block.members.find(
                (m) =>
                    m.syllableId === member.syllableId &&
                    Math.abs(m.sourceStartForMatchSec! - member.sourceStartSec) < 1e-3 &&
                    Math.abs(m.sourceEndForMatchSec! - member.sourceEndSec) < 1e-3,
            );
            if (real !== undefined) matched.push(block);
        }
        if (matched.length < 2) continue;
        matched.sort((a, b) => a.startSec - b.startSec);
        const clipById = new Map(args.clips.map((clip) => [clip.id, clip]));
        const sameRoot = matched.every((block) => {
            const clip = clipById.get(block.members[0]?.clipId ?? "");
            return clip !== undefined && clip.trackId === args.rootTrackId;
        });
        if (!sameRoot) continue;
        const adjacent = matched.every((block, index) =>
            index === 0 ? true : blocksAreAdjacent(matched[index - 1], block),
        );
        if (!adjacent) continue;
        // 持久化记录与旧会话记录可能指向同一组；后者优先，避免重复块。
        if (
            merged.some((existing) => {
                const existingIds = new Set(
                    existing.members.map((m) => m.syllableId).filter(Boolean),
                );
                return matched.every((block) =>
                    block.members.some(
                        (m) => m.syllableId !== undefined && existingIds.has(m.syllableId),
                    ),
                );
            })
        ) {
            continue;
        }
        merged.push({
            id: record.id,
            rootTrackId: args.rootTrackId,
            kind: "merged",
            members: matched.flatMap((block) => block.members),
            startSec: Math.min(...matched.map((block) => block.startSec)),
            endSec: Math.max(...matched.map((block) => block.endSec)),
            averageValue: null,
        });
    }
    return merged;
}

/** 已登记合并块占用的自动块 / 旧 Clip。 */
function occupiedByMerged(args: {
    readonly state: ParamBlockState;
    readonly clips: readonly ParamBlockSourceClip[];
}): { blockIds: Set<string>; clipIds: Set<string> } {
    const blockIds = new Set<string>();
    const clipIds = new Set<string>();
    for (const record of args.state.mergedBlocks) {
        for (const id of record.blockIds ?? []) blockIds.add(id);
        for (const id of record.clipIds ?? []) clipIds.add(id);
    }
    for (const record of args.state.persistedMerges ?? []) {
        for (const member of record.members) {
            blockIds.add(member.syllableId);
        }
    }
    return { blockIds, clipIds };
}

/** 状态里未被合并记录吞掉的自动块。 */
function materializeAutoBlocks(
    args: {
        readonly state: ParamBlockState;
        readonly clips: readonly ParamBlockSourceClip[];
    },
    auto: readonly ParamBlock[],
): ParamBlock[] {
    const occupied = occupiedByMerged(args);
    return auto.filter(
        (block) =>
            !occupied.blockIds.has(block.id) &&
            !block.members.some(
                (member) =>
                    member.syllableId !== undefined && occupied.blockIds.has(member.syllableId),
            ) &&
            !occupied.clipIds.has(block.members[0]?.clipId ?? ""),
    );
}

/**
 * 合并重叠块后的最终派生。
 *
 * 【硬不变量】同一根轨道组里不允许出现两个时间重叠的块。手动合并记录先生效，
 * 其余自动块再按真实时间重叠做一次传递闭包归并（A 与 B 重叠、B 与 C 重叠，
 * 即使 A 与 C 不直接重叠，也归入同一组）。这样“重叠块”在数据层面就不存在。
 *
 * 【自动归并不登记进 state】它必须只是派生结果。若把它的记录塞回
 * `mergedBlocks`，块拖动期间每次重算都会与上一条记录互相覆盖，同一区间会在
 * “合并块”和成员块之间来回跳，表现为两个块之间频繁闪烁。这里直接在派生层
 * 归并，并给出独立的稳定 id；用户手动合并仍由 `mergeParamBlocks` 记录。
 */
function materializeNonOverlappingParamBlocks(args: {
    readonly rootTrackId: string;
    readonly clips: readonly ParamBlockSourceClip[];
    readonly state: ParamBlockState;
}): ParamBlock[] {
    const auto = autoBlocksForClips(args.rootTrackId, args.clips);
    const merged = materializeMergedRecords(args, auto);
    const remaining = materializeAutoBlocks(args, auto);
    const result = mergeOverlappingParamBlocks({
        state: { mergedBlocks: [] },
        blocks: remaining,
        includeExisting: false,
    });
    const autoById = new Map(remaining.map((block) => [block.id, block]));
    const overlapMerged: ParamBlock[] = [];
    for (const record of result.ok ? result.state.mergedBlocks : []) {
        const blocks = (record.blockIds ?? [])
            .map((id) => autoById.get(id))
            .filter((block): block is ParamBlock => block !== undefined)
            .sort((a, b) => a.startSec - b.startSec);
        if (blocks.length < 2) continue;
        overlapMerged.push({
            id: overlapGroupId(blocks.map((block) => block.id)),
            rootTrackId: args.rootTrackId,
            kind: "merged",
            members: blocks.flatMap((block) => block.members),
            startSec: Math.min(...blocks.map((block) => block.startSec)),
            endSec: Math.max(...blocks.map((block) => block.endSec)),
            averageValue: null,
        });
    }
    const swallowed = new Set(
        (result.ok ? result.state.mergedBlocks : []).flatMap((record) => record.blockIds ?? []),
    );
    const combined = [
        ...merged,
        ...overlapMerged,
        ...remaining.filter((block) => !swallowed.has(block.id)),
    ];
    /*
     * 手动合并记录之间也可能形成重叠（旧会话状态 / 解合并后的边界变化）。
     * 对**所有**结果再做一次归并，保证输出层面没有重叠块。
     */
    return collapseOverlappingResult(combined, args.rootTrackId);
}

/** 把任意块列表按时间重叠归并成互不重叠的块（最终兜底）。 */
function collapseOverlappingResult(
    blocks: readonly ParamBlock[],
    rootTrackId: string,
): ParamBlock[] {
    const sorted = blocks
        .filter((block) => Number.isFinite(block.startSec) && Number.isFinite(block.endSec))
        .slice()
        .sort((a, b) => a.startSec - b.startSec);
    const groups: ParamBlock[][] = [];
    let current: ParamBlock[] = [];
    let currentEnd = Number.NEGATIVE_INFINITY;
    for (const block of sorted) {
        if (current.length > 0 && block.startSec < currentEnd - 1e-9) {
            current.push(block);
            currentEnd = Math.max(currentEnd, block.endSec);
            continue;
        }
        if (current.length > 0) groups.push(current);
        current = [block];
        currentEnd = block.endSec;
    }
    if (current.length > 0) groups.push(current);
    return groups.map((group) => {
        if (group.length === 1) return group[0];
        return {
            id: overlapGroupId(group.map((block) => block.id)),
            rootTrackId,
            kind: "merged" as const,
            members: group.flatMap((block) => block.members),
            startSec: Math.min(...group.map((block) => block.startSec)),
            endSec: Math.max(...group.map((block) => block.endSec)),
            averageValue: null,
        };
    });
}

/**
 * 把块的时间区间与平均参数投影成可绘制 / 可命中的矩形。
 *
 * @param args.valueToY 参数值 → 视口 y 的**唯一**投影函数（与曲线共用）。
 * @param args.minValue / maxValue 当前参数值域。缺省时只能得到最小高度，
 *   调用方应尽量提供真实值域。
 */
export function layoutParamBlockRects(args: {
    readonly blocks: readonly ParamBlock[];
    readonly valueToY: (value: number) => number;
    readonly viewportHeightPx: number;
    readonly minValue?: number | null;
    readonly maxValue?: number | null;
    readonly selectedBlockIds: ReadonlySet<string>;
    /** 当前参数窗口（秒）；只用于区分“样本完整”，不裁掉块本身。 */
    readonly sampleStartSec?: number | null;
    readonly sampleEndSec?: number | null;
    /** 当前参数的一行（一个半音 / 一个值步）在视口中的像素高度。 */
    readonly rowHeightPx?: number | null;
    /** 当前参数窗口（帧域）；与块完全不相交时标记 `sampled=false`。 */
    readonly windowStartFrame?: number | null;
    readonly windowEndFrame?: number | null;
    readonly framePeriodMs?: number | null;
    /** 当前参数值域（给宿主与纯布局共用；缺省用传进来的 valueToY）。 */
    readonly valueDomain?: {
        readonly min: number;
        readonly max: number;
        readonly center: number;
        readonly span: number;
    } | null;
}): ParamBlockRect[] {
    const height = Math.max(1, finite(args.viewportHeightPx, 1));
    const rowHeight = Number.isFinite(args.rowHeightPx)
        ? Math.max(1, args.rowHeightPx as number)
        : null;
    const fixedHeight = Math.max(
        1,
        Math.min(height, (rowHeight ?? 16) * PARAM_BLOCK_ROW_HEIGHT_MULTIPLIER),
    );
    const rects: ParamBlockRect[] = [];
    let ordinal = 0;
    for (const block of args.blocks) {
        if (
            !Number.isFinite(block.startSec) ||
            !Number.isFinite(block.endSec) ||
            !(block.endSec > block.startSec)
        ) {
            continue;
        }
        /*
         * 窗口只决定**样本是否完整**，不决定块是否存在。
         * 块的时间投影始终由当前轴/Clip 真实区间给出；若拿旧窗口当裁剪器，
         * 滚动或导入后会出现整批块凭空消失。
         */
        const sampled =
            args.sampleStartSec == null ||
            args.sampleEndSec == null ||
            (block.endSec >= args.sampleStartSec - 1e-6 &&
                block.startSec <= args.sampleEndSec + 1e-6);
        if (block.averageValue === null || !Number.isFinite(block.averageValue)) {
            continue;
        }
        /*
         * 取数窗口只决定**样本是否完整**，不决定块是否存在于画面。
         * 块的时间投影与曲线一样由当前轴决定；若用旧窗口裁掉整个块，滚动或
         * 导入后会出现“整批块突然消失”，而它们的标题 / 选中态仍然有效。
         * 纵向是否在视口内由下面的 top/height 判定。
         */
        const heightPx = fixedHeight;
        /*
         * 值域视口由**宿主实时提供**：纵向滚动 / 缩放时，valueToY 的闭包可能
         * 还停在 React 提交时的 ref；这里若带了 valueDomain 就按它显式投影，
         * 保证块的位置与当前曲线采用同一帧的值域。
         */
        const domain = args.valueDomain;
        const projectedY =
            domain !== null &&
            domain !== undefined &&
            Number.isFinite(domain.span) &&
            domain.span > 1e-9
                ? (() => {
                      const span = domain.span;
                      const min = Math.max(
                          domain.min,
                          Math.min(domain.max - span, domain.center - span / 2),
                      );
                      const t = (block.averageValue - min) / span;
                      return (1 - Math.min(1, Math.max(0, t))) * height;
                  })()
                : finite(args.valueToY(block.averageValue), height / 2);
        const topPx = projectedY - heightPx / 2;
        // 完全离开纵向视口：不生成几何，也不命中。
        if (topPx + heightPx <= 0 || topPx >= height) continue;
        ordinal += 1;
        rects.push({
            id: block.id,
            startSec: block.startSec,
            endSec: block.endSec,
            ordinal,
            sampled,
            topPx,
            heightPx,
            averageValue: block.averageValue,
            selected: args.selectedBlockIds.has(block.id),
            merged: block.kind === "merged",
        });
    }
    return rects;
}

/**
 * 拖动时的**纵向值偏移**（唯一来源，预览与提交共用）。
 *
 * 音高 / 音分 / 度数 / 张力等是线性刻度，直接照指针位移。
 *
 * 对块拖动来说，`valueDelta` 本身就是指针在**值域纵轴**上的位移；把它再按
 * “锚点值”缩放会让块只移动一部分距离（锚点越小时越明显）。因此所有参数
 * 统一做线性偏移，保证块中心与指针同步；值域钳制仍由调用方的写入入口负责。
 */
export function paramBlockDragValue(args: {
    readonly param: string;
    readonly original: number;
    readonly anchorValue: number;
    readonly valueDelta: number;
}): number {
    if (!Number.isFinite(args.original)) return args.original;
    if (!Number.isFinite(args.valueDelta)) return args.original;
    return args.original + args.valueDelta;
}

/**
 * 吸附拖动时的**目标值**（预览标签与提交共用）。
 *
 * - pitch：`step=1` → 半音；有 `scaleNotes` → 工程音阶；
 * - 其它参数：按 `step` 取整。
 * 返回的是**位移**，调用方再加到原曲线上。
 */
export function resolveParamBlockSnapDelta(args: {
    readonly param: string;
    readonly startValue: number;
    readonly rawDelta: number;
    readonly step: number;
    readonly scaleNotes?: readonly number[] | null;
}): number {
    if (!Number.isFinite(args.rawDelta)) return 0;
    if (args.param === "pitch") {
        if (args.scaleNotes && args.scaleNotes.length > 0) {
            const from = snapScaleMidi(args.startValue, args.scaleNotes);
            const to = snapScaleMidi(args.startValue + args.rawDelta, args.scaleNotes);
            return to - from;
        }
        return Math.round(args.rawDelta);
    }
    const step = Math.max(1e-9, Number(args.step) || 0);
    return Math.round(args.rawDelta / step) * step;
}

/**
 * 吸附步进：音高可按半音 / 工程音阶；其余参数按调用方给的网格步长。
 * 返回**吸附后的值偏移**，保证预览起点与提交起点一致。
 */
export function snapParamBlockValueDelta(args: {
    readonly delta: number;
    readonly step: number;
}): number {
    const step = Math.max(1e-9, Number(args.step) || 0);
    return Math.round(args.delta / step) * step;
}

/**
 * 把一次块拖动换算成**统一的写入描述**。
 *
 * 自由拖动：值偏移 = 指针位移。
 * 吸附拖动：先把目标值吸附到半音 / 音阶 / 网格，再用「吸附目标 − 起点均值」作为
 * 偏移。这样预览与写回都基于同一个数学量，不会出现“松手跳一格”。
 */
export function resolveParamBlockDragWrite(args: {
    readonly param: string;
    readonly startValue: number;
    readonly candidateValue: number;
    readonly snap: boolean;
    readonly step: number;
    readonly scaleNotes?: readonly number[] | null;
}): number {
    if (!Number.isFinite(args.candidateValue)) return 0;
    const rawDelta = args.candidateValue - args.startValue;
    if (!args.snap) return rawDelta;
    if (args.param === "pitch" && args.scaleNotes && args.scaleNotes.length > 0) {
        // 以拖动**起点**为音级锚点：先把候选值吸到最近音级，再按起点音级换算
        // 半音差；这样 60 → 61.2 在 C 大调下落到 62（而不是因 0.2 残差回到 60）。
        const from = snapScaleMidi(args.startValue, args.scaleNotes);
        const to = snapScaleMidi(args.startValue + rawDelta, args.scaleNotes);
        // 向下拖时取“不越过锚点”的另一侧音级，避免半音残差让方向反转。
        if (rawDelta < 0 && to >= from) {
            const lower = snapScaleMidiBelow(args.startValue + rawDelta - 1e-9, args.scaleNotes);
            return lower - from;
        }
        return to - from;
    }
    return snapParamBlockValueDelta({ delta: rawDelta, step: args.step });
}

function snapScaleMidi(midi: number, notes: readonly number[]): number {
    const normalized = [...new Set(notes.map((n) => ((Math.round(n) % 12) + 12) % 12))];
    if (normalized.length === 0) return Math.round(midi);
    const base = Math.floor(midi);
    let best = base;
    let bestDistance = Number.POSITIVE_INFINITY;
    for (let candidate = base - 12; candidate <= base + 12; candidate += 1) {
        if (!normalized.includes(((candidate % 12) + 12) % 12)) continue;
        const distance = Math.abs(candidate - midi);
        if (distance < bestDistance) {
            best = candidate;
            bestDistance = distance;
        }
    }
    return best;
}

function snapScaleMidiBelow(midi: number, notes: readonly number[]): number {
    const normalized = [...new Set(notes.map((n) => ((Math.round(n) % 12) + 12) % 12))];
    if (normalized.length === 0) return Math.floor(midi);
    const base = Math.floor(midi);
    let best = base;
    let bestDistance = Number.POSITIVE_INFINITY;
    for (let candidate = base; candidate >= base - 24; candidate -= 1) {
        if (!normalized.includes(((candidate % 12) + 12) % 12)) continue;
        const distance = midi - candidate;
        if (distance >= -1e-9 && distance < bestDistance) {
            best = candidate;
            bestDistance = distance;
        }
    }
    return best;
}

/**
 * 一键合并：把所有**时间上真正重叠**的块各自合成一组。
 *
 * 相邻（首尾相接）不算重叠；只有 `start < previousEnd - epsilon` 才并入同组。
 * 已合并的块不再参与（避免重复记录）；没有任何重叠时返回 `nothing_to_merge`。
 */
export function mergeOverlappingParamBlocks(args: {
    readonly state: ParamBlockState;
    readonly blocks: readonly ParamBlock[];
    /**
     * 派生层的一次性归并：只返回本批新重叠组。
     *
     * 用户显式执行「合并当前重叠」时保留默认 `true`，结果可写入状态；
     * `materialize` 只需要本批分组，避免已有记录被重复并入。
     */
    readonly includeExisting?: boolean;
}): BlockMergeResult {
    const candidates = args.blocks.slice().sort((a, b) => a.startSec - b.startSec);
    const groups: ParamBlock[][] = [];
    let current: ParamBlock[] = [];
    let currentEnd = Number.NEGATIVE_INFINITY;
    for (const block of candidates) {
        if (current.length > 0 && block.startSec < currentEnd - 1e-9) {
            current.push(block);
            currentEnd = Math.max(currentEnd, block.endSec);
            continue;
        }
        if (current.length > 1) groups.push(current);
        current = [block];
        currentEnd = block.endSec;
    }
    if (current.length > 1) groups.push(current);
    if (groups.length === 0) return { ok: false, reason: "nothing_to_merge" };

    const mergedBlocks = [
        ...(args.includeExisting === false ? [] : args.state.mergedBlocks),
        ...groups.map((group) => {
            const blockIds = group.map((block) => block.id);
            /*
             * 组内成员自身的成员块也要保留：若其中某个块本来就是 merged，
             * 只记它的 id 会让下一次派生找不到对应自动块而丢内容。
             */
            const sourceIds = group.flatMap((block) =>
                block.kind === "merged"
                    ? args.state.mergedBlocks
                          .filter((record) => record.id === block.id)
                          .flatMap((record) => record.blockIds ?? [])
                    : [block.id],
            );
            return {
                id: `merged:${(sourceIds.length > 0 ? sourceIds : blockIds).join("+")}`,
                blockIds: sourceIds.length > 0 ? sourceIds : blockIds,
            };
        }),
    ];
    return {
        ok: true,
        state: { ...args.state, mergedBlocks },
    };
}

/**
 * 把选中的块投影成逐帧写入片段。
 *
 * - 时间范围取块的真实区间（绘制用的缝隙不参与编辑）；
 * - 每个块独立取 `sourceAt`，所以多选时不会把别的块的曲线当基准；
 * - 重叠帧后写覆盖先写（稳定排序，可复现）；
 * - `valueDelta` 由 {@link paramBlockDragValue} 逐帧应用，pitch 的 0 哨兵保持 0。
 */
export function buildParamBlockDragSegments(args: {
    readonly blocks: readonly { readonly members: readonly ParamBlockMember[] }[];
    readonly framePeriodMs: number;
    readonly param: string;
    readonly anchorValue: number;
    readonly valueDelta: number;
    readonly sourceAt: (frame: number) => number;
}): Array<{ startFrame: number; values: number[] }> {
    const fp = Math.max(1e-6, args.framePeriodMs);
    const spans = args.blocks
        .flatMap((block) => block.members)
        .map((member) => {
            const startFrame = Math.max(0, Math.floor((member.startSec * 1000) / fp));
            const endFrame = Math.max(startFrame, Math.ceil((member.endSec * 1000) / fp) - 1);
            return { startFrame, endFrame };
        })
        .sort((a, b) => a.startFrame - b.startFrame);
    if (spans.length === 0) return [];

    const merged: Array<{ startFrame: number; endFrame: number }> = [];
    for (const span of spans) {
        const previous = merged[merged.length - 1];
        if (previous !== undefined && span.startFrame <= previous.endFrame + 1) {
            previous.endFrame = Math.max(previous.endFrame, span.endFrame);
        } else {
            merged.push({ ...span });
        }
    }

    return merged
        .map((span) => {
            const values: number[] = [];
            for (let frame = span.startFrame; frame <= span.endFrame; frame += 1) {
                const source = args.sourceAt(frame);
                if (args.param === "pitch" && source <= 0) {
                    values.push(source);
                    continue;
                }
                values.push(
                    paramBlockDragValue({
                        param: args.param,
                        original: source,
                        anchorValue: args.anchorValue,
                        valueDelta: args.valueDelta,
                    }),
                );
            }
            return { startFrame: span.startFrame, values };
        })
        .filter((segment) => segment.values.length > 0);
}

/**
 * 拖动块的**实时预览**：只返回受影响的逐帧值。
 *
 * 与提交使用同一 `paramBlockDragValue` 与同一区间合并逻辑；数据源固定为
 * **手势开始时的已提交曲线**（`pv.orig`），所以预览不会把上一帧结果再累加一次。
 * 降采样的预览与全分辨率提交只在采样密度上不同，公式不分叉。
 */
export function buildParamBlockPreviewSegments(args: {
    readonly blocks: readonly { readonly members: readonly ParamBlockMember[] }[];
    readonly framePeriodMs: number;
    readonly param: string;
    readonly anchorValue: number;
    readonly valueDelta: number;
    readonly pv: {
        readonly startFrame: number;
        readonly stride: number;
        readonly edit: readonly number[];
        readonly orig: readonly number[];
    };
}): Array<{ startFrame: number; values: number[] }> {
    const step = Math.max(1, Math.floor(args.pv.stride));
    return buildParamBlockDragSegments({
        blocks: args.blocks,
        framePeriodMs: args.framePeriodMs,
        param: args.param,
        anchorValue: args.anchorValue,
        valueDelta: args.valueDelta,
        sourceAt: (frame) => {
            const index = Math.round((frame - args.pv.startFrame) / step);
            /*
             * 预览必须一直以**手势起点**的已提交曲线为基准。若读 live edit，
             * 上一帧的预览会成为下一帧的原值，拖动会被累积放大；松手提交
             * 则从真实曲线重算，落点因此与控制路径不一致。
             */
            const base = args.pv.orig;
            return index >= 0 && index < base.length ? Number(base[index]) || 0 : 0;
        },
    });
}

/**
 * 块命中测试：必须同时落在时间区间与**可见矩形**内。
 *
 * 【为什么不能只按时间列判断】块有真实高度；只按 x 选中会让用户点到块外、
 * 甚至点到曲线空白处仍被当成选块，多选时尤其容易误操作。
 */
export function hitTestParamBlocks<
    T extends {
        readonly id: string;
        readonly startSec: number;
        readonly endSec: number;
        readonly topPx: number;
        readonly heightPx: number;
    },
>(args: {
    readonly blocks: readonly T[];
    readonly sec: number;
    readonly localY: number;
    /** 命中容差（CSS px）：细块比 1px 边框更容易点中。 */
    readonly slopPx?: number;
}): T | null {
    const sec = finite(args.sec, Number.NaN);
    const y = finite(args.localY, Number.NaN);
    if (!Number.isFinite(sec) || !Number.isFinite(y)) return null;
    const slop = Math.max(0, finite(args.slopPx ?? 2, 2));
    // 反向遍历：后绘制的块在上，命中优先给用户实际看到的那个。
    for (let index = args.blocks.length - 1; index >= 0; index -= 1) {
        const block = args.blocks[index];
        if (block === undefined) continue;
        const startSec = Math.min(block.startSec, block.endSec);
        const endSec = Math.max(block.startSec, block.endSec);
        const top = block.topPx;
        const bottom = block.topPx + block.heightPx;
        if (sec < startSec || sec > endSec) continue;
        if (y < top - slop || y > bottom + slop) continue;
        return block;
    }
    return null;
}

/**
 * 最近块命中：用于工具刚打开、宿主尚未出首帧时没有块几何的情况。
 *
 * 只按时间选取与指针最近、且横向距离在容差内的块；纵向不参与，因为此时
 * 还没有可信的像素位置。一旦宿主出帧，调用方应优先使用真实矩形命中。
 */
export function nearestParamBlockByTime<
    T extends {
        readonly id: string;
        readonly startSec: number;
        readonly endSec: number;
    },
>(blocks: readonly T[], sec: number, toleranceSec: number): T | null {
    const target = finite(sec, Number.NaN);
    if (!Number.isFinite(target) || blocks.length === 0) return null;
    const tolerance = Math.max(0, finite(toleranceSec, 0));
    let best: T | null = null;
    let bestDistance = Number.POSITIVE_INFINITY;
    for (const block of blocks) {
        const start = Math.min(block.startSec, block.endSec);
        const end = Math.max(block.startSec, block.endSec);
        const distance =
            target >= start && target <= end ? 0 : target < start ? start - target : target - end;
        if (distance > tolerance) continue;
        if (distance < bestDistance) {
            best = block;
            bestDistance = distance;
        }
    }
    return best;
}

/**
 * 块内参数值的**高密度簇中心**（默认块位置算法）。
 *
 * 不再是算术均值：先按 `binWidth` 分箱，取样本权重最高的主簇；相邻箱按半权
 * 计入，避免值恰好落在箱边界时被拆成两簇。簇内以实际样本取均值，因此多峰
 * 材料落在“大部分帧所在的位置”，长颤音的对称极值也不会抬高块。
 * 密度接近（差值在 1/8 内）时取更靠近中位数的簇，保持视觉稳定。
 *
 * @param binWidth 显式箱宽；缺省由 {@link inferBlockBinWidth} 按参数量级推断。
 */
export function averageParamValueInSpan(args: {
    readonly startSec: number;
    readonly endSec: number;
    readonly values: readonly number[];
    readonly startFrame: number;
    readonly stride: number;
    readonly framePeriodMs: number;
    /** 参数是否为音高：跳过 <= 0 的无声帧，避免把静音拉低块高。 */
    readonly isPitch: boolean;
    /** 参数名，用于选择箱宽 / 静音判定。 */
    readonly param?: string;
    /** 显式箱宽（参数值单位）；缺省按参数推断。 */
    readonly binWidth?: number;
}): number | null {
    const fp = Math.max(1e-6, args.framePeriodMs);
    const stride = Math.max(1, args.stride);
    if (!Array.isArray(args.values) || args.values.length === 0) return null;
    const startFrame = Math.max(0, Math.ceil((args.startSec * 1000) / fp));
    const endFrame = Math.max(startFrame + 1, Math.ceil((args.endSec * 1000) / fp));
    const samples: number[] = [];
    for (let frame = startFrame; frame < endFrame; frame += stride) {
        const index = Math.floor((frame - args.startFrame) / stride);
        if (index < 0 || index >= args.values.length) continue;
        const value = Number(args.values[index]);
        if (!Number.isFinite(value)) continue;
        if (args.isPitch && value <= 0) continue;
        samples.push(value);
    }
    return modeCenterOfValues(samples, {
        param: args.param,
        isPitch: args.isPitch,
        binWidth: args.binWidth,
    });
}

/**
 * 一组参数值的密度众数中心（纯函数，宿主与面板共用）。
 *
 * 返回主簇的**样本均值**：箱只用于找出主簇，最终位置仍由真实样本决定，
 * 不会出现半音格的量化跳变。
 */
export function modeCenterOfValues(
    values: readonly number[],
    opts?: {
        readonly param?: string;
        readonly isPitch?: boolean;
        readonly binWidth?: number;
    },
): number | null {
    if (values.length === 0) return null;
    const sorted = values
        .filter((value) => Number.isFinite(value))
        .slice()
        .sort((a, b) => a - b);
    if (sorted.length === 0) return null;
    if (sorted.length === 1) return sorted[0] as number;
    const middle = Math.floor(sorted.length / 2);
    const median =
        sorted.length % 2 === 0
            ? ((sorted[middle - 1] as number) + (sorted[middle] as number)) / 2
            : (sorted[middle] as number);
    const q1 = sorted[Math.floor(sorted.length * 0.25)] as number;
    const q3 = sorted[Math.floor(sorted.length * 0.75)] as number;
    const spread = q3 - q1;
    const fallback = inferBlockBinWidth(opts?.param, opts?.isPitch === true, median);
    const binWidth = Math.max(
        1e-9,
        Number.isFinite(opts?.binWidth) && (opts?.binWidth as number) > 0
            ? (opts?.binWidth as number)
            : spread > fallback * 0.25
              ? Math.max(fallback, spread / 12)
              : fallback,
    );

    const counts = new Map<number, number[]>();
    for (const value of sorted) {
        const bin = Math.floor(value / binWidth);
        const bucket = counts.get(bin);
        if (bucket === undefined) counts.set(bin, [value]);
        else bucket.push(value);
    }
    const bins = [...counts.keys()].sort((a, b) => a - b);
    let best: { weight: number; center: number; distance: number } | null = null;
    for (const bin of bins) {
        const own = counts.get(bin) ?? [];
        const prev = counts.get(bin - 1) ?? [];
        const next = counts.get(bin + 1) ?? [];
        // 主箱必须非空；相邻箱只加权，不与主箱争夺“峰”的身份。
        if (own.length === 0) continue;
        const merged = [...prev, ...own, ...next];
        const weight = own.length + prev.length * 0.5 + next.length * 0.5;
        const center = merged.reduce((sum, value) => sum + value, 0) / merged.length;
        const candidate = {
            weight,
            center,
            distance: Math.abs(center - median),
        };
        /*
         * 只接受**明显更密**的簇；密度接近时保留第一个峰，并优先更靠近中位数的
         * 那个。不能让“后遍历到的峰”凭 1 个样本的差就替换，否则稳定音上零星
         * 的滑音会把整块拖走。
         */
        if (
            best === null ||
            candidate.weight > best.weight * 1.125 ||
            (candidate.weight >= best.weight * 0.875 && candidate.distance < best.distance - 1e-9)
        ) {
            best = candidate;
        }
    }
    return best === null ? (sorted[0] as number) : best.center;
}

/**
 * 从宿主镜像的逐帧采样中求某时间段的代表值。
 *
 * 采样窗口与块不相交时返回 null（宿主回退到 spec 的快照值），而不是伪造 0。
 */
export function modeCenterOfValuesInSamples(
    startSec: number,
    endSec: number,
    samples: {
        readonly values: readonly number[];
        readonly startFrame: number;
        readonly stride: number;
        readonly framePeriodMs: number;
        readonly isPitch: boolean;
        readonly param: string;
        readonly binWidth?: number;
    },
): number | null {
    if (samples.values.length === 0) return null;
    const fp = Math.max(1e-6, samples.framePeriodMs);
    const stride = Math.max(1, samples.stride);
    const startFrame = Math.max(0, Math.ceil((startSec * 1000) / fp));
    const endFrame = Math.max(startFrame + 1, Math.ceil((endSec * 1000) / fp));
    const values: number[] = [];
    for (let frame = startFrame; frame < endFrame; frame += stride) {
        const index = Math.floor((frame - samples.startFrame) / stride);
        if (index < 0 || index >= samples.values.length) continue;
        const value = Number(samples.values[index]);
        if (!Number.isFinite(value)) continue;
        if (samples.isPitch && value <= 0) continue;
        values.push(value);
    }
    return modeCenterOfValues(values, {
        param: samples.param,
        isPitch: samples.isPitch,
        binWidth: samples.binWidth,
    });
}

/** 当前参数的一行像素高；块高 = 它的 1.5 倍。 */
export function blockRowHeightPx(
    valueToY: ((value: number, heightPx: number) => number) | undefined,
    viewportHeightPx: number,
    param: string,
): number {
    if (param !== "pitch" || valueToY === undefined) return 16 * PARAM_BLOCK_ROW_HEIGHT_MULTIPLIER;
    const row = Math.abs(valueToY(60, viewportHeightPx) - valueToY(61, viewportHeightPx));
    return (Number.isFinite(row) && row > 1e-6 ? row : 16) * PARAM_BLOCK_ROW_HEIGHT_MULTIPLIER;
}

/** 块的短标签：音高用音名，其余参数显示一位小数。 */
export function formatParamBlockLabel(
    value: number | null,
    ordinal: number,
    param: string,
): string {
    if (value === null || !Number.isFinite(value)) return `#${ordinal}`;
    if (param === "pitch") {
        const rounded = Math.round(value);
        return `#${ordinal} ${midiLabel(rounded)}`;
    }
    return `#${ordinal} ${value.toFixed(1)}`;
}

/** 纯函数音名（避免 paramBlocks 在 node 单测里引入渲染模块）。 */
function midiLabel(midi: number): string {
    const names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
    return `${names[((midi % 12) + 12) % 12]}${Math.floor(midi / 12) - 1}`;
}

/** 按参数量级推断块位置直方图的箱宽。 */
function inferBlockBinWidth(param: string | undefined, isPitch: boolean, value: number): number {
    if (isPitch || param === "pitch") return 1;
    if (
        param === "tension" ||
        param?.endsWith("_cents") === true ||
        param?.endsWith("_degrees") === true
    ) {
        return 50;
    }
    if (param === "volume" || param === "dyn" || param === "dyn_edit") return 0.04;
    return Math.max(1e-6, Math.abs(value) * 0.02);
}
