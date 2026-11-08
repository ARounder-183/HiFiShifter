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

import type { ParamLineBlock, ParamLineBlockSet } from "./paramLineBlocks";
import { placeParamLineBlock, resizeParamLineBlock } from "./paramLineBlocks";
import { paramLineValueCenter } from "./paramLineBlockValue";
import {
    createLineBlock,
    deleteLineBlocks,
    lineBlocksForPanel,
    lineFrame,
    mergeLineBlocks,
    resizeLineBlock,
    splitLineBlock,
    withParamLineBlocks,
} from "./paramLineBlockEditing";

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
    /** 时间线 Clip 的真实 active take；手动块持久化必须使用它。 */
    readonly activeTakeId?: string;
    readonly takes?: ReadonlyArray<{ readonly id: string }>;
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
    readonly lineBlock?: ParamLineBlock;
    readonly lineFramePeriodMs?: number;
    readonly id: string;
    readonly rootTrackId: string;
    readonly kind: "auto" | "merged";
    readonly members: readonly ParamBlockMember[];
    readonly startSec: number;
    readonly endSec: number;
    /**
     * 独立块的几何锚点：块当时所属音节 / 容器的工程秒范围。
     *
     * 这不是父子关系。它只用于 Clip 移动 / 拉伸后把**这个块自己**的区间
     * 重新映射到同一位置；切分后的左右块各自保存锚点，互不引用。
     */
    readonly anchorStartSec?: number;
    readonly anchorEndSec?: number;
    /**
     * 块的代表参数值（**密度众数中心**，不是算术均值）。
     *
     * 用块内出现最密集的参数区间代表块的高度位置；颤音、短滑音与少量离群点
     * 不会被拉平。字段名保留 `averageValue` 以兼容既有宿主 / 测试载荷。
     */
    readonly averageValue: number | null;
    /** 音高块显式保存的 MIDI 中心；其他参数必须从各自曲线派生位置。 */
    readonly manualValue?: number | null;
    /**
     * 块间圆滑宽度（秒）：向两侧各 blend 这么多秒，跨越相邻块边界做线性过渡。
     * 只对同一 Clip/Take 内**首尾相接**的块生效；缺省 0 = 硬切换。
     */
    readonly transitionSec?: number;
    /** 实时拖动 / 过渡预览时显式保持连续中心；缺省吸附到半音行心。 */
    readonly displayCenterSnapToSemitone?: boolean;
}

/**
 * 参数块的可见矩形。
 *
 * 块高只由当前参数的一行像素高决定（1.2 行高），代表参数值决定纵向中心。
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
    /** 块中心 y；与曲线 `valueToY(midi + 0.5)` 同口径。 */
    readonly centerYPx: number;
    /** 与像素同源的块代表值（供拖动锚点直接读取）。 */
    readonly averageValue: number | null;
    readonly selected: boolean;
    readonly merged: boolean;
    /**
     * 本次时间伸缩手势的预览，使用**手势起点**的代表值锁定纵向中心。
     *
     * 过渡曲线会在相接点附近改变 live 曲线的主簇；若宿主继续按新主簇
     * 重算代表值，块会上下移动。这里只锁显示，不改变持久化高度锚点。
     */
    readonly displayCenterValue?: number | null;
}

/** 块边界命中结果：像素热区优先，避免窄块只能点中 1px 边线。 */
export interface ParamBlockEdgeHit<T> {
    readonly block: T;
    readonly edge: "left" | "right";
}

/** 块上的指针反馈：决定光标与手柄可见度。 */
export type ParamBlockHoverKind = "body" | "left-edge" | "right-edge" | "transition";

export function paramBlockCursorForHover(kind: ParamBlockHoverKind | null): string {
    if (kind === "left-edge" || kind === "right-edge") return "ew-resize";
    if (kind === "transition") return "ew-resize";
    if (kind === "body") return "grab";
    return "default";
}

/**
 * 块的左右边缘命中。
 *
 * 只按时间容差判断时，`pxPerSec` 很小会把“离边缘几像素”算成很大的秒数，
 * 窄块几乎整块都变成缩放热区。这里统一用**视口像素**：先把块左右端点投影成
 * x，再看指针 x 与哪一侧最近，并限制在热区半宽内。
 */
export function hitTestParamBlockEdges<
    T extends {
        readonly id: string;
        readonly startSec: number;
        readonly endSec: number;
        readonly topPx: number;
        readonly heightPx: number;
    },
>(args: {
    readonly blocks: readonly T[];
    readonly clientX: number;
    readonly localY: number;
    readonly secToX: (sec: number) => number;
    /** 单侧热区半宽（CSS px）。 */
    readonly hitPx?: number;
    /** 纵向允许超出块矩形的容差（CSS px）。 */
    readonly slopPx?: number;
    /** 只在指定块的端点内命中；hover 可传当前块，pointerdown 再按各自优先级筛选。 */
    readonly blockId?: string;
}): ParamBlockEdgeHit<T> | null {
    const hitPx = Math.max(1, finite(args.hitPx ?? 6, 6));
    const slop = Math.max(0, finite(args.slopPx ?? 4, 4));
    let best: { block: T; edge: "left" | "right"; distance: number } | null = null;
    for (let index = args.blocks.length - 1; index >= 0; index -= 1) {
        const block = args.blocks[index];
        if (block === undefined) continue;
        if (args.blockId !== undefined && block.id !== args.blockId) continue;
        if (args.localY < block.topPx - slop || args.localY > block.topPx + block.heightPx + slop) {
            continue;
        }
        const leftX = args.secToX(Math.min(block.startSec, block.endSec));
        const rightX = args.secToX(Math.max(block.startSec, block.endSec));
        for (const edge of ["left", "right"] as const) {
            const edgeX = edge === "left" ? leftX : rightX;
            const distance = Math.abs(args.clientX - edgeX);
            if (distance > hitPx) continue;
            if (best === null || distance < best.distance - 1e-9) {
                best = { block, edge, distance };
            }
        }
    }
    return best === null ? null : { block: best.block, edge: best.edge };
}

/**
 * 相接块过渡手柄的命中区。
 *
 * 手柄绘制在块右边界外侧上方，视觉上是“圆点 + 横线”。命中只覆盖这条
 * 水平短线附近，**不覆盖块右边缘的缩放热区**：两者在空间上分开，避免
 * 圆滑手柄把伸缩控件吃掉。
 */
export function hitTestParamBlockTransitionHandles<
    T extends {
        readonly id?: string;
        readonly topPx: number;
        readonly heightPx: number;
        readonly transitionHandle?: {
            readonly sec: number;
            readonly leftId: string;
            readonly rightId: string;
            readonly widthSec: number;
        } | null;
    },
>(args: {
    readonly blocks: readonly T[];
    readonly clientX: number;
    readonly localY: number;
    readonly secToX: (sec: number) => number;
    readonly hitHalfWidthPx?: number;
    readonly hitHalfHeightPx?: number;
    /** 手柄圆点相对右边界的水平偏移（CSS px）；0 = 正上方对齐。 */
    readonly outsetPx?: number;
    /** 手柄圆点相对块顶的上抬量（CSS px，正值在块外）。 */
    readonly outsetYPx?: number;
}): {
    readonly leftId: string;
    readonly rightId: string;
    readonly sec: number;
    readonly widthSec: number;
    readonly topPx: number;
    readonly heightPx: number;
} | null {
    // 命中区比视觉圆点/横线略大，保证细指针也能抓住；仍与块右边缘分开。
    const halfW = Math.max(1, finite(args.hitHalfWidthPx ?? 12, 12));
    const halfH = Math.max(1, finite(args.hitHalfHeightPx ?? 7, 7));
    const outsetX = finite(
        args.outsetPx ?? PARAM_BLOCK_TRANSITION_HANDLE_OUTSET_X_PX,
        PARAM_BLOCK_TRANSITION_HANDLE_OUTSET_X_PX,
    );
    const outsetY = Math.max(
        0,
        finite(args.outsetYPx ?? PARAM_BLOCK_TRANSITION_HANDLE_OUTSET_Y_PX, 6),
    );
    let best: {
        leftId: string;
        rightId: string;
        sec: number;
        widthSec: number;
        topPx: number;
        heightPx: number;
        distance: number;
    } | null = null;
    const byId = new Map(args.blocks.map((block) => [block.id, block]));
    for (const block of args.blocks) {
        const handle = block.transitionHandle;
        if (handle === null || handle === undefined) continue;
        const x = args.secToX(handle.sec);
        const handleX = x + outsetX;
        const handleY = paramBlockTransitionHandleY(
            block.topPx,
            byId.get(handle.rightId)?.topPx,
            outsetY,
        );
        const dx = args.clientX - handleX;
        const dy = args.localY - handleY;
        if (Math.abs(dx) > halfW || Math.abs(dy) > halfH) continue;
        const distance = Math.hypot(dx, dy);
        if (best === null || distance < best.distance) {
            best = {
                leftId: handle.leftId,
                rightId: handle.rightId,
                sec: handle.sec,
                widthSec: handle.widthSec,
                topPx: block.topPx,
                heightPx: block.heightPx,
                distance,
            };
        }
    }
    if (best === null) return null;
    return {
        leftId: best.leftId,
        rightId: best.rightId,
        sec: best.sec,
        widthSec: best.widthSec,
        topPx: best.topPx,
        heightPx: best.heightPx,
    };
}

/** 块边缘延展的实时网格吸附 + 最小宽度 / 邻居边界约束。 */
export function resolveParamBlockResizeTarget(args: {
    readonly edge: "left" | "right";
    readonly startSec: number;
    readonly endSec: number;
    readonly rawTargetSec: number;
    readonly snapToGrid: (sec: number) => number;
    /** 原始指针边界：即使吸附后落在限制之外，也允许联动压缩邻居。 */
    readonly unclampedTargetSec?: number;
    readonly minBlockSec: number;
    readonly neighborStartSec?: number | null;
    readonly neighborEndSec?: number | null;
}): number {
    const minSec = Math.max(1e-6, finite(args.minBlockSec, MIN_AUTO_BLOCK_SEC));
    /*
     * `unclampedTargetSec` 是未受邻居限制的指针端点。吸附永远基于它，而不是
     * 上一帧已钳制的 target；否则越靠近邻居，吸附结果越会被拉回旧边界。
     */
    const raw =
        args.unclampedTargetSec !== undefined && Number.isFinite(args.unclampedTargetSec)
            ? args.unclampedTargetSec
            : args.rawTargetSec;
    const snapped = finite(args.snapToGrid(raw), raw);
    const plan = resolveParamBlockResizePlan({
        edge: args.edge,
        blockStartSec: args.startSec,
        blockEndSec: args.endSec,
        targetSec: snapped,
        neighbor:
            args.neighborStartSec != null && args.neighborEndSec != null
                ? {
                      startSec: args.neighborStartSec,
                      endSec: args.neighborEndSec,
                  }
                : null,
        minBlockSec: minSec,
    });
    return args.edge === "left" ? plan.block.startSec : plan.block.endSec;
}

/** 块缩到最小宽度以下时应直接删除；预览 / 提交共用同一判定。 */
export function shouldRemoveResizedParamBlock(args: {
    readonly pointerSec: number;
    readonly startSec: number;
    readonly endSec: number;
    readonly edge: "left" | "right";
    readonly minBlockSec: number;
}): boolean {
    const minSec = Math.max(1e-6, finite(args.minBlockSec, MIN_AUTO_BLOCK_SEC));
    if (args.edge === "right") return args.pointerSec <= args.startSec + minSec + 1e-9;
    return args.pointerSec >= args.endSec - minSec - 1e-9;
}

/**
 * 块边缘延展时，相邻块被压缩后的**剩余可见区间**。
 *
 * 以前的逻辑把右边缘一起改到新起点：延展左块得到 `[0,1.5]`，右块却成了
 * `[1.5,2]`，用户直觉上会认为“后一个块被缩短了”，这是对的；但当左块只延长
 * 一点点、右块几乎是紧邻时，必须保证它仍有最小宽度。返回 null 表示压缩后
 * 不可见，应直接隐藏而不是生成零宽块。
 */
export function compressedNeighborSpan(args: {
    readonly edge: "left" | "right";
    readonly neighborStartSec: number;
    readonly neighborEndSec: number;
    readonly resizedEdgeSec: number;
    readonly minBlockSec: number;
}): { readonly startSec: number; readonly endSec: number } | null {
    const minSec = Math.max(1e-6, finite(args.minBlockSec, MIN_AUTO_BLOCK_SEC));
    /*
     * 右扩压缩右邻居**起点**，左扩压缩左邻居**终点**；另一侧始终不动。
     * 预览必须和提交用同一份“邻居剩余区间”，否则拖动时看不到缩短，
     * 松手后才突然跳/消失。
     */
    const sourceStart = Math.min(args.neighborStartSec, args.neighborEndSec);
    const sourceEnd = Math.max(args.neighborStartSec, args.neighborEndSec);
    // 邻居只剩最小宽度时仍应可见；只有完全退化才隐藏。
    if (sourceEnd - sourceStart < minSec) return null;
    if (args.edge === "right") {
        const startSec = Math.min(sourceEnd - minSec, Math.max(args.resizedEdgeSec, sourceStart));
        return { startSec, endSec: sourceEnd };
    }
    const endSec = Math.max(sourceStart + minSec, Math.min(args.resizedEdgeSec, sourceEnd));
    return { startSec: sourceStart, endSec };
}

/**
 * 伸缩某一侧时需要逐个处理的块。
 *
 * 不再只取“紧贴着的第一个邻居”：一次拖动可能跨过空隙并连续覆盖多个块。
 * 同一根轨道组内的块都参与，按拖动方向由近到远排序；具体哪些块被碰到，
 * 由 `resolveParamBlockResizePlan` 根据目标边界决定。
 */
export function paramBlockResizeCandidates<T extends ParamBlock>(
    blocks: readonly T[],
    block: T,
    edge: "left" | "right",
): T[] {
    return blocks
        .filter(
            (candidate) =>
                candidate.id !== block.id &&
                candidate.rootTrackId === block.rootTrackId &&
                (edge === "right"
                    ? candidate.startSec >= block.endSec - 1e-6
                    : candidate.endSec <= block.startSec + 1e-6),
        )
        .sort((a, b) => (edge === "right" ? a.startSec - b.startSec : b.endSec - a.endSec));
}

export interface ParamBlockResizeNeighborPlan {
    readonly id: string;
    readonly startSec: number;
    readonly endSec: number;
    /** 被拖块已经完整越过该块，松手时应删除它。 */
    readonly remove: boolean;
}

export interface ParamBlockResizePlan {
    readonly lineLayout?: readonly ParamLineBlock[];
    readonly lineFramePeriodMs?: number;
    /** 被拖块的新区间。 */
    readonly block: { readonly startSec: number; readonly endSec: number };
    /** 被部分缩短或完整覆盖的右侧 / 左侧块，按拖动方向排列。 */
    readonly neighbors: readonly ParamBlockResizeNeighborPlan[];
    /** 当前块已缩到最小宽度以下，松手时应删除。 */
    readonly removeBlock: boolean;
}

/** 伸缩预览的一帧几何：每帧都从手势起点重新投影，不沿用上一帧结果。 */
export interface ParamBlockResizeDisplay {
    readonly id: string;
    readonly startSec: number;
    readonly endSec: number;
    readonly selected: boolean;
}

/**
 * 把伸缩计划投影成**完整**的显示块列表。
 *
 * @param baseBlocks 手势起点的全部块几何（冻结快照），必须包含当前块与
 *   所有会被压缩的候选块。绝不能用上一帧的 live 列表：某帧块被移除后，
 *   后续再拉回来就找不到源块，预览会永久缺失直到松手。
 */
export function projectParamBlockResizeDisplay(args: {
    readonly baseBlocks: readonly {
        readonly id: string;
        readonly startSec: number;
        readonly endSec: number;
        readonly selected?: boolean;
    }[];
    readonly blockId: string;
    readonly plan: ParamBlockResizePlan;
}): ParamBlockResizeDisplay[] {
    if (args.plan.lineLayout && args.plan.lineFramePeriodMs)
        return args.plan.lineLayout.map((block) => ({
            id: block.id,
            startSec: (block.startFrame * args.plan.lineFramePeriodMs!) / 1000,
            endSec: (block.endFrameExclusive * args.plan.lineFramePeriodMs!) / 1000,
            selected: block.id === args.blockId,
        }));
    const neighborById = new Map(args.plan.neighbors.map((item) => [item.id, item]));
    const out: ParamBlockResizeDisplay[] = [];
    for (const block of args.baseBlocks) {
        if (block.id === args.blockId) {
            if (args.plan.removeBlock) continue;
            out.push({
                id: block.id,
                startSec: args.plan.block.startSec,
                endSec: args.plan.block.endSec,
                selected: true,
            });
            continue;
        }
        const neighbor = neighborById.get(block.id);
        if (neighbor !== undefined) {
            if (neighbor.remove) continue;
            out.push({
                id: block.id,
                startSec: neighbor.startSec,
                endSec: neighbor.endSec,
                selected: false,
            });
            continue;
        }
        out.push({
            id: block.id,
            startSec: block.startSec,
            endSec: block.endSec,
            selected: block.selected === true,
        });
    }
    return out;
}

/**
 * 纵向块拖动提交后的显示交接是否结束。
 *
 * 松手时 live 曲线已是最终值，但 `paramView` 与拖动快照的释放顺序若早一帧，
 * 宿主会先按旧主簇画一次，表现为块闪回原位。直到曲线窗口真正包含提交值，
 * 才允许释放这次交接；超时必须强制结束，避免异常路径永久锁定。
 */
export function canReleaseParamBlockDragCommit(args: {
    readonly committed: boolean;
    readonly startedAt: number;
    readonly now: number;
    readonly curveSettled: boolean;
    /**
     * 当前窗口是否已经包含提交后的曲线值。
     *
     * 本地回显可能复用同一个 `paramView.key`，因此不能只看 key 是否变化。
     * 只要新值已进入窗口，就可以安全撤掉拖动快照。
     */
    readonly valuesApplied?: boolean;
    readonly timeoutMs?: number;
}): boolean {
    if (!args.committed) return false;
    const timeoutMs = Math.max(1, args.timeoutMs ?? 1200);
    return (
        args.curveSettled || args.valuesApplied === true || args.now - args.startedAt >= timeoutMs
    );
}

export type ParamBlockDragCommitReleaseState =
    "waiting-for-curve" | "waiting-for-paint" | "released";

/**
 * 拖动交接的两阶段释放。
 *
 * 数据就绪时当前 React 渲染仍必须使用拖动快照；等这个快照完成一次绘制后，
 * 下一个绘制时机才撤下它并让正常派生接替。跳过 `waiting-for-paint` 会在一帧里
 * 先按旧主簇画回原位，再跳到提交位置，也就是用户看到的闪烁。
 */
export function advanceParamBlockDragCommitRelease(args: {
    readonly releaseState: ParamBlockDragCommitReleaseState;
    readonly committed: boolean;
    readonly startedAt: number;
    readonly now: number;
    readonly curveSettled: boolean;
    readonly valuesApplied?: boolean;
    readonly timeoutMs?: number;
}): ParamBlockDragCommitReleaseState {
    if (args.releaseState === "released") return "released";
    if (args.releaseState === "waiting-for-paint") return "released";
    return canReleaseParamBlockDragCommit(args) ? "waiting-for-paint" : "waiting-for-curve";
}

/** 拖动窗口在提交交接完成前始终沿用起点快照，即使异步取数换了窗口 key。 */
export function shouldUseParamBlockDragSnapshotForRender(args: {
    readonly snapshotKey: string;
    readonly paramViewKey: string | null;
    /** A live gesture owns its frozen window even if a background fetch rekeys paramView. */
    readonly gestureActive: boolean;
    readonly commitHeld: boolean;
}): boolean {
    return args.gestureActive || args.commitHeld || args.paramViewKey === args.snapshotKey;
}

/**
 * 纵向块拖动松手时锁存的**最终显示中心**。
 *
 * 普通绘制会把 pitch 中心吸附到最近半音行心；若交接锁存未吸附的 60.4，
 * 松手后就会先显示 60.4、再跳到 60。自由拖动 / Alt 缩放必须保持连续。
 */
export function resolveParamBlockCommitDisplayCenter(args: {
    readonly rawCenter: number | null;
    readonly param: string;
    readonly continuous: boolean;
}): number | null {
    if (args.rawCenter === null || !Number.isFinite(args.rawCenter)) return null;
    if (args.param !== "pitch" || args.continuous) return args.rawCenter;
    return Math.round(args.rawCenter);
}

/**
 * 普通块平移必须继续读取实时曲线中心；只有 Alt 音高缩放才允许锁住中心，
 * 因为缩放会改变块内起伏而不改变块应有的代表音高。
 */
export function shouldKeepParamBlockCenterLockedAfterDrag(args: {
    readonly pitchScale: boolean;
}): boolean {
    return args.pitchScale;
}

/**
 * 边缘伸缩的唯一几何来源。
 *
 * 预览、拖动中的宿主刷新和松手提交都消费这一份计划，避免“拖动时看见 A，
 * 松手后变成 B”。右边缘只改右端，左边缘只改左端；邻块被压缩时使用同一个
 * 端点，严格首尾相接，不留下会被后续重建误判成重叠的浮点缝。
 */
export function resolveParamBlockResizePlan(args: {
    readonly lineLayout?: ParamLineBlockSet;
    readonly blockId?: string;
    readonly edge: "left" | "right";
    readonly blockStartSec: number;
    readonly blockEndSec: number;
    readonly targetSec: number;
    /** 同方向全部候选块；会按目标区间筛出真正受影响的块。 */
    readonly neighbors?: readonly (
        | ParamBlock
        | {
              readonly id?: string;
              readonly startSec: number;
              readonly endSec: number;
          }
    )[];
    /** 兼容单邻居调用。 */
    readonly neighbor?:
        | {
              readonly id?: string;
              readonly startSec: number;
              readonly endSec: number;
          }
        | ParamBlock
        | null;
    readonly minBlockSec: number;
}): ParamBlockResizePlan {
    if (args.lineLayout && args.blockId) {
        const fp = args.lineLayout.framePeriodMs;
        const start = Math.max(
            0,
            Math.round(((args.edge === "left" ? args.targetSec : args.blockStartSec) * 1000) / fp),
        );
        const end = Math.max(
            0,
            Math.round(((args.edge === "right" ? args.targetSec : args.blockEndSec) * 1000) / fp),
        );
        const removeBlock = end <= start;
        const after = removeBlock
            ? args.lineLayout.blocks.filter((block) => block.id !== args.blockId)
            : resizeParamLineBlock(args.lineLayout.blocks, args.blockId, start, end);
        return {
            block: { startSec: (start * fp) / 1000, endSec: (end * fp) / 1000 },
            neighbors: [],
            removeBlock,
            lineLayout: after,
            lineFramePeriodMs: fp,
        };
    }
    const minSec = Math.max(1e-6, finite(args.minBlockSec, MIN_AUTO_BLOCK_SEC));
    const blockStart = Math.min(args.blockStartSec, args.blockEndSec);
    const blockEnd = Math.max(args.blockStartSec, args.blockEndSec);
    const target = finite(args.targetSec, args.edge === "left" ? blockStart : blockEnd);
    const candidates = [
        ...(args.neighbors ?? []),
        ...(args.neighbor === null || args.neighbor === undefined ? [] : [args.neighbor]),
    ]
        .filter(
            (candidate, index, all) =>
                all.findIndex(
                    (other) =>
                        "id" in other &&
                        "id" in candidate &&
                        other.id !== undefined &&
                        other.id === candidate.id,
                ) === index,
        )
        .filter((candidate) =>
            args.edge === "right"
                ? candidate.startSec >= blockEnd - 1e-6
                : candidate.endSec <= blockStart + 1e-6,
        )
        .sort((a, b) => (args.edge === "right" ? a.startSec - b.startSec : b.endSec - a.endSec));
    if (args.edge === "right") {
        const lower = blockStart + minSec;
        // 右侧没有硬上限：拖过整块时可继续向前吞并后续块。
        const blockNextEnd = Math.max(lower, target);
        const removeBlock = blockNextEnd <= lower + 1e-9;
        const neighbors: ParamBlockResizeNeighborPlan[] = [];
        if (!removeBlock) {
            for (const candidate of candidates) {
                if (blockNextEnd <= candidate.startSec + 1e-9) continue;
                const removed =
                    blockNextEnd >= candidate.endSec - 1e-9 ||
                    candidate.endSec - blockNextEnd < minSec - 1e-9;
                neighbors.push({
                    id:
                        "id" in candidate && candidate.id !== undefined
                            ? candidate.id
                            : `neighbor:${candidate.startSec}`,
                    startSec: removed ? candidate.startSec : blockNextEnd,
                    endSec: candidate.endSec,
                    remove: removed,
                });
            }
        }
        return {
            block: { startSec: blockStart, endSec: blockNextEnd },
            neighbors,
            removeBlock,
        };
    }
    const upper = blockEnd - minSec;
    const blockNextStart = Math.min(upper, target);
    const removeBlock = blockNextStart >= upper - 1e-9;
    const neighbors: ParamBlockResizeNeighborPlan[] = [];
    if (!removeBlock) {
        for (const candidate of candidates) {
            if (blockNextStart >= candidate.endSec - 1e-9) continue;
            const removed =
                blockNextStart <= candidate.startSec + 1e-9 ||
                blockNextStart - candidate.startSec < minSec - 1e-9;
            neighbors.push({
                id:
                    "id" in candidate && candidate.id !== undefined
                        ? candidate.id
                        : `neighbor:${candidate.startSec}`,
                startSec: candidate.startSec,
                endSec: removed ? candidate.endSec : blockNextStart,
                remove: removed,
            });
        }
    }
    return {
        block: { startSec: blockNextStart, endSec: blockEnd },
        neighbors,
        removeBlock,
    };
}

/** 创建块：按指针横向位移得到吸附后的区间，粘贴到网格线而不是松手才对齐。 */
export function resolveParamBlockCreateSpan(args: {
    readonly framePeriodMs?: number;
    readonly anchorSec: number;
    readonly pointerSec: number;
    readonly snapToGrid: (sec: number) => number;
    readonly snapEnabled: boolean;
    readonly minBlockSec: number;
}): { readonly startSec: number; readonly endSec: number } {
    if (args.framePeriodMs) {
        const fp = args.framePeriodMs / 1000;
        const snap = (sec: number) =>
            Math.max(0, Math.round((args.snapEnabled ? args.snapToGrid(sec) : sec) / fp));
        const anchor = snap(args.anchorSec);
        let target = snap(args.pointerSec);
        if (target === anchor) target = anchor + 1;
        return { startSec: Math.min(anchor, target) * fp, endSec: Math.max(anchor, target) * fp };
    }
    const minSec = Math.max(1e-6, finite(args.minBlockSec, MIN_AUTO_BLOCK_SEC));
    const snap = (sec: number): number =>
        args.snapEnabled ? finite(args.snapToGrid(sec), sec) : sec;
    const anchor = snap(args.anchorSec);
    let target = snap(args.pointerSec);
    if (Math.abs(target - anchor) < minSec) {
        target = anchor + (target < anchor ? -minSec : minSec);
    }
    const startSec = Math.min(anchor, target);
    const endSec = Math.max(anchor, target);
    return { startSec, endSec };
}

/** 切开位置：只有光标靠近网格线时才吸附，否则保留精确时间。 */
export function resolveParamBlockSplitSec(args: {
    readonly sec: number;
    readonly bpm: number;
    readonly timeSignatureDenominator: number;
    readonly stepBeats: number;
    readonly pxPerSec: number;
    readonly thresholdPx: number;
    readonly snapToggleHeld?: boolean;
}): number {
    if (args.snapToggleHeld === true || !Number.isFinite(args.sec)) return args.sec;
    const bpm = Number.isFinite(args.bpm) && args.bpm > 0 ? args.bpm : 120;
    const denominator = Math.max(1, Math.round(args.timeSignatureDenominator) || 4);
    const stepBeats = Math.max(1e-9, Number(args.stepBeats) || 1);
    const gridSec = (60 / bpm) * stepBeats * (4 / denominator);
    const snapped = Math.round(args.sec / gridSec) * gridSec;
    const distancePx = Math.abs(snapped - args.sec) * Math.max(1, args.pxPerSec);
    return distancePx <= Math.max(0, args.thresholdPx) ? snapped : args.sec;
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
    /** 合并时各块的几何；用于切分 / 伸缩后仍按当前同级块重建。 */
    readonly blockRanges?: readonly {
        readonly id: string;
        readonly startSec: number;
        readonly endSec: number;
    }[];
    readonly clipIds?: readonly string[];
}

export interface ParamBlockState {
    readonly lineLayout?: ParamLineBlockSet;
    /** Transient edit basis for optimistic concurrency/replay, never persisted. */
    readonly lineEditBase?: ParamLineBlockSet;
    readonly lineRootTrackId?: string;
    readonly mergedBlocks: readonly MergedParamBlock[];
    /** 上一批被删除的块，供用户一键恢复；仅在会话内有效。 */
    readonly lastDeletedBlocks?: readonly PersistedParamBlockDeletion[];
    /**
     * 工程持久化的合并记录（v7）：成员用 Clip / Take / 音节稳定引用，
     * 不随自动数组序号变化。与旧会话 `mergedBlocks` 并存，派生时二者共同生效。
     */
    readonly persistedMerges?: readonly PersistedParamBlockMerge[];
    /**
     * 用户手动切分 / 伸缩后的块区间（工程 v7）。
     *
     * 记录的是**成员稳定引用 + 用户编辑后的工程秒区间**，不是数组下标。
     * 自动分析边界变化时，按源域成员校核；对不上的记录忽略，旧工程没有该
     * 字段时行为不变。
     */
    readonly persistedEdits?: readonly PersistedParamBlockEdit[];
    /**
     * 兼容旧设置项：重叠块现在**无条件**合并，此字段不再参与派生。
     * 保留字段只为旧会话 / 设置载荷可继续解析。
     */
    readonly mergeOverlapsActive?: boolean;
}

export const EMPTY_PARAM_BLOCK_STATE: ParamBlockState = {
    mergedBlocks: [],
    persistedMerges: [],
    persistedEdits: [],
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

/** 用户手动切分 / 伸缩后的块区间（工程级持久化）。 */
export interface PersistedParamBlockEdit {
    readonly id: string;
    readonly members: readonly {
        readonly clipId: string;
        readonly takeId: string;
        readonly syllableId: string;
        readonly sourceStartSec: number;
        readonly sourceEndSec: number;
    }[];
    /** 用户编辑后的区间（工程秒）。 */
    readonly startSec: number;
    readonly endSec: number;
    /** 编辑当时的成员工程秒区间；用于移动 / 拉伸 Clip 后恢复切分位置。 */
    readonly memberStartSec?: number;
    readonly memberEndSec?: number;
    /**
     * 独立编辑块（新切分 / 伸缩 / 新增）。
     *
     * 旧工程的编辑段没有这个字段，仍走旧的父自动块兼容路径；新记录不再
     * 通过 `blockIds` 建立父子关系，左右块也不共享成员。
     */
    readonly explicit?: boolean;
    /**
     * 编辑所作用的自动块 id（稳定）。
     *
     * 切分后的左右段共享成员引用，不能再按“当前区间与自动块重叠”判断占用；
     * 保存原自动块 id 才能在切分后只吞掉被切的那个块，后续音节块继续显示。
     * 旧工程无此字段时走旧的区间兜底。
     */
    readonly blockIds?: readonly string[];
    /**
     * 该记录来自已有自动块 / 手动块被覆盖后的“让位压缩”。
     *
     * 没有它时，创建新块产生的左段会与旧的手动块共用同一成员身份，
     * materialize 会把两者当成同一个块并吞掉新块。
     */
    readonly overlapTrim?: boolean;
    /** 用户从块编辑视图删除该段；保留曲线与源音符。 */
    readonly deleted?: boolean;
    /**
     * 用户手动新增的块。没有自动音节成员，因此它的身份完全来自这条编辑记录；
     * 新增块不参与自动音节合并的“成员覆盖”判定。
     */
    readonly manual?: boolean;
    /** 手动新增块的高度锚点（当前参数值；不是曲线副本）。 */
    readonly value?: number | null;
    /** 块间圆滑宽度（秒）；只作用于当前块与其左右相接块。 */
    readonly transitionSec?: number;
}

/**
 * 被用户删除的块。
 *
 * 删除不是从音频里移除音符，而是**从参数块编辑视图隐藏**它；因此不能改写
 * 源域音节分析，也不能删除参数曲线。记录成员稳定引用后，自动分析重跑仍能
 * 把同一批块继续隐藏，直到用户显式恢复。
 */
export interface PersistedParamBlockDeletion {
    readonly id: string;
    readonly members: readonly {
        readonly clipId: string;
        readonly takeId: string;
        readonly syllableId: string;
        readonly sourceStartSec: number;
        readonly sourceEndSec: number;
    }[];
    readonly blockIds?: readonly string[];
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

export type ParamBlockEditFailureReason = "not_found" | "too_short" | "no_members";

export type ParamBlockEditResult =
    | { readonly ok: true; readonly state: ParamBlockState; readonly blockId: string }
    | { readonly ok: false; readonly reason: ParamBlockEditFailureReason };

export interface ParamBlockSplitInput {
    readonly state: ParamBlockState;
    readonly block: ParamBlock;
    /** 切分点（工程秒）。 */
    readonly sec: number;
    /** 可选 Clip 几何；用于给两个独立块保存同一容器锚点。 */
    readonly clip?: ParamBlockSourceClip;
}

export interface ParamBlockResizeInput {
    readonly state: ParamBlockState;
    readonly block: ParamBlock;
    readonly edge: "left" | "right";
    /** 拖到的目标边缘位置（工程秒）。右边缘用 startSec，左边缘用 endSec。 */
    readonly targetSec: number;
    /** 由 `resolveParamBlockResizePlan` 解析出的目标块区间；优先于按边缘单边推导。 */
    readonly targetSpan?: {
        readonly startSec: number;
        readonly endSec: number;
    } | null;
    /** 邻居应被压缩到的精确区间；与预览共用，不能在这里另算一套。 */
    readonly neighborSpan?: {
        readonly startSec: number;
        readonly endSec: number;
    } | null;
    /** 共享多块计划中受影响块的目标区间；`remove=true` 表示完整覆盖。 */
    readonly neighborSpans?: readonly {
        readonly id: string;
        readonly startSec: number;
        readonly endSec: number;
        readonly remove?: boolean;
    }[];
    /** 与 `neighborSpans` 对应的完整块（用于成员与锚点）；不必包含未受影响块。 */
    readonly neighborBlocks?: readonly ParamBlock[];
    readonly neighborStartSec?: number | null;
    readonly neighborEndSec?: number | null;
    /** 相邻块；延展跨越它时同步压缩它，保持首尾相接且不重叠。 */
    readonly neighbor?: ParamBlock | null;
    readonly minBlockSec?: number;
}

/** 用户在块层手动画出的新块（横向范围 + 当前参数预览值）。 */
export interface ParamBlockCreateInput {
    readonly state: ParamBlockState;
    readonly clip?: ParamBlockSourceClip;
    /** All source Clips intersected by the preview's root-track interval. */
    readonly clips?: readonly ParamBlockSourceClip[];
    readonly startSec: number;
    readonly endSec: number;
    readonly value: number | null;
    /** 可选：指定新块自己的编辑记录 id，便于创建后立即继续同一块手势。 */
    readonly blockId?: string;
    /** 当前块层的既有块；新块会按“后做意图”覆盖它们。 */
    readonly existingBlocks?: readonly ParamBlock[];
    readonly minBlockSec?: number;
}

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
export const PARAM_BLOCK_ROW_HEIGHT_MULTIPLIER = 1.2;
/**
 * 块间圆滑手柄的几何（CSS px）。
 *
 * 手柄位于右边界**正上方**：圆点中心与右边界对齐，只向上抬到块外；
 * 横向拖动时向左增大平滑、向右减小平滑。命中区略大于视觉圆点，
 * 但不侵占右边缘正中的缩放热区。
 */
export const PARAM_BLOCK_TRANSITION_HANDLE_DOT_RADIUS_PX = 3.5;
export const PARAM_BLOCK_TRANSITION_HANDLE_OUTSET_X_PX = 0;
export const PARAM_BLOCK_TRANSITION_HANDLE_OUTSET_Y_PX = 6;
export const PARAM_BLOCK_TRANSITION_HANDLE_LINE_LENGTH_PX = 8;

/** Keep the grip above both ends of its curve, including at a small zoom. */
export function paramBlockTransitionHandleY(
    leftTopPx: number,
    rightTopPx?: number,
    outsetY = PARAM_BLOCK_TRANSITION_HANDLE_OUTSET_Y_PX,
): number {
    return Math.min(leftTopPx, rightTopPx ?? leftTopPx) - outsetY;
}
/** 兼容旧调用点：历史命名，语义已改为“距块顶的上抬量”。 */
export const PARAM_BLOCK_TRANSITION_HANDLE_INSET_Y_PX = PARAM_BLOCK_TRANSITION_HANDLE_OUTSET_Y_PX;

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
 * 1. 显式 `syllableSpansSec`（mock / 分析结果）；
 * 2. 显式 mock 音节；
 * 3. 其余音频 Clip 尚无分析结果时返回空集，不伪造“整段一块”。
 *
 * 显式音节会先裁剪到 Clip 范围、去掉零宽段。
 * 分析得到的边界是权威，不再按固定时长机械等分。
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
        /*
         * 音节边界之间可能有极小的浮点缝（分析窗 hop、源域映射都会留下几毫秒）。
         * 视觉上它们应是一串首尾相接的 MIDI 音符，不能被渲染成“人为间隙”。
         * 只消除小于 20ms 的缝，真实停顿 / 休止仍保留。
         */
        for (let index = 1; index < spans.length; index += 1) {
            const previous = spans[index - 1];
            const current = spans[index];
            if (
                previous !== undefined &&
                current !== undefined &&
                current.startSec > previous.endSec &&
                current.startSec - previous.endSec < 0.02
            ) {
                previous.endSec = current.startSec;
            }
        }
        /*
         * 【分析前必须为空】音频 Clip 没有音节分析时，前端不知道发音边界。
         * 以前这里回退成 `[member]`，用户会在分析完成前先看到一个覆盖整段
         * Clip 的假块；它既不是真实音节，也会让切分 / 伸缩记录与随后的分析
         * 结果错位。MIDI Clip 已在上面提供精确音节，不经过这里。
         */
        /*
         * 源域分析结果可能只出现在 `sourceSyllableSpansSec`，而工程秒
         * `syllableSpansSec` 由调用方投影后才传；不能因为前者在这一步还没
         * 展开，就把整条 Clip 判成“尚未分析”。
         */
        const hasSourceAnalysis = (clip.sourceSyllableSpansSec?.length ?? 0) > 0;
        if (spans.length === 0 && !hasSourceAnalysis && clip.mockSyllables !== true) continue;
        const effective =
            spans.length > 0
                ? // 模型/分析给出的音节边界就是权威；不再按 1.4 秒机械等分，
                  // 否则长元音会被切成一堆与发音无关的块。
                  spans
                : hasSourceAnalysis
                  ? []
                  : mockSyllableSpans(member);

        /*
         * 【源域数组存在时，工程秒只用于显示】`syllableSpansSec` 是已投影结果，
         * 可能被视觉层的"首尾相接合并"压成更少的区间；不能再按它的下标去取
         * `sourceSyllableSpansSec`。用源域数组自投射得到一一对应的工程区间，
         * 这是块身份与持久化合并的共同基准。
         *
         * 【逐段投影】不能只投第一批再看长度：`projectSourceSpansToTimeline`
         * 只返回**当前可见窗口内**的区间；Clip 从 1.5s 开始、源域从 0s 起时，
         * 第一段会被裁掉，整批结果可能为空，而后续段其实仍可见。
         */
        const sourceSpans = clip.sourceSyllableSpansSec ?? [];
        const projectedFromSource: Array<{
            startSec: number;
            endSec: number;
            sourceIndex: number;
        }> = [];
        for (let sourceIndex = 0; sourceIndex < sourceSpans.length; sourceIndex += 1) {
            const [sourceStartSec, sourceEndSec] = sourceSpans[sourceIndex] ?? [];
            if (!(sourceEndSec > sourceStartSec)) continue;
            const projected = projectMidiNoteSpansToTimeline({
                ...clip,
                midiNoteData: [{ startSec: sourceStartSec, endSec: sourceEndSec, note: 60 }],
            })[0];
            if (projected === undefined || !(projected.endSec > projected.startSec)) continue;
            projectedFromSource.push({ ...projected, sourceIndex });
        }
        const effectiveSpans =
            projectedFromSource.length > 0
                ? projectedFromSource
                : effective.map((span, index) => ({ ...span, sourceIndex: index }));
        effectiveSpans.sort((a, b) => a.startSec - b.startSec);
        for (let index = 0; index < effectiveSpans.length; index += 1) {
            const span = effectiveSpans[index];
            const hasSourceSpan =
                clip.sourceSyllableSpansSec !== undefined && clip.sourceSyllableSpansSec.length > 0;
            const sourceSpan = hasSourceSpan
                ? clip.sourceSyllableSpansSec?.[span.sourceIndex]
                : undefined;
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
            const blockIdSourceIndex =
                projectedFromSource.length > 0
                    ? `${hasSourceSpan ? "source" : "timeline"}:${sourceStart.toFixed(3)}:${sourceEnd.toFixed(3)}:${span.startSec.toFixed(3)}:${span.endSec.toFixed(3)}`
                    : `${span.sourceIndex}`;
            blocks.push({
                id: `auto:${clip.id}:${blockIdSourceIndex}`,
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
                anchorStartSec: member.startSec,
                anchorEndSec: member.endSec,
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

/** 在同一容器的可见块里，找与 block 首尾相接的左右邻居。 */
export function adjacentParamBlocks(
    blocks: readonly ParamBlock[],
    block: ParamBlock,
): { left: ParamBlock | null; right: ParamBlock | null } {
    const clipId = block.members[0]?.clipId ?? "";
    const sameContainer = (candidate: ParamBlock) =>
        candidate.id !== block.id && candidate.members.some((member) => member.clipId === clipId);
    const left =
        blocks
            .filter(
                (candidate) =>
                    sameContainer(candidate) && Math.abs(candidate.endSec - block.startSec) <= 1e-6,
            )
            .sort((a, b) => b.endSec - a.endSec)[0] ?? null;
    const right =
        blocks
            .filter(
                (candidate) =>
                    sameContainer(candidate) && Math.abs(candidate.startSec - block.endSec) <= 1e-6,
            )
            .sort((a, b) => a.startSec - b.startSec)[0] ?? null;
    return { left, right };
}

/** 块间过渡手柄的命中区（秒）；只对相接块返回。 */
export function paramBlockTransitionHandles(
    blocks: readonly ParamBlock[],
): Array<{ leftId: string; rightId: string; sec: number; transitionSec: number }> {
    const handles: Array<{
        leftId: string;
        rightId: string;
        sec: number;
        transitionSec: number;
    }> = [];
    const sorted = sortParamBlocksByTime(blocks);
    for (let index = 0; index < sorted.length - 1; index += 1) {
        const block = sorted[index];
        const right = sorted[index + 1];
        if (block === undefined || right === undefined) continue;
        if (block.members[0]?.clipId !== right.members[0]?.clipId) continue;
        if (Math.abs(block.endSec - right.startSec) > 1e-6) continue;
        const sec = block.endSec;
        if (handles.some((handle) => Math.abs(handle.sec - sec) <= 1e-6)) continue;
        handles.push({
            leftId: block.id,
            rightId: right.id,
            sec,
            // 相接关系的宽度只属于“左块 → 右块”这一条边；从右块读会让
            // 右块自己的右边界宽度串回来，表现为拖一个手柄影响整排。
            transitionSec: Math.max(0, block.transitionSec ?? 0),
        });
    }
    return handles;
}

/** 按稳定时间顺序生成手柄；不受 React 派生数组的临时顺序影响。 */
export function sortedParamBlockTransitionHandles(
    blocks: readonly ParamBlock[],
): Array<{ leftId: string; rightId: string; sec: number; transitionSec: number }> {
    return paramBlockTransitionHandles(blocks);
}

/** 时间轴顺序的单一来源；命中、手柄与显示都用这个顺序。 */
export function sortParamBlocksByTime<
    T extends {
        readonly id: string;
        readonly startSec: number;
        readonly endSec: number;
    },
>(blocks: readonly T[]): T[] {
    return blocks.slice().sort((a, b) => {
        const byStart = a.startSec - b.startSec;
        if (Math.abs(byStart) > 1e-9) return byStart;
        const byEnd = a.endSec - b.endSec;
        if (Math.abs(byEnd) > 1e-9) return byEnd;
        return a.id < b.id ? -1 : a.id > b.id ? 1 : 0;
    });
}

/** 两个块是否在时间上首尾相接或重叠。 */
export function blocksAreAdjacent(a: ParamBlock, b: ParamBlock): boolean {
    return a.startSec <= b.endSec + 1e-6 && b.startSec <= a.endSec + 1e-6;
}

/** 是否已存在把这组 Clip 全部覆盖的合并块。 */
function hasSameMergedBlock(
    state: ParamBlockState,
    blockIds: readonly string[],
    mergedId: string,
): boolean {
    const wanted = new Set(blockIds);
    return state.mergedBlocks.some(
        (merged) =>
            merged.id === mergedId ||
            (merged.blockIds !== undefined &&
                merged.blockIds.length === wanted.size &&
                merged.blockIds.every((id) => wanted.has(id))),
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
    if (args.state.lineLayout) return mergeLineBlocks(args.state, args.selectedBlockIds);
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

    /*
     * 合并 id 必须由**成员稳定引用**生成，不能由当前数组顺序或当前派生 id 拼成。
     * 拖动提交后自动块可能重新排序；若 id 跟着变，选中态、上下文菜单与持久化
     * 记录会在两套块集合之间来回切，表现为块在两个位置闪烁。
     */
    const stableIds = ordered
        .flatMap((block) => block.members.map((member) => member.syllableId ?? member.clipId))
        .sort();
    /*
     * 合并组保存的是**块当前几何**，而不只是 id。
     * 切分 / 伸缩后块 id 可能保持稳定但区间已变；只按 id 重新派生会退回
     * 旧自动块范围，导致合并块显示错误或把邻居吞掉。
     */
    const blockIds = ordered.map((block) => block.id);
    const blockRanges = ordered.map((block) => ({
        id: block.id,
        startSec: block.startSec,
        endSec: block.endSec,
    }));
    const mergedId = overlapGroupId(stableIds);
    if (hasSameMergedBlock(args.state, blockIds, mergedId)) {
        return { ok: false, reason: "already_merged" };
    }
    return {
        ok: true,
        state: {
            ...args.state,
            mergedBlocks: [
                ...args.state.mergedBlocks,
                {
                    id: mergedId,
                    blockIds,
                    blockRanges,
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
        persistedEdits: (state.persistedEdits ?? []).filter(
            (edit) =>
                !wanted.has(edit.id) &&
                !edit.members.some((member) => wantedSyllables.has(member.syllableId)),
        ),
    };
}

function blockMemberRefs(block: ParamBlock): PersistedParamBlockEdit["members"] | null {
    const members = block.members
        .filter(
            (
                member,
            ): member is typeof member & {
                takeId: string;
                syllableId: string;
            } =>
                member.takeId !== undefined &&
                member.syllableId !== undefined &&
                (member.sourceStartSec !== undefined ||
                    member.sourceStartForMatchSec !== undefined) &&
                (member.sourceEndSec !== undefined || member.sourceEndForMatchSec !== undefined),
        )
        .map((member) => ({
            clipId: member.clipId,
            takeId: member.takeId,
            syllableId: member.syllableId,
            sourceStartSec:
                member.sourceStartForMatchSec ?? member.sourceStartSec ?? member.startSec,
            sourceEndSec: member.sourceEndForMatchSec ?? member.sourceEndSec ?? member.endSec,
        }));
    return members.length > 0 ? members : null;
}

function replaceBlockEdits(
    state: ParamBlockState,
    replacedBlockId: string,
    nextEdits: readonly PersistedParamBlockEdit[],
): ParamBlockState {
    const replaced = new Set([replacedBlockId, ...nextEdits.map((edit) => edit.id)]);
    return {
        ...state,
        persistedEdits: [
            ...(state.persistedEdits ?? []).filter((edit) => !replaced.has(edit.id)),
            ...nextEdits,
        ],
        persistedMerges: (state.persistedMerges ?? []).filter((record) => !replaced.has(record.id)),
        mergedBlocks: state.mergedBlocks.filter((record) => !replaced.has(record.id)),
    };
}

/** 手动块与切分 / 伸缩共用 edits；后两者先按精确几何移除旧的手动记录。 */
function removeManualBlockAt(
    state: ParamBlockState,
    clipId: string,
    startSec: number,
    endSec: number,
): ParamBlockState {
    return {
        ...state,
        persistedEdits: (state.persistedEdits ?? []).filter(
            (edit) =>
                !(
                    edit.manual === true &&
                    edit.members[0]?.clipId === clipId &&
                    Math.abs(edit.startSec - startSec) <= 1e-6 &&
                    Math.abs(edit.endSec - endSec) <= 1e-6
                ),
        ),
    };
}

/**
 * 在指定秒位置切块。
 *
 * 产生两个稳定的用户编辑段：左段 `[start, sec)`、右段 `[sec, end)`。它们共享
 * 原块成员引用，因此继续参与原有合并记录校核；原合并组在本操作中被消耗，
 * 新段作为独立块存在。块太窄或切分点不在块内时拒绝，避免误点产生细到
 * 点不中的碎片。
 */
export function splitParamBlock(args: ParamBlockSplitInput): ParamBlockEditResult {
    if (args.state.lineLayout) return splitLineBlock(args.state, args.block, args.sec);
    const block = args.block;
    const sec = args.sec;
    const epsilon = 1e-9;
    const validSplit =
        Number.isFinite(args.sec) && sec > block.startSec + epsilon && block.endSec > sec + epsilon;
    if (!validSplit) {
        return { ok: false, reason: "too_short" };
    }
    const sourceMembers = blockMemberRefs(block);
    if (sourceMembers === null) return { ok: false, reason: "no_members" };
    const leftId = `edit:${block.id}:left:${sec.toFixed(9)}`;
    const rightId = `edit:${block.id}:right:${sec.toFixed(9)}`;
    /*
     * 左右段是**同级独立块**：不引用父块 id，也不共享成员。
     * 源域区间只在各自子块内部按切分比例拆开；之后任何拖动 / 伸缩都只读写
     * 当前块的成员，不会触碰另一块。
     */
    const splitMembers = (segStartSec: number, segEndSec: number): typeof sourceMembers | null => {
        const out: PersistedParamBlockEdit["members"][number][] = [];
        for (const member of sourceMembers) {
            const spanStart = block.startSec;
            const span = Math.max(1e-9, block.endSec - block.startSec);
            const startT = Math.max(0, Math.min(1, (segStartSec - spanStart) / span));
            const endT = Math.max(0, Math.min(1, (segEndSec - spanStart) / span));
            const sourceStart = member.sourceStartSec;
            const sourceEnd = member.sourceEndSec;
            const sourceSpan = Math.max(0, sourceEnd - sourceStart);
            const segSourceStart = sourceStart + sourceSpan * startT;
            const segSourceEnd = sourceStart + sourceSpan * endT;
            if (!(segSourceEnd > segSourceStart + 1e-12)) continue;
            out.push({
                ...member,
                syllableId: `${member.syllableId}:${segStartSec.toFixed(9)}:${segEndSec.toFixed(9)}`,
                sourceStartSec: segSourceStart,
                sourceEndSec: segSourceEnd,
            });
        }
        return out.length > 0 ? out : null;
    };
    const leftMembers = splitMembers(block.startSec, sec);
    const rightMembers = splitMembers(sec, block.endSec);
    if (leftMembers === null || rightMembers === null) {
        return { ok: false, reason: "too_short" };
    }
    /*
     * 锚点必须描述**整段容器**，不能取块起点。切分后左右块都保存同一个
     * 容器锚点；恢复时仅用各自记录区间映射，彼此互不引用。
     */
    const blockClip = args.clip ?? undefined;
    const anchorStartSec =
        blockClip === undefined
            ? (block.anchorStartSec ??
              block.members.reduce(
                  (min, member) => Math.min(min, member.sourceStartForMatchSec ?? member.startSec),
                  Number.POSITIVE_INFINITY,
              ))
            : clipSpan(blockClip).startSec;
    const anchorEndSec =
        blockClip === undefined
            ? (block.anchorEndSec ??
              block.members.reduce(
                  (max, member) => Math.max(max, member.sourceEndForMatchSec ?? member.endSec),
                  Number.NEGATIVE_INFINITY,
              ))
            : clipSpan(blockClip).endSec;
    const left: PersistedParamBlockEdit = {
        id: leftId,
        members: leftMembers,
        startSec: block.startSec,
        endSec: sec,
        memberStartSec: anchorStartSec,
        memberEndSec: anchorEndSec,
        explicit: true,
        // 仅用于隐藏被替换的自动块，不表达左右块之间的父子关系。
        blockIds: [block.id],
    };
    const right: PersistedParamBlockEdit = {
        id: rightId,
        members: rightMembers,
        startSec: sec,
        endSec: block.endSec,
        memberStartSec: anchorStartSec,
        memberEndSec: anchorEndSec,
        explicit: true,
        blockIds: [block.id],
    };
    const baseState = removeManualBlockAt(
        args.state,
        block.members[0]?.clipId ?? "",
        block.startSec,
        block.endSec,
    );
    return {
        ok: true,
        blockId: leftId,
        state: replaceBlockEdits(baseState, block.id, [left, right]),
    };
}

/**
 * 拖动块左 / 右边缘。
 *
 * 只修改被抓住的那一侧，另一侧保持不动；不能越过相邻块，也不能缩到点不中。
 * 与切分一样写入 `persistedEdits`，下一次 materialize 仍由稳定成员引用恢复。
 */
export function resizeParamBlock(args: ParamBlockResizeInput): ParamBlockEditResult {
    if (args.state.lineLayout)
        return resizeLineBlock(args.state, args.block, args.edge, args.targetSec, args.targetSpan);
    const block = args.block;
    const minSec = Math.max(MIN_AUTO_BLOCK_SEC, args.minBlockSec ?? 0.04);
    const nextStart =
        args.targetSpan?.startSec ?? (args.edge === "left" ? args.targetSec : block.startSec);
    const nextEnd =
        args.targetSpan?.endSec ?? (args.edge === "right" ? args.targetSec : block.endSec);
    if (!Number.isFinite(nextStart) || !Number.isFinite(nextEnd) || nextEnd - nextStart < minSec) {
        return { ok: false, reason: "too_short" };
    }
    const members = blockMemberRefs(block);
    if (members === null) return { ok: false, reason: "no_members" };
    const anchorStartSec = block.anchorStartSec ?? block.startSec;
    const anchorEndSec = block.anchorEndSec ?? block.endSec;
    const manualExisting =
        args.state.persistedEdits?.find(
            (edit) => edit.id === block.id && edit.manual === true && edit.deleted !== true,
        ) ?? null;
    let nextEdits: PersistedParamBlockEdit[] = [
        {
            id: manualExisting?.id ?? block.id,
            members: manualExisting?.members ?? members,
            startSec: nextStart,
            endSec: nextEnd,
            memberStartSec: anchorStartSec,
            memberEndSec: anchorEndSec,
            explicit: true,
            manual: manualExisting?.manual,
            value: manualExisting?.value,
        },
    ];
    const replacedIds = [block.id];
    /*
     * 相邻连续块联动压缩：右边缘向右延展时，后一个块起点随之右移；
     * 左边缘向左延展时，前一个块终点随之左移。只对首尾相接的块生效，
     * 中间隔着休止的块保持不动。若压缩后不足最小宽度则拒绝整次改边。
     */
    const plannedNeighbors =
        args.neighborSpans !== undefined && args.neighborSpans.length > 0
            ? args.neighborSpans
            : args.neighborSpan === undefined || args.neighborSpan === null
              ? []
              : [
                    {
                        id: args.neighbor?.id ?? "",
                        startSec: args.neighborSpan.startSec,
                        endSec: args.neighborSpan.endSec,
                        remove: false,
                    },
                ];
    const neighbor = args.neighbor;
    const neighborBlockById = new Map<string, ParamBlock>();
    if (neighbor !== undefined && neighbor !== null) {
        neighborBlockById.set(neighbor.id, neighbor);
    }
    for (const candidate of args.neighborBlocks ?? []) {
        neighborBlockById.set(candidate.id, candidate);
    }
    /*
     * 邻居只在真正被压缩时才需要拥有可持久化成员。老逻辑即使没有重叠也
     * 要求邻居成员，会让“只缩自己的右边”误报 no_members；而在共享计划
     * 存在时，邻块区间已经由计划明确给出，不需要再跑另一套压缩公式。
     */
    const shouldCompressNeighbor =
        plannedNeighbors.length > 0 ||
        (neighbor !== undefined &&
            neighbor !== null &&
            (args.edge === "right"
                ? nextEnd > neighbor.startSec + 1e-6
                : nextStart < neighbor.endSec - 1e-6));
    /*
     * 多块覆盖：一次伸缩可能压短 / 删除多个块。每块只写一条显式记录；
     * `remove` 的块写删除记录，否则会在重建时从自动块池重新出现。
     * 单邻居兼容路径仍使用原来的 `neighbor` 参数。
     */
    if (plannedNeighbors.length > 0) {
        for (const planned of plannedNeighbors) {
            const target = planned.id.length > 0 ? neighborBlockById.get(planned.id) : neighbor;
            if (target === undefined || target === null) continue;
            const neighborMembers = blockMemberRefs(target);
            if (neighborMembers === null) {
                return { ok: false, reason: "no_members" };
            }
            const neighborAnchorStart = target.anchorStartSec ?? target.startSec;
            const neighborAnchorEnd = target.anchorEndSec ?? target.endSec;
            nextEdits = [
                ...nextEdits,
                {
                    id: target.id,
                    members: neighborMembers,
                    startSec: planned.startSec,
                    endSec: planned.endSec,
                    memberStartSec: neighborAnchorStart,
                    memberEndSec: neighborAnchorEnd,
                    explicit: true,
                    deleted: planned.remove === true,
                    blockIds: [target.id],
                },
            ];
            replacedIds.push(target.id);
        }
    }
    if (
        plannedNeighbors.length === 0 &&
        neighbor !== undefined &&
        neighbor !== null &&
        shouldCompressNeighbor
    ) {
        if (args.edge === "right" && nextEnd > neighbor.startSec + 1e-6) {
            /*
             * 只压缩邻居起点，不允许靠扩展把它删掉。目标已经由
             * `resolveParamBlockResizeTarget` 钳在 neighborEnd − min；这里仍再校核
             * 一次，防止外部调用绕过 resolver 直接构造输入。
             */
            /*
             * 吸附模式下列边缘可能一步跨过整个邻居。提交端不能直接拒绝
             * 整次编辑（用户会看到“松手回弹”），也不能把邻居删掉；统一钳到
             * 邻居终点 − 最小宽度，与预览和 `resolveParamBlockResizeTarget`
             * 共用同一条几何约束。
             */
            /*
             * 有共享计划时直接使用它的邻块区间；`blockEnd` 也取同一端点，
             * 保证提交态严格相接。旧调用未传计划时才回退到本地压缩。
             */
            const compressedStart =
                args.neighborSpan?.startSec ??
                Math.min(neighbor.endSec - minSec, Math.max(nextEnd, neighbor.startSec));
            const compressedEnd = args.neighborSpan?.endSec ?? neighbor.endSec;
            const blockEnd = Math.min(nextEnd, compressedStart);
            if (blockEnd - nextStart < minSec) return { ok: false, reason: "too_short" };
            nextEdits[0] = { ...nextEdits[0], endSec: blockEnd };
            const neighborMembers = blockMemberRefs(neighbor);
            if (neighborMembers === null) return { ok: false, reason: "no_members" };
            const neighborAnchorStart = neighbor.anchorStartSec ?? neighbor.startSec;
            const neighborAnchorEnd = neighbor.anchorEndSec ?? neighbor.endSec;
            nextEdits = [
                ...nextEdits,
                {
                    id: neighbor.id,
                    members: neighborMembers,
                    startSec: compressedStart,
                    endSec: compressedEnd,
                    memberStartSec: neighborAnchorStart,
                    memberEndSec: neighborAnchorEnd,
                    explicit: true,
                },
            ];
            replacedIds.push(neighbor.id);
        } else if (args.edge === "left" && nextStart < neighbor.endSec - 1e-6) {
            const compressedEnd =
                args.neighborSpan?.endSec ??
                Math.max(neighbor.startSec + minSec, Math.min(nextStart, neighbor.endSec));
            const compressedStart = args.neighborSpan?.startSec ?? neighbor.startSec;
            const blockStart = Math.max(nextStart, compressedEnd);
            if (nextEnd - blockStart < minSec) return { ok: false, reason: "too_short" };
            nextEdits[0] = { ...nextEdits[0], startSec: blockStart };
            const neighborMembers = blockMemberRefs(neighbor);
            if (neighborMembers === null) return { ok: false, reason: "no_members" };
            const neighborAnchorStart = neighbor.anchorStartSec ?? neighbor.startSec;
            const neighborAnchorEnd = neighbor.anchorEndSec ?? neighbor.endSec;
            nextEdits = [
                ...nextEdits,
                {
                    id: neighbor.id,
                    members: neighborMembers,
                    startSec: compressedStart,
                    endSec: compressedEnd,
                    memberStartSec: neighborAnchorStart,
                    memberEndSec: neighborAnchorEnd,
                    explicit: true,
                },
            ];
            replacedIds.push(neighbor.id);
        }
    }
    let state = removeManualBlockAt(
        args.state,
        block.members[0]?.clipId ?? "",
        block.startSec,
        block.endSec,
    );
    if (neighbor !== undefined && neighbor !== null && shouldCompressNeighbor) {
        state = removeManualBlockAt(
            state,
            neighbor.members[0]?.clipId ?? "",
            neighbor.startSec,
            neighbor.endSec,
        );
    }
    for (const edit of nextEdits) {
        state = replaceBlockEdits(state, edit.id, []);
    }
    const removedNeighborIds = nextEdits
        .filter((edit) => edit.deleted === true)
        .map((edit) => edit.id);
    state = {
        ...state,
        persistedEdits: [
            ...(state.persistedEdits ?? []).filter(
                (edit) =>
                    !replacedIds.includes(edit.id) &&
                    !nextEdits.some((next) => next.id === edit.id) &&
                    !removedNeighborIds.includes(edit.id),
            ),
            ...nextEdits,
        ],
    };
    return {
        ok: true,
        blockId: block.id,
        state,
    };
}

/**
 * 在块层手动新增一个块。
 *
 * 这不是“改音频”：新增块只增加参数块的视图边界与高度锚点，曲线、音节分析
 * 和 Clip 几何都不改。它与切分 / 伸缩共用 `persistedEdits`，因此工程保存与
 * 重新打开后的身份规则一致。
 */
export function createParamBlock(args: ParamBlockCreateInput): ParamBlockEditResult {
    if (args.state.lineLayout)
        return createLineBlock(args.state, args.startSec, args.endSec, args.blockId);
    if (!args.clip) return { ok: false, reason: "no_members" };
    const minSec = Math.max(MIN_AUTO_BLOCK_SEC, args.minBlockSec ?? 0.04);
    const startSec = Math.min(args.startSec, args.endSec);
    const endSec = Math.max(args.startSec, args.endSec);
    if (!Number.isFinite(startSec) || !Number.isFinite(endSec) || endSec - startSec < minSec) {
        return { ok: false, reason: "too_short" };
    }
    const sources = (args.clips ?? [args.clip]).filter(
        (clip) => clip.startSec < endSec - 1e-9 && clip.startSec + clip.lengthSec > startSec + 1e-9,
    );
    if (sources.length === 0) return { ok: false, reason: "too_short" };
    const member = {
        startSec: Math.min(...sources.map((clip) => clipSpan(clip).startSec)),
        endSec: Math.max(...sources.map((clip) => clipSpan(clip).endSec)),
    };
    const clippedStart = Math.max(member.startSec, Math.min(member.endSec, startSec));
    const clippedEnd = Math.max(member.startSec, Math.min(member.endSec, endSec));
    if (clippedEnd - clippedStart < minSec) {
        return { ok: false, reason: "too_short" };
    }
    const members = sources.map((clip) => {
        const span = clipSpan(clip);
        const rate =
            Number.isFinite(clip.playbackRate) && (clip.playbackRate as number) > 1e-6
                ? (clip.playbackRate as number)
                : 1;
        const localStart = Math.max(clippedStart, span.startSec) - span.startSec;
        const localEnd = Math.min(clippedEnd, span.endSec) - span.startSec;
        const sourceBase = Math.max(0, clip.sourceStartSec ?? 0);
        const reverseEnd = clip.sourceEndSec ?? sourceBase + clip.lengthSec * rate;
        const sourceStart = clip.reversed
            ? reverseEnd - localEnd * rate
            : sourceBase + localStart * rate;
        const sourceEnd = clip.reversed
            ? reverseEnd - localStart * rate
            : sourceBase + localEnd * rate;
        const takeId = clip.activeTakeId ?? clip.takes?.[0]?.id ?? clip.takeId ?? clip.id;
        return {
            clipId: clip.id,
            takeId,
            syllableId: `${takeId}:manual:${sourceStart.toFixed(3)}:${sourceEnd.toFixed(3)}`,
            sourceStartSec: sourceStart,
            sourceEndSec: sourceEnd,
        };
    });
    const sourceStart = members[0].sourceStartSec;
    const sourceEnd = members[members.length - 1].sourceEndSec;
    /*
     * 新块压住的旧块整块让位，而不是拆成左右残段。
     *
     * 产品语义是“新建块覆盖旧块”，不是把旧块从中间挖掉：只要与新区间相交，
     * 旧块就应从预览和结果里消失。旧块若是用户原本就有的手动块，先写一条
     * 显式删除记录，保证保存重开时不会回弹。
     */
    const coveredEdits: PersistedParamBlockEdit[] = [];
    const coveredIds = new Set<string>();
    for (const existing of args.existingBlocks ?? []) {
        if (existing.startSec >= clippedEnd - 1e-9 || existing.endSec <= clippedStart + 1e-9) {
            continue;
        }
        const existingMembers = blockMemberRefs(existing);
        if (existingMembers === null) continue;
        const existingEdit = (args.state.persistedEdits ?? []).find(
            (edit) => edit.id === existing.id && edit.deleted !== true,
        );
        coveredIds.add(existing.id);
        coveredEdits.push({
            id: existingEdit?.id ?? `deleted-overlap:${existing.id}`,
            members: existingEdit?.members ?? existingMembers,
            startSec: existing.startSec,
            endSec: existing.endSec,
            memberStartSec: existing.anchorStartSec ?? existing.startSec,
            memberEndSec: existing.anchorEndSec ?? existing.endSec,
            explicit: true,
            blockIds: [existing.id],
            manual: existingEdit?.manual,
            value: existingEdit?.value,
            deleted: true,
        });
    }
    const edit: PersistedParamBlockEdit = {
        id:
            args.blockId ??
            `edit:manual:${args.clip.id}:${sourceStart.toFixed(3)}:${sourceEnd.toFixed(3)}`,
        members,
        startSec: clippedStart,
        endSec: clippedEnd,
        memberStartSec: member.startSec,
        memberEndSec: member.endSec,
        manual: true,
        explicit: true,
        value: args.value,
    };
    return {
        ok: true,
        blockId: edit.id,
        state: {
            ...args.state,
            persistedEdits: [
                ...(args.state.persistedEdits ?? []).filter((item) => !coveredIds.has(item.id)),
                ...coveredEdits,
                edit,
            ],
        },
    };
}

/**
 * Persist MIDI-note-like logical centers without changing block geometry.
 *
 * Auto blocks normally derive their y from curve samples. Once a user moves a
 * block, that explicit center becomes authoritative. Existing split/manual edit
 * records are updated in place; an unchanged auto block receives a
 * self-contained explicit record so the center survives a plugin/project reopen.
 */
export function setParamBlockLogicalCenters(args: {
    readonly state: ParamBlockState;
    readonly blocks: readonly ParamBlock[];
    readonly centers: ReadonlyMap<string, number>;
}): ParamBlockState {
    if (args.state.lineLayout) return args.state;
    const replacements = new Map<string, PersistedParamBlockEdit>();
    for (const block of args.blocks) {
        const center = args.centers.get(block.id);
        if (center === undefined || !Number.isFinite(center)) continue;
        const existing = (args.state.persistedEdits ?? []).find(
            (edit) =>
                edit.deleted !== true &&
                (edit.id === block.id || edit.blockIds?.includes(block.id) === true),
        );
        const members = existing?.members ?? blockMemberRefs(block);
        if (members === null) continue;
        const centerOnlyId = `center:${block.id}`;
        const id = existing?.id ?? centerOnlyId;
        replacements.set(id, {
            id,
            members,
            startSec: existing?.startSec ?? block.startSec,
            endSec: existing?.endSec ?? block.endSec,
            memberStartSec: existing?.memberStartSec ?? block.anchorStartSec ?? block.startSec,
            memberEndSec: existing?.memberEndSec ?? block.anchorEndSec ?? block.endSec,
            explicit: existing?.explicit ?? false,
            blockIds: existing?.blockIds ?? [block.id],
            overlapTrim: existing?.overlapTrim,
            deleted: false,
            manual: existing?.manual,
            value: center,
            transitionSec: existing?.transitionSec,
        });
    }
    if (replacements.size === 0) return args.state;
    return {
        ...args.state,
        persistedEdits: [
            ...(args.state.persistedEdits ?? []).filter((edit) => !replacements.has(edit.id)),
            ...replacements.values(),
        ],
    };
}

/**
 * 删除块（参数块视图语义）：只隐藏该块，不改写音频、参数曲线或音节分析。
 *
 * 删除记录按成员稳定引用保存；下次自动分析重跑后仍能命中同一批块。若删除的
 * 是合并块，则把整个合并组一并隐藏，避免它的成员自动块重新出现。
 */
export function deleteParamBlocks(args: {
    readonly state: ParamBlockState;
    readonly blocks: readonly ParamBlock[];
    readonly selectedBlockIds: readonly string[];
}): ParamBlockEditResult {
    if (args.state.lineLayout) return deleteLineBlocks(args.state, args.selectedBlockIds);
    const selected = args.blocks.filter((block) => args.selectedBlockIds.includes(block.id));
    if (selected.length === 0) return { ok: false, reason: "not_found" };
    const members = selected.flatMap((block) => blockMemberRefs(block) ?? []);
    if (members.length === 0) return { ok: false, reason: "no_members" };
    const edit: PersistedParamBlockEdit = {
        id: `deleted:${selected
            .map((block) => block.id)
            .sort()
            .join("+")}`,
        members,
        /*
         * 删除记录只表达“视图里隐藏”，但仍要通过后端与持久化模型的区间
         * 有效性校核（宽度 >= MIN_AUTO_BLOCK_SEC）。用块当前区间作为记录
         * 区间即可：它不会移动块，只用于保存 / 重读时保持这条删除意图。
         */
        startSec: Math.min(...selected.map((block) => block.startSec)),
        endSec: Math.max(
            ...selected.map((block) => block.endSec),
            Math.min(...selected.map((block) => block.startSec)) + MIN_AUTO_BLOCK_SEC,
        ),
        memberStartSec: Math.min(...selected.map((block) => block.startSec)),
        memberEndSec: Math.max(...selected.map((block) => block.endSec)),
        blockIds: selected.map((block) => block.id),
    };
    const ids = new Set(selected.map((block) => block.id));
    return {
        ok: true,
        blockId: edit.id,
        state: {
            ...args.state,
            lastDeletedBlocks: [
                ...(args.state.lastDeletedBlocks ?? []),
                ...(args.state.persistedEdits ?? []).filter((existing) =>
                    (existing.blockIds ?? []).some((id) => ids.has(id)),
                ),
            ],
            persistedEdits: [
                ...(args.state.persistedEdits ?? []).filter(
                    (existing) => !(existing.blockIds ?? []).some((id) => ids.has(id)),
                ),
                { ...edit, deleted: true },
            ],
            persistedMerges: (args.state.persistedMerges ?? []).filter(
                (record) => !ids.has(record.id),
            ),
            mergedBlocks: args.state.mergedBlocks.filter((record) => !ids.has(record.id)),
        },
    };
}

/**
 * 把块缩到最小宽度以内时提交删除。
 *
 * 删除记录与 `deleteParamBlocks` 同形，但不依赖“当前选中集合”：边缘手势
 * 可能发生在未选中的块上，若复用选中删除会删错其他块。
 */
export function removeParamBlockAtMinWidth(args: {
    readonly state: ParamBlockState;
    readonly block: ParamBlock;
}): ParamBlockEditResult {
    return deleteParamBlocks({
        state: args.state,
        blocks: [args.block],
        selectedBlockIds: [args.block.id],
    });
}

/** 恢复之前被隐藏的块；不会改动曲线或源音符。 */
export function restoreDeletedParamBlocks(args: {
    readonly state: ParamBlockState;
    /** 只恢复这些自动块的删除记录；空 = 恢复最近一批。 */
    readonly blockIds?: readonly string[];
}): ParamBlockEditResult {
    const ids = new Set(
        args.blockIds !== undefined && args.blockIds.length > 0
            ? args.blockIds
            : (args.state.lastDeletedBlocks ?? []).flatMap((record) => record.blockIds ?? []),
    );
    if (ids.size === 0) return { ok: false, reason: "not_found" };
    return {
        ok: true,
        blockId: args.blockIds?.[0] ?? "restore",
        state: {
            ...args.state,
            lastDeletedBlocks: [],
            persistedEdits: (args.state.persistedEdits ?? []).filter(
                (record) =>
                    !(record.deleted === true && (record.blockIds ?? []).some((id) => ids.has(id))),
            ),
        },
    };
}

/** 是否已有会遮住这些自动块的融合（手动 / 持久化 / 重叠）记录。 */
export function canMergeParamBlocks(
    _state: ParamBlockState,
    blocks: readonly ParamBlock[],
    selectedBlockIds: readonly string[],
): boolean {
    const selected = blocks.filter((block) => selectedBlockIds.includes(block.id));
    if (selected.length < 2) return false;
    const ordered = sortParamBlocksByTime(selected);
    const rootTrackId = ordered[0]?.rootTrackId;
    if (rootTrackId === undefined || ordered.some((block) => block.rootTrackId !== rootTrackId)) {
        return false;
    }
    // 块之间是同级关系；已合并块也可以与相邻块继续合并。
    return !ordered.some(
        (block, index) => index > 0 && !blocksAreAdjacent(ordered[index - 1], block),
    );
}

/** 当前块是否已被用户删除（按稳定成员引用 / 自动块 id 命中）。 */
export function isParamBlockDeleted(state: ParamBlockState, block: ParamBlock): boolean {
    if (state.lineLayout) return !state.lineLayout.blocks.some((item) => item.id === block.id);
    const ids = new Set((block.members.map((member) => member.syllableId) ?? []).filter(Boolean));
    return (state.persistedEdits ?? []).some(
        (record) =>
            record.deleted === true &&
            ((record.blockIds ?? []).includes(block.id) ||
                record.members.some((member) => ids.has(member.syllableId))),
    );
}

/**
 * 设置相接块的圆滑宽度；左右两块都写入同一值，保证手柄从任一侧看一致。
 * 这是块间关系，但不改变块自身的起止时间或高度。
 */
export function setParamBlockTransition(args: {
    readonly state: ParamBlockState;
    readonly left: ParamBlock;
    readonly right: ParamBlock;
    readonly transitionSec: number;
    readonly minBlockSec?: number;
}): ParamBlockEditResult {
    const width = Math.max(0, Number(args.transitionSec) || 0);
    if (args.state.lineLayout)
        return {
            ok: true,
            blockId: args.left.id,
            state: withParamLineBlocks(
                args.state,
                args.state.lineLayout.blocks.map((block) =>
                    block.id === args.left.id
                        ? { ...block, transitionFrames: lineFrame(args.state, width) }
                        : block,
                ),
            ),
        };
    // 过渡只作用于右块，宽度最多占右块一半；左块宽度不参与限制。
    const span = Math.max(0, args.right.endSec - args.right.startSec);
    const clamped = Math.min(width, span / 2);
    const leftMembers = blockMemberRefs(args.left);
    if (leftMembers === null) return { ok: false, reason: "no_members" };
    /*
     * transitionSec 只挂在**左块**上，表示“本块右边界通向下一块”的那一条
     * 相接关系。以前左右两侧各存一份、映射时取 max，等于把同一个宽度同时
     * 写进相邻两条边界；拖一个手柄时整排都会跟着变。
     */
    const left: PersistedParamBlockEdit = {
        id: args.left.id,
        members: leftMembers,
        startSec: args.left.startSec,
        endSec: args.left.endSec,
        memberStartSec: args.left.anchorStartSec ?? args.left.startSec,
        memberEndSec: args.left.anchorEndSec ?? args.left.endSec,
        explicit: true,
        transitionSec: clamped,
        blockIds: [args.left.id],
    };
    return {
        ok: true,
        blockId: args.left.id,
        state: {
            ...args.state,
            persistedEdits: [
                ...(args.state.persistedEdits ?? []).filter((edit) => edit.id !== left.id),
                left,
            ],
        },
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
    if (args.state.lineLayout) return lineBlocksForPanel(args.rootTrackId, args.state);
    const centerRecords = (args.state.persistedEdits ?? []).filter(
        (edit) => edit.deleted !== true && edit.id.startsWith("center:"),
    );
    const geometryState: ParamBlockState = {
        ...args.state,
        persistedEdits: (args.state.persistedEdits ?? []).filter(
            (edit) => !edit.id.startsWith("center:"),
        ),
    };
    const blocks = materializeNonOverlappingParamBlocks({
        ...args,
        state: geometryState,
    });
    if (centerRecords.length === 0) return blocks;
    const centers = new Map<string, number>();
    for (const record of centerRecords) {
        if (record.value === null || record.value === undefined) continue;
        for (const id of record.blockIds ?? []) centers.set(id, record.value);
    }
    return blocks.map((block) => {
        const center = centers.get(block.id);
        return center === undefined
            ? block
            : { ...block, averageValue: center, manualValue: center };
    });
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
    for (const record of args.state.persistedEdits ?? []) {
        for (const member of record.members) {
            blockIds.add(member.syllableId);
        }
    }
    return { blockIds, clipIds };
}

/**
 * 用户手动切分 / 伸缩记录 → 实际块。
 *
 * 校验成员仍存在且源域区间未变；片段按记录里的工程秒显示。用户在 Clip 整体
 * 移动后仍需看到编辑段跟着动，因此这里按**成员对应自动块**当前区间与原始
 * 记录的比例映射，而不是把旧工程秒当成绝对值。
 */
function materializeEditedRecords(
    args: {
        readonly rootTrackId: string;
        readonly clips: readonly ParamBlockSourceClip[];
        readonly state: ParamBlockState;
        /** 这些块已被本次伸缩完整覆盖；不要从自动块池复活。 */
        readonly deletedBlockIds?: ReadonlySet<string>;
    },
    auto: readonly ParamBlock[],
): ParamBlock[] {
    const clipById = new Map(args.clips.map((clip) => [clip.id, clip]));
    /** 显式块是自包含的：只按自身记录恢复，不回查父音节或兄弟块。 */
    const materializeExplicitRecord = (record: PersistedParamBlockEdit): ParamBlock | null => {
        if (record.members.length === 0) return null;
        const ownerClip = clipById.get(record.members[0]?.clipId ?? "");
        if (ownerClip === undefined) return null;
        /*
         * 手动新增块可能暂时没有对应自动音节（空轨 / 已删除音节）。它没有父块可
         * 回查，只能按记录保存的容器锚点恢复；否则保存重开后会直接消失。
         */
        if (record.manual === true) {
            const containers = record.members
                .map((member) => clipById.get(member.clipId))
                .filter((clip): clip is ParamBlockSourceClip => clip !== undefined);
            const clipStart = Math.min(
                ...containers.map((clip) => Math.max(0, finite(clip.startSec, 0))),
            );
            const clipEnd = Math.max(
                ...containers.map(
                    (clip) =>
                        Math.max(0, finite(clip.startSec, 0)) +
                        Math.max(0, finite(clip.lengthSec, 0)),
                ),
            );
            const anchorStart = record.memberStartSec ?? record.startSec;
            const anchorEnd = record.memberEndSec ?? record.endSec;
            const scaleStart = clipStart + (record.startSec - anchorStart);
            const scaleEnd = clipStart + (record.endSec - anchorStart);
            const startSec = Math.max(clipStart, Math.min(clipEnd, scaleStart));
            const endSec = Math.max(clipStart, Math.min(clipEnd, scaleEnd));
            if (endSec - startSec < MIN_AUTO_BLOCK_SEC) return null;
            return {
                id: record.id,
                rootTrackId: args.rootTrackId,
                kind: "auto",
                members: record.members.map((member) => ({
                    clipId: member.clipId,
                    takeId: member.takeId,
                    syllableId: member.syllableId,
                    sourceStartSec: member.sourceStartSec,
                    sourceEndSec: member.sourceEndSec,
                    startSec,
                    endSec,
                })),
                startSec,
                endSec,
                anchorStartSec: anchorStart,
                anchorEndSec: anchorEnd,
                averageValue: record.value ?? null,
                manualValue: record.value ?? null,
                transitionSec: record.transitionSec,
            };
        }

        /*
         * `memberStartSec/memberEndSec` 是记录时的**容器锚点**，不是本块
         * 自己的边界。独立块的边界在 `blockIds` 仍存在时直接以当前自动块
         * 为准；旧记录才退回源区间做比例映射。
         */
        const recordAnchorStart = record.memberStartSec ?? record.startSec;
        const recordAnchorEnd = Math.max(
            recordAnchorStart + MIN_AUTO_BLOCK_SEC,
            record.memberEndSec ?? record.endSec,
        );
        const currentStart = Math.max(0, finite(ownerClip.startSec, 0));
        const currentEnd = currentStart + Math.max(0, finite(ownerClip.lengthSec, 0));
        const ownerAnchor = clipSpan(ownerClip);
        /*
         * 记录同时保存“当前块区间”和“当时容器锚点”。
         * 若容器仍属于同一 Clip 几何，就直接使用记录区间；只有 Clip 整体
         * 移动 / 拉伸时，才按锚点比例映射。不能用“端点是否等于锚点”猜，
         * 否则切分内点会被错当成容器外边界。
         */
        const anchorWidth = Math.max(1e-9, recordAnchorEnd - recordAnchorStart);
        const currentWidth = Math.max(1e-9, currentEnd - currentStart);
        const sameContainer =
            Math.abs(ownerAnchor.startSec - recordAnchorStart) <= 1e-6 &&
            Math.abs(ownerAnchor.endSec - recordAnchorEnd) <= 1e-6;
        /*
         * 同一容器内必须是**刚性平移**：左右块共享容器锚点，若按容器比例
         * 缩放，切分点会被拉开（0.5 秒的块变成 1 秒）。只有 Clip 自身被
         * 拉伸时才按比例映射，那属于源时间伸缩，不是块编辑。
         */
        const scale = sameContainer ? 1 : currentWidth / anchorWidth;
        const rawStartSec = currentStart + (record.startSec - recordAnchorStart) * scale;
        const rawEndSec = currentStart + (record.endSec - recordAnchorStart) * scale;
        const startSec = Math.max(currentStart, Math.min(currentEnd, rawStartSec));
        const endSec = Math.max(currentStart, Math.min(currentEnd, rawEndSec));
        const splitRecord = record.id.includes(":left:") || record.id.includes(":right:");
        if (
            !(endSec > startSec + 1e-9) ||
            (!splitRecord && endSec - startSec < MIN_AUTO_BLOCK_SEC)
        ) {
            return null;
        }

        const members: ParamBlockMember[] = [];
        for (const member of record.members) {
            const clip = clipById.get(member.clipId);
            if (clip === undefined) continue;
            const clipStart = Math.max(0, finite(clip.startSec, 0));
            const clipEnd = clipStart + Math.max(0, finite(clip.lengthSec, 0));
            /*
             * 成员区间必须和块边界用同一个 `scale`。以前这里只做平移，
             * Clip 拉伸（currentWidth != anchorWidth）时块端点会伸缩，但块内
             * 成员仍停在旧宽度，后续切分 / 拖动的命中与手势都不一致。
             */
            const rawMemberStart = currentStart + (record.startSec - recordAnchorStart) * scale;
            const rawMemberEnd = currentStart + (record.endSec - recordAnchorStart) * scale;
            const memberStart = Math.max(clipStart, Math.min(clipEnd, rawMemberStart));
            const memberEnd = Math.max(clipStart, Math.min(clipEnd, rawMemberEnd));
            members.push({
                clipId: member.clipId,
                takeId: member.takeId,
                syllableId: member.syllableId,
                sourceStartSec: member.sourceStartSec,
                sourceEndSec: member.sourceEndSec,
                sourceStartForMatchSec: member.sourceStartSec,
                sourceEndForMatchSec: member.sourceEndSec,
                startSec: memberStart,
                endSec: memberEnd,
            });
        }
        if (members.length === 0) return null;

        return {
            id: record.id,
            rootTrackId: args.rootTrackId,
            kind: "auto",
            members,
            startSec,
            endSec,
            anchorStartSec: recordAnchorStart,
            anchorEndSec: recordAnchorEnd,
            averageValue: record.value ?? null,
            manualValue: record.value ?? null,
            transitionSec: record.transitionSec,
        };
    };
    const findBySource = (
        member: PersistedParamBlockEdit["members"][number],
    ): ParamBlock | undefined => {
        const exact = persistedAuto.get(
            `${member.clipId}|${member.takeId}|${member.sourceStartSec.toFixed(3)}|${member.sourceEndSec.toFixed(3)}`,
        );
        if (exact !== undefined) return exact;
        // 切分后成员是原音节的子区间；按包含关系回溯到父自动块，
        // 再由记录自己的 start/end 表达左右段，不能把两段合并。
        return auto.find((block) =>
            block.members.some(
                (candidate) =>
                    candidate.clipId === member.clipId &&
                    candidate.takeId === member.takeId &&
                    (candidate.sourceStartForMatchSec ?? candidate.sourceStartSec ?? NaN) <=
                        member.sourceStartSec + 1e-3 &&
                    (candidate.sourceEndForMatchSec ?? candidate.sourceEndSec ?? NaN) >=
                        member.sourceEndSec - 1e-3,
            ),
        );
    };
    const persistedAuto = new Map(
        auto.flatMap((block) =>
            block.members
                .filter(
                    (
                        member,
                    ): member is typeof member & {
                        syllableId: string;
                        sourceStartForMatchSec: number;
                        sourceEndForMatchSec: number;
                        takeId: string;
                    } =>
                        member.syllableId !== undefined &&
                        member.takeId !== undefined &&
                        member.sourceStartForMatchSec !== undefined &&
                        member.sourceEndForMatchSec !== undefined,
                )
                .map(
                    (member) =>
                        [
                            `${member.clipId}|${member.takeId}|${member.sourceStartForMatchSec.toFixed(3)}|${member.sourceEndForMatchSec.toFixed(3)}`,
                            block,
                        ] as const,
                ),
        ),
    );
    const editedAutos = new Set<string>();
    const covered = new Set<string>();
    for (const edit of args.state.persistedEdits ?? []) {
        if (edit.deleted === true) continue;
        for (const id of edit.blockIds ?? []) covered.add(id);
    }
    const edited: ParamBlock[] = [];
    const autoById = new Map(auto.map((block) => [block.id, block]));
    /** 显式块记录会整体接管所属容器，旧父子合并 / 编辑记录不再叠加。 */
    const explicitClips = new Set(
        (args.state.persistedEdits ?? [])
            .filter((edit) => edit.explicit === true)
            .flatMap((edit) => edit.members.map((member) => member.clipId)),
    );
    for (const record of args.state.persistedEdits ?? []) {
        if (record.deleted === true) continue;
        if (args.deletedBlockIds?.has(record.id) === true) continue;
        if (
            explicitClips.size > 0 &&
            record.explicit !== true &&
            record.members.some((member) => explicitClips.has(member.clipId))
        ) {
            continue;
        }
        if (record.explicit === true) {
            const materialized = materializeExplicitRecord(record);
            if (materialized !== null) edited.push(materialized);
            continue;
        }
        const members: ParamBlockMember[] = [];
        const matchedBlocks: ParamBlock[] = [];
        let memberSpanStart = Number.POSITIVE_INFINITY;
        let memberSpanEnd = Number.NEGATIVE_INFINITY;
        for (const member of record.members) {
            const block = findBySource(member);
            if (block === undefined || matchedBlocks.includes(block)) continue;
            const memberProjectedStart = Math.max(block.startSec, record.startSec);
            const memberProjectedEnd = Math.min(block.endSec, record.endSec);
            if (memberProjectedEnd - memberProjectedStart > 1e-9) {
                // 切分后的成员与记录几何求交；不能拿完整父音节当当前段。
                memberSpanStart = Math.min(memberSpanStart, memberProjectedStart);
                memberSpanEnd = Math.max(memberSpanEnd, memberProjectedEnd);
            }
            /*
             * 改边会同时写“当前块 + 相邻块”两条记录，而两条记录可能共享
             * 相同的 Clip/Take 成员引用。只按成员查找会把邻居也卷进来，导致
             * 几何用错原块。若记录声明了 blockIds，就只接受其中仍存在的块。
             */
            if (
                record.blockIds !== undefined &&
                record.blockIds.length > 0 &&
                !record.blockIds.some(
                    (id) => autoById.get(id) === block || autoById.get(id) === undefined,
                )
            ) {
                continue;
            }
            matchedBlocks.push(block);
            /*
             * 几何可以回溯到父自动块，但块的**成员身份**必须保留记录自己的
             * 子区间；否则左右两段 materialize 后仍共享父音节成员，拖动会
             * 同时改整段，表现为连体块。
             */
            members.push({
                ...block.members[0],
                syllableId: member.syllableId,
                sourceStartSec: member.sourceStartSec,
                sourceEndSec: member.sourceEndSec,
                sourceStartForMatchSec: member.sourceStartSec,
                sourceEndForMatchSec: member.sourceEndSec,
                startSec: block.startSec,
                endSec: block.endSec,
            });
            memberSpanStart = Math.min(memberSpanStart, block.startSec);
            memberSpanEnd = Math.max(memberSpanEnd, block.endSec);
        }
        if (members.length === 0) continue;
        /*
         * 编辑段记录的是工程秒。切分产生的左右段共享同一组成员，不能拿
         * `record.endSec - record.startSec` 推断比例；而应把记录区间的两个端点
         * 分别投影到**成员原始时间范围**，再把同一个比例应用到成员当前范围。
         *
         * 这样 Clip 整体移动时表现为平移，Clip 拉伸（成员范围变长）时也保持
         * 段落在音符内的相对位置。对于未离开成员范围的普通伸缩，映射退化为
         * 原来的工程秒 ± 位移。
         */
        const originalStart = Math.min(...matchedBlocks.map((block) => block.startSec));
        const originalEnd = Math.max(...matchedBlocks.map((block) => block.endSec));
        /*
         * 成员现在是切分后的**子区间**。比例基准必须取记录保存时的父区块，
         * 而不是当前自动块（它仍覆盖整个原音节）；否则 Clip 移动被当成比例
         * 拉伸，左段会从当前位置退回 0。
         */
        const recordedStart = Math.min(
            record.memberStartSec ?? memberSpanStart,
            record.memberEndSec ?? memberSpanEnd,
        );
        const recordedEnd = Math.max(
            record.memberStartSec ?? memberSpanStart,
            record.memberEndSec ?? memberSpanEnd,
        );
        const currentStart = Math.min(...matchedBlocks.map((block) => block.startSec));
        const currentEnd = Math.max(...matchedBlocks.map((block) => block.endSec));
        /*
         * 切分记录允许“超出原音节”的段（用户把左段延长到相邻音节）。
         * 这时成员仍是完整父音节，但编辑几何必须用记录自身的边界，不能继续
         * 按父音节比例截断，否则拖到 1.5 的段会退回 1.0。
         */
        const recordBeyondMember =
            record.startSec < recordedStart - 1e-6 || record.endSec > recordedEnd + 1e-6;
        const splitRecord = record.id.includes(":left:") || record.id.includes(":right:");
        const isSplitSegment =
            record.members.length === 1 &&
            record.memberStartSec !== undefined &&
            record.memberEndSec !== undefined &&
            splitRecord &&
            !recordBeyondMember;
        if (recordBeyondMember || isSplitSegment) {
            const delta = originalStart - recordedStart;
            edited.push({
                id: record.id,
                rootTrackId: args.rootTrackId,
                kind: matchedBlocks.length > 1 ? "merged" : "auto",
                members,
                startSec: record.startSec + delta,
                endSec: record.endSec + delta,
                averageValue: null,
            });
            continue;
        }
        /*
         * Clip 移动时，切分段的**内部端点**应整体平移；只有与成员边界重合的
         * 外端点才落在成员当前边界上。否则右半段会从 mate 的旧 endSec
         * （2.0）算出 5.0，而不是 Clip 新的末尾 4.0。
         */
        const originalWidth = Math.max(1e-9, originalEnd - originalStart);
        const memberBase = record.memberStartSec ?? recordedStart;
        const memberWidth = Math.max(1e-9, (record.memberEndSec ?? recordedEnd) - memberBase);
        const recordStartT = Math.max(0, Math.min(1, (record.startSec - memberBase) / memberWidth));
        const recordEndT = Math.max(0, Math.min(1, (record.endSec - memberBase) / memberWidth));
        /*
         * 端点按记录保存时的父区间比例映射。只有“记录端点 == 父区间端点”
         * 才吸附到当前父边界；切分内点（如 0.5）不能被 `isMemberStart`
         * 误判成 0，否则左段会被拉回整块。
         */
        const movedStart =
            Math.abs(record.startSec - recordedStart) <= 1e-6
                ? currentStart
                : originalStart + recordStartT * originalWidth;
        const movedEnd =
            Math.abs(record.endSec - recordedEnd) <= 1e-6
                ? currentEnd
                : originalStart + recordEndT * originalWidth;
        if (!(movedEnd > movedStart + 1e-9)) {
            continue;
        }
        edited.push({
            id: record.id,
            rootTrackId: args.rootTrackId,
            kind: matchedBlocks.length > 1 ? "merged" : "auto",
            members,
            startSec: movedStart,
            endSec: movedEnd,
            averageValue: record.manual === true ? (record.value ?? null) : null,
        });
    }
    /*
     * 切分后两个编辑段共享同一个成员自动块；只吞掉**被编辑的那个自动块**，
     * 不能按编辑段当前秒数再把同 Clip 的后续自动块吞掉。这样 `[0,0.5]`
     * 与 `[0.5,1]` 之外，后面的 `[1,2]` 仍会显示。旧快照没有稳定 blockIds，
     * 只能按编辑段区间兜底。
     */
    const autoByStableId = new Map(auto.map((block) => [block.id, block]));
    for (const record of args.state.persistedEdits ?? []) {
        if (record.deleted === true) continue;
        if (record.blockIds === undefined || record.blockIds.length === 0) continue;
        for (const id of record.blockIds) {
            const block = autoByStableId.get(id);
            if (block !== undefined) editedAutos.add(block.id);
        }
    }
    return edited;
}

/** 状态里未被合并记录吞掉的自动块。 */
function materializeAutoBlocks(
    args: {
        readonly state: ParamBlockState;
        readonly clips: readonly ParamBlockSourceClip[];
        readonly deletedBlockIds?: ReadonlySet<string>;
    },
    auto: readonly ParamBlock[],
): ParamBlock[] {
    const occupied = occupiedByMerged(args);
    const editedAutos = new Set<string>();
    const editedSpans: Array<{ startSec: number; endSec: number }> = [];
    for (const edit of args.state.persistedEdits ?? []) {
        /*
         * 显式编辑块保存的是当时可见几何，不是源域子区间。
         * 它只替换被操作的原块；同一 Clip 后续未编辑的自动音节必须继续显示，
         * 因此这里不能用成员源区间或当前块区间去“涂掉”整段。
         */
        if (edit.explicit === true) {
            if (edit.deleted === true) continue;
            for (const id of edit.blockIds ?? []) editedAutos.add(id);
            continue;
        }
        for (const id of edit.blockIds ?? []) editedAutos.add(id);
        if (edit.deleted === true) continue;
        if (edit.memberStartSec !== undefined && edit.memberEndSec !== undefined) {
            for (const block of auto) {
                if (
                    block.startSec < edit.memberEndSec - 1e-6 &&
                    block.endSec > edit.memberStartSec + 1e-6
                ) {
                    editedAutos.add(block.id);
                }
            }
        }
        // 切分后的左右子成员各自占用原自动块，父自动块不能重新露面。
        if (
            edit.memberStartSec !== undefined &&
            edit.memberEndSec !== undefined &&
            edit.members.length === 1
        ) {
            editedAutos.add(edit.blockIds?.[0] ?? `${edit.id}:parent`);
        }
        // 仅有旧快照（无稳定 id）才按成员原始范围兜底；新记录按 blockIds
        // 精确吞块，不能因为同一 Clip 的后续自动块相接而误吞。
        if (edit.blockIds === undefined || edit.blockIds.length === 0) {
            editedSpans.push({
                startSec: edit.memberStartSec ?? edit.startSec,
                endSec: edit.memberEndSec ?? edit.endSec,
            });
        }
    }
    return auto.filter(
        (block) =>
            args.deletedBlockIds?.has(block.id) !== true &&
            !occupied.blockIds.has(block.id) &&
            !block.members.some(
                (member) =>
                    member.syllableId !== undefined && occupied.blockIds.has(member.syllableId),
            ) &&
            !occupied.clipIds.has(block.members[0]?.clipId ?? "") &&
            !editedAutos.has(block.id) &&
            !editedSpans.some((span) => {
                // 编辑段与成员**首尾相接**时，自动块已经由 `edited` 完整表达；
                // 若不把它视为已占用，切分后原块会叠在左右段下方，松手刷新时
                // 看似闪回。这里按“接触即占用”处理，重叠归并仍由最终兜底负责。
                return block.startSec <= span.endSec + 1e-9 && block.endSec >= span.startSec - 1e-9;
            }),
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
    /*
     * 被伸缩完整覆盖的块通过删除记录持久化。显式布局分支会先移除被替换的
     * 自动块，因此删除集合要在这里一次性算好，传给所有派生分支。
     */
    const deletedBlockIds = new Set(
        (args.state.persistedEdits ?? [])
            .filter((edit) => edit.deleted === true)
            .flatMap((edit) => edit.blockIds ?? []),
    );
    const materializeArgs = { ...args, deletedBlockIds };
    const explicitClips = new Set(
        (args.state.persistedEdits ?? [])
            .filter((edit) => edit.explicit === true)
            .flatMap((edit) => edit.members.map((member) => member.clipId)),
    );
    if (explicitClips.size > 0) {
        /*
         * 用户只要编辑过某容器的块，该容器就进入“显式同级块布局”：
         * 未被替换的自动块保留，被替换的原块由显式块独立表达。
         * 旧 v7 父子合并 / 切分记录在语义上已经过时，必须隔离掉，
         * 否则它们会重新吞块，产生“切分后又连体”的回归。
         */
        const explicitState: ParamBlockState = {
            ...args.state,
            mergedBlocks: [],
            persistedMerges: [],
        };
        const explicit = materializeEditedRecords(
            { ...materializeArgs, state: explicitState },
            auto,
        ).filter((block) => block.members.some((member) => explicitClips.has(member.clipId)));
        /*
         * 被替换的原自动块在显式记录里通过 `blockIds` 标记；先按它算出
         * 容器内仍应显示的自动块，再把显式块与它们合并。合并组也用显式块的
         * 当前 id 重建，保证左右块是同级目标。
         */
        /*
         * 显式布局整体接管该容器：自动块只作为未覆盖区间的补集保留。
         * 只要显式块（切分 / 伸缩 / 新增）覆盖了某段自动块，就吞掉那一整块；
         * 未覆盖的自动块继续显示。这样不会因 `blockIds` 让原自动块重新出现。
         */
        const keptAuto = auto.filter((block) => {
            if (deletedBlockIds.has(block.id)) return false;
            if (!block.members.some((member) => explicitClips.has(member.clipId))) return false;
            return !explicit.some(
                (edit) =>
                    edit.startSec < block.endSec - 1e-6 && edit.endSec > block.startSec + 1e-6,
            );
        });
        // Explicit records only own their member Clips. Reuse the normal
        // pipeline for every other Clip, including its existing merges/edits.
        const untouched = materializeNonOverlappingParamBlocks({
            ...args,
            clips: args.clips.filter((clip) => !explicitClips.has(clip.id)),
            state: {
                ...args.state,
                persistedEdits: (args.state.persistedEdits ?? []).filter(
                    (edit) => !edit.members.some((member) => explicitClips.has(member.clipId)),
                ),
            },
        });
        const combined = [...explicit, ...keptAuto, ...untouched];
        return collapseOverlappingResult(combined, args.rootTrackId);
    }
    const merged = materializeMergedRecords(args, auto);
    const edited = materializeEditedRecords(materializeArgs, auto);
    /*
     * 编辑记录按成员稳定引用接管自动块；但改边会写“当前块 + 相邻块”两条
     * 记录，若仍按 `editedAutos` 把所有记录指向的自动块都吞掉，会把相邻块
     * 自己的新边界也一起丢掉。因此先从自动块池里移除真正产生编辑段的原块，
     * 再按记录自身的 `blockIds` 去掉改边记录对应的原块。
     */
    const editedSourceIds = new Set(
        edited.flatMap((block) => block.members.map((m) => m.syllableId)),
    );
    const remaining = materializeAutoBlocks(materializeArgs, auto).filter((block) => {
        if (deletedBlockIds.has(block.id)) return false;
        const editCovers = (args.state.persistedEdits ?? []).some(
            (edit) =>
                edit.blockIds?.includes(block.id) &&
                !block.members.some((member) => editedSourceIds.has(member.syllableId)),
        );
        return !editCovers;
    });
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
        ...edited,
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
        /*
         * 只合并**真正重叠**的块。像素吸附会留下 1e-9~1e-6 量级的端点误差，
         * 若把它当重叠，会把左右相邻的两个显式块误并成一个大块；松手后
         * 用户看到的就是“左右块都消失，只剩中间一段”。
         */
        if (current.length > 0 && block.startSec < currentEnd - 1e-3) {
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
    /** 参数名；只有 pitch 需要半键中心补偿（与曲线共用口径）。 */
    readonly param?: string;
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
         * 块中心走与曲线**完全相同**的 valueToY 口径。pitch 在进入投影前
         * 加 0.5（N 键中心）；不能再在这里手算 normalized，否则半键补偿
         * 会被漏掉或重复，纵向缩放后也会出现额外偏差。
         */
        const centerY = finite(
            args.valueToY(
                args.param === "pitch"
                    ? paramBlockCenterValue("pitch", block.averageValue, {
                          snapToSemitone: block.displayCenterSnapToSemitone !== false,
                      })
                    : block.averageValue,
            ),
            height / 2,
        );
        const topPx = centerY - heightPx / 2;
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
            centerYPx: centerY,
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
 * 块间圆滑：以相接点为对称中心，向左、右各作用 `transitionSec`。
 *
 * 过渡不是把原曲线替换成一条独立贝塞尔；它与相接点两侧的**原曲线**
 * 混合。左边界处权重 0（保留左块原走势），到相接点附近权重升高并平滑
 * 拉向目标值，再向右回落到右块原走势。两侧都改，但都不被覆盖。
 */
export function buildParamBlockTransitionSegments(args: {
    /** 相接点帧（左块右边界 = 右块左边界）。 */
    readonly boundaryFrame: number;
    /** 过渡宽度（帧），相接点左右各延展这么多帧。 */
    readonly widthFrames: number;
    readonly targetValue: number;
    /** 左块最左帧；过渡不能写出这个范围。 */
    readonly leftStartFrame: number;
    /** 右块最右帧；过渡不能写出这个范围。 */
    readonly rightEndFrame: number;
    /** 相接点两侧的原曲线；缺失时退回左 / 右块代表值。 */
    readonly sourceAt?: (frame: number) => number;
    readonly leftValue: number;
    readonly rightValue: number;
}): Array<{ startFrame: number; values: number[] }> {
    const width = Math.max(0, Math.floor(args.widthFrames));
    if (width <= 0) return [];
    const start = Math.max(args.leftStartFrame, args.boundaryFrame - width);
    const end = Math.min(args.rightEndFrame, args.boundaryFrame + width - 1);
    if (end < start) return [];
    const values: number[] = [];
    for (let frame = start; frame <= end; frame += 1) {
        /*
         * 以相接点为峰值的三角形权重：两端为 0、相接点为 1。
         * 目标值不是“覆盖”，只决定这一峰把原曲线拉多近。
         */
        const distance = Math.abs(frame - args.boundaryFrame);
        const t = Math.max(0, Math.min(1, 1 - distance / Math.max(1, width)));
        const smooth = t * t * (3 - 2 * t);
        const source = args.sourceAt?.(frame);
        const base = Number.isFinite(source)
            ? (source as number)
            : frame < args.boundaryFrame
              ? args.leftValue
              : args.rightValue;
        values.push(base + (args.targetValue - base) * smooth);
    }
    return values.length > 0 ? [{ startFrame: start, values }] : [];
}

/**
 * 手柄拖动时计算 live 预览：读取当前窗口的右块原曲线，按平滑权重拉向目标值。
 *
 * 与提交共用 `buildParamBlockTransitionSegments`，因此预览和最终写回后端
 * 是同一套数学；只写相接点右侧，左侧块永远不受影响。
 */
export function buildParamBlockTransitionPreview(args: {
    readonly boundaryFrame: number;
    readonly widthFrames: number;
    readonly targetValue: number;
    readonly leftStartFrame: number;
    readonly rightEndFrame: number;
    readonly leftValue: number;
    readonly rightValue: number;
    readonly paramView: {
        readonly startFrame: number;
        readonly stride: number;
        readonly edit: readonly number[];
    };
}): Array<{ startFrame: number; values: number[] }> {
    const pv = args.paramView;
    const stride = Math.max(1, Math.floor(pv.stride));
    const sourceAt = (frame: number): number => {
        const index = Math.round((frame - pv.startFrame) / stride);
        const value = pv.edit[index];
        return value !== undefined && Number.isFinite(value) ? value : args.targetValue;
    };
    return buildParamBlockTransitionSegments({ ...args, sourceAt });
}

/** 把全分辨率写入片段立即映射回当前参数窗口，供提交后的本地回显使用。 */
export function applyFullResSegmentsToParamView<
    T extends {
        readonly startFrame: number;
        readonly stride: number;
        readonly edit: readonly number[];
    },
>(args: {
    readonly paramView: T;
    readonly segments: readonly {
        readonly startFrame: number;
        readonly values: readonly number[];
    }[];
}): T & { readonly edit: number[] } {
    const next = args.paramView.edit.slice();
    const stride = Math.max(1, Math.round(args.paramView.stride) || 1);
    for (const segment of args.segments) {
        const startFrame = Math.floor(segment.startFrame);
        const firstIndex = Math.max(
            0,
            Math.ceil((startFrame - args.paramView.startFrame) / stride),
        );
        const lastIndex = Math.min(
            next.length - 1,
            Math.floor(
                (startFrame + segment.values.length - 1 - args.paramView.startFrame) / stride,
            ),
        );
        for (let targetIndex = firstIndex; targetIndex <= lastIndex; targetIndex += 1) {
            const frame = args.paramView.startFrame + targetIndex * stride;
            const value = segment.values[frame - startFrame];
            if (value === undefined) continue;
            next[targetIndex] = value;
        }
    }
    return { ...args.paramView, edit: next };
}

/** Compare committed values only at the frames actually sampled by the backend. */
export function paramViewContainsCommittedSegments(args: {
    readonly paramView: {
        readonly startFrame: number;
        readonly stride: number;
        readonly edit: readonly number[];
    };
    readonly segments: readonly {
        readonly startFrame: number;
        readonly values: readonly number[];
    }[];
    readonly tolerance?: number;
}): boolean {
    const stride = Math.max(1, Math.round(args.paramView.stride) || 1);
    const tolerance = Math.max(0, args.tolerance ?? 1e-5);
    for (const segment of args.segments) {
        const startFrame = Math.floor(segment.startFrame);
        const endFrame = startFrame + segment.values.length - 1;
        const firstIndex = Math.max(
            0,
            Math.ceil((startFrame - args.paramView.startFrame) / stride),
        );
        const lastIndex = Math.min(
            args.paramView.edit.length - 1,
            Math.floor((endFrame - args.paramView.startFrame) / stride),
        );
        for (let index = firstIndex; index <= lastIndex; index += 1) {
            const frame = args.paramView.startFrame + index * stride;
            const value = segment.values[frame - startFrame];
            const current = args.paramView.edit[index];
            if (
                value === undefined ||
                current === undefined ||
                !Number.isFinite(current) ||
                !Number.isFinite(value) ||
                Math.abs(current - value) > tolerance
            ) {
                return false;
            }
        }
    }
    return true;
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
 * 块编辑的默认吸附策略。
 *
 * - 纵向（音高）：默认吸附半音；按住取消吸附键才连续自由移动。
 * - 横向（时间）：默认吸附 BPM / 拍号网格；按住取消吸附键才自由落点。
 *
 * 这里刻意**不读** `pitchSnapEnabled` / `timelineSnap.enabled`：块编辑是
 * 明确的结构编辑，默认吸附是产品语义；临时取消由修饰键决定。
 */
export function resolveParamBlockPitchSnapDelta(args: {
    readonly startValue: number;
    readonly rawDelta: number;
    readonly snapToggleHeld: boolean;
}): number {
    if (args.snapToggleHeld) return args.rawDelta;
    /*
     * 吸附的是**块中心的绝对落点**，不是位移量本身。
     * 若起点中心是 60.4，位移 0 仍应落到 60 行心，而不是停在 60.4；
     * 位移 0.6 则落到 61。这样静态显示与拖动吸附使用同一套行心语义。
     */
    const anchorRow = Math.round(args.startValue);
    const targetRow = Math.round(anchorRow + args.rawDelta);
    return targetRow - args.startValue;
}

export function snapParamBlockTimeByDefault(args: {
    readonly sec: number;
    readonly bpm: number;
    readonly timeSignatureDenominator: number;
    readonly stepBeats: number;
    readonly snapToggleHeld: boolean;
}): number {
    if (args.snapToggleHeld) return args.sec;
    return snapParamBlockTimeToGrid({ ...args, enabled: true });
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

/** 把工程秒吸附到 BPM / 拍号网格；`subdivision` 是每拍的细分数。 */
export function snapParamBlockTimeToGrid(args: {
    readonly sec: number;
    readonly bpm: number;
    readonly timeSignatureDenominator: number;
    /** 网格步长（拍）。调用方直接复用时间轴 `gridStepBeats`。 */
    readonly stepBeats: number;
    readonly enabled: boolean;
}): number {
    if (!args.enabled || !Number.isFinite(args.sec)) return args.sec;
    const bpm = Number.isFinite(args.bpm) && args.bpm > 0 ? args.bpm : 120;
    const denominator = Math.max(1, Math.round(args.timeSignatureDenominator) || 4);
    const stepBeats = Math.max(1e-9, Number(args.stepBeats) || 1);
    // 工程拍号的分母不是四分音符时，秒/拍要按分母换算；bar 仅用于文档语义。
    const gridSec = (60 / bpm) * stepBeats * (4 / denominator);
    return Math.round(args.sec / gridSec) * gridSec;
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

/** 快速拖动时 pointerup 可能没有对应的最后一条 pointermove；松手坐标必须优先。 */
export function resolveParamBlockDragRelease(args: {
    readonly didDrag: boolean;
    readonly startClientY: number;
    readonly lastMoveClientY: number;
    readonly releaseClientY: number;
}): { readonly didDrag: boolean; readonly clientY: number } {
    const clientY = Number.isFinite(args.releaseClientY)
        ? args.releaseClientY
        : args.lastMoveClientY;
    const didDrag =
        args.didDrag || (Number.isFinite(clientY) && Math.abs(clientY - args.startClientY) > 3);
    return { didDrag, clientY };
}

/**
 * 拖动期间块的显示中心只由**手势起点 + 指针位移**决定。
 *
 * 在此之前，块的 y 位置由实时曲线主簇反算。即使最终曲线正确，低缩放的
 * stride、局部静音帧和异步窗口切换也会让每一帧算出的主簇略有不同，表现为
 * 块在上下两个像素位置之间闪烁；松手后的全分辨率曲线还会再算一次，造成
 * 最后一次回跳。拖动和交接期间直接投影这一个确定值，曲线仍独立实时更新。
 */
export function projectParamBlockDragDisplayValue(args: {
    readonly startValue: number | null;
    readonly valueDelta: number;
    readonly param: string;
    readonly snapToSemitone: boolean;
}): number | null {
    if (
        args.startValue === null ||
        !Number.isFinite(args.startValue) ||
        !Number.isFinite(args.valueDelta)
    ) {
        return null;
    }
    const value = args.startValue + args.valueDelta;
    return args.param === "pitch" && args.snapToSemitone ? Math.round(value) : value;
}

/**
 * 创建块时把对应曲线整体平移到目标值，保留原有起伏与滑音轮廓。
 * 音高 0 表示无声帧，保持不动。
 */
export function shiftParamBlockCurveValuesToTarget(args: {
    readonly values: readonly number[];
    readonly target: number;
    readonly param: string;
    readonly curveOwned?: boolean;
}): number[] {
    if (!Number.isFinite(args.target)) return args.values.slice();
    const editable = args.values.filter(
        (value) => Number.isFinite(value) && (args.param !== "pitch" || value > 0),
    );
    if (editable.length === 0)
        return args.curveOwned ? args.values.map(() => args.target) : args.values.slice();
    const center =
        (args.curveOwned
            ? paramLineValueCenter(editable, args.param === "pitch")
            : modeCenterOfValues(editable, {
                  param: args.param,
                  isPitch: args.param === "pitch",
              })) ?? editable[0];
    const delta = args.target - (center as number);
    return args.values.map((value) =>
        Number.isFinite(value) && (args.param !== "pitch" || value > 0) ? value + delta : value,
    );
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
    readonly fillEmptyPitch?: boolean;
}): Array<{ startFrame: number; values: number[] }> {
    const fp = Math.max(1e-6, args.framePeriodMs);
    const spans = args.blocks
        .flatMap((block) => block.members)
        .map((member) => {
            /*
             * 块时间区间必须转成**同一采样网格**的帧下标；预览的 stride 可能
             * 大于 1，写入函数会按 `(frame - startFrame) / stride` 反解。
             * 若这里按 1 帧对齐而预览按 stride 网格对齐，边缘会差一帧，
             * 拖动时块高度就会落在两个位置之间闪。
             */
            const startFrame = Math.max(0, Math.round((member.startSec * 1000) / fp));
            const endFrame = Math.max(startFrame, Math.round((member.endSec * 1000) / fp) - 1);
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
            const emptyPitch =
                args.fillEmptyPitch &&
                args.param === "pitch" &&
                !Array.from({ length: span.endFrame - span.startFrame + 1 }, (_, i) =>
                    args.sourceAt(span.startFrame + i),
                ).some((value) => value > 0);
            for (let frame = span.startFrame; frame <= span.endFrame; frame += 1) {
                const source = args.sourceAt(frame);
                if (emptyPitch) {
                    values.push(
                        paramBlockDragValue({
                            param: args.param,
                            original: args.anchorValue,
                            anchorValue: args.anchorValue,
                            valueDelta: args.valueDelta,
                        }),
                    );
                    continue;
                }
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
    readonly fillEmptyPitch?: boolean;
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
        fillEmptyPitch: args.fillEmptyPitch,
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

/** 同时间位置、不同纵向高度的重叠块：优先命中 y 距离最近的，而不是数组最后一个。 */
export function hitTestParamBlocksAtPoint<
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
    readonly slopPx?: number;
}): T | null {
    const sec = finite(args.sec, Number.NaN);
    const y = finite(args.localY, Number.NaN);
    if (!Number.isFinite(sec) || !Number.isFinite(y)) return null;
    const slop = Math.max(0, finite(args.slopPx ?? 4, 4));
    let best: { block: T; distance: number } | null = null;
    for (const block of args.blocks) {
        if (sec < block.startSec || sec > block.endSec) continue;
        const center = block.topPx + block.heightPx / 2;
        const distance = Math.abs(y - center);
        if (distance > block.heightPx / 2 + slop) continue;
        if (best === null || distance < best.distance) best = { block, distance };
    }
    return best?.block ?? null;
}

/** 新块压到的旧块 id：创建预览与提交共用同一个“整块让位”判定。 */
export function blocksCoveredByParamBlockCreate<
    T extends { readonly id: string; readonly startSec: number; readonly endSec: number },
>(args: {
    readonly blocks: readonly T[];
    readonly startSec: number;
    readonly endSec: number;
}): readonly string[] {
    const ids = new Set<string>();
    for (const block of args.blocks) {
        if (block.id === "__create__") continue;
        if (block.endSec <= args.startSec + 1e-9 || block.startSec >= args.endSec - 1e-9) {
            continue;
        }
        ids.add(block.id);
    }
    return [...ids];
}

/**
 * 新建块时旧块让位的**预览几何**：与提交语义一致，压住的旧块整块消失。
 *
 * 预览使用当前可见块（而不是持久化记录），让用户在松手前就看见最终布局；
 * 不能拆成左右残段，否则松手后残段会突然消失，造成“闪一下”的错觉。
 */
export function previewParamBlockCreateLayout<
    T extends {
        readonly id: string;
        readonly startSec: number;
        readonly endSec: number;
        readonly averageValue?: number | null;
        readonly selected: boolean;
        readonly displayCenterValue?: number | null;
        /** pitch 块中心是否吸附半音行心；实时拖动时显式 false。 */
        readonly displayCenterSnapToSemitone?: boolean;
    },
>(args: {
    readonly blocks: readonly T[];
    readonly framePeriodMs?: number;
    readonly startSec: number;
    readonly endSec: number;
    readonly createId: string;
    readonly value: number | null;
}): Array<
    | T
    | {
          readonly id: string;
          readonly startSec: number;
          readonly endSec: number;
          readonly averageValue?: number | null;
          readonly selected: boolean;
          readonly displayCenterValue?: number | null;
      }
> {
    if (args.framePeriodMs) {
        const fp = args.framePeriodMs;
        const before = args.blocks.map((block) => ({
            id: block.id,
            startFrame: Math.round((block.startSec * 1000) / fp),
            endFrameExclusive: Math.round((block.endSec * 1000) / fp),
        }));
        const target = {
            id: args.createId,
            startFrame: Math.round((Math.min(args.startSec, args.endSec) * 1000) / fp),
            endFrameExclusive: Math.round((Math.max(args.startSec, args.endSec) * 1000) / fp),
        };
        return placeParamLineBlock(before, target).map((block) => {
            const original =
                args.blocks.find((old) => old.id === block.id) ??
                args.blocks.find(
                    (old) =>
                        old.startSec <= (block.startFrame * fp) / 1000 + 1e-9 &&
                        old.endSec >= (block.endFrameExclusive * fp) / 1000 - 1e-9,
                );
            return {
                ...original,
                id: block.id,
                startSec: (block.startFrame * fp) / 1000,
                endSec: (block.endFrameExclusive * fp) / 1000,
                selected: block.id === target.id,
                averageValue: block.id === target.id ? args.value : original?.averageValue,
                displayCenterValue:
                    block.id === target.id ? args.value : original?.displayCenterValue,
            };
        });
    }
    const next: Array<
        | T
        | {
              id: string;
              startSec: number;
              endSec: number;
              averageValue?: number | null;
              selected: boolean;
              displayCenterValue?: number | null;
          }
    > = [];
    const coveredIds = new Set(
        blocksCoveredByParamBlockCreate({
            blocks: args.blocks,
            startSec: args.startSec,
            endSec: args.endSec,
        }),
    );
    /*
     * 旧块被新块压住时整块消失，不拆成左右残段：预览阶段必须与松手后的
     * 提交语义一致（提交同样整块删除），否则松手瞬间会看见残段“跳回来”。
     */
    for (const block of args.blocks) {
        if (block.id === args.createId) continue;
        if (coveredIds.has(block.id)) continue;
        next.push(block);
    }
    next.push({
        id: args.createId,
        startSec: Math.min(args.startSec, args.endSec),
        endSec: Math.max(args.startSec, args.endSec),
        averageValue: args.value,
        selected: true,
    });
    return next;
}

/** 创建预览每帧都从按下时的块集重建，拖回后自动恢复曾覆盖的块。 */
export function projectParamBlockCreatePreviewFromBase<
    T extends {
        readonly id: string;
        readonly startSec: number;
        readonly endSec: number;
        readonly averageValue?: number | null;
        readonly selected: boolean;
        readonly displayCenterValue?: number | null;
        readonly displayCenterSnapToSemitone?: boolean;
    },
>(args: {
    readonly baseBlocks: readonly T[];
    readonly framePeriodMs?: number;
    readonly startSec: number;
    readonly endSec: number;
    readonly createId: string;
    readonly value: number | null;
}): ReturnType<typeof previewParamBlockCreateLayout<T>> {
    return previewParamBlockCreateLayout({ ...args, blocks: args.baseBlocks });
}

/**
 * 新建块的持久化值。保持与指针所在半音中心一致；宿主绘制时再加半键视觉偏移，
 * 这样“拖到哪个键”松手后仍然停在那个键上，而不是高/低半音。
 */
export function resolveCreatedParamBlockValue(args: {
    readonly pointerValue: number;
    readonly param: string;
    readonly snapToggleHeld: boolean;
}): number {
    const value = Number.isFinite(args.pointerValue) ? args.pointerValue : 0;
    if (args.param !== "pitch") return value;
    return args.snapToggleHeld ? value : Math.round(value);
}

/** 创建块可用参数：当前根轨道组内、与区间相交的 Clip（重叠时取中心最近者）。 */
export function clipAtParamBlockCreateSpan<
    T extends {
        readonly startSec: number;
        readonly lengthSec: number;
    },
>(args: {
    readonly clips: readonly T[];
    readonly startSec: number;
    readonly endSec: number;
}): T | null {
    const center = (args.startSec + args.endSec) / 2;
    const centerSec = Number.isFinite(center) ? center : args.startSec;
    let best: { clip: T; contains: number; distance: number } | null = null;
    for (const clip of args.clips) {
        const end = clip.startSec + clip.lengthSec;
        if (args.endSec < clip.startSec - 1e-9 || args.startSec > end + 1e-9) continue;
        const contains = centerSec >= clip.startSec && centerSec <= end ? 0 : 1;
        const distance = Math.abs(centerSec - (clip.startSec + clip.lengthSec / 2));
        if (
            best === null ||
            contains < best.contains ||
            (contains === best.contains && distance < best.distance)
        ) {
            best = { clip, contains, distance };
        }
    }
    return best?.clip ?? null;
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

/** 播放头严格位于块内部时返回该块；正好落在公共边界上返回 null。 */
export function paramBlockContainingSec<
    T extends {
        readonly startSec: number;
        readonly endSec: number;
    },
>(blocks: readonly T[], sec: number): T | null {
    if (!Number.isFinite(sec)) return null;
    return (
        blocks.find(
            (block) =>
                sec > Math.min(block.startSec, block.endSec) + 1e-9 &&
                sec < Math.max(block.startSec, block.endSec) - 1e-9,
        ) ?? null
    );
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

/** 当前参数的一行像素高；块高 = 它的 1.2 倍。 */
export function blockRowHeightPx(
    valueToY: ((value: number, heightPx: number) => number) | undefined,
    viewportHeightPx: number,
    param: string,
): number {
    if (param !== "pitch" || valueToY === undefined) return 16 * PARAM_BLOCK_ROW_HEIGHT_MULTIPLIER;
    const row = Math.abs(valueToY(60, viewportHeightPx) - valueToY(61, viewportHeightPx));
    return (Number.isFinite(row) && row > 1e-6 ? row : 16) * PARAM_BLOCK_ROW_HEIGHT_MULTIPLIER;
}

/**
 * 块中心对应的**曲线坐标**。
 *
 * 曲线用 `valueToY(midi + 0.5)` 把 MIDI N 画在 N 键中心；块中心必须走同一
 * 口径。非 pitch 参数没有半键概念，直接使用参数值。
 */
export function paramBlockCenterValue(
    param: string,
    value: number,
    opts?: { readonly snapToSemitone?: boolean },
): number {
    if (param !== "pitch") return value;
    const snap = opts?.snapToSemitone !== false;
    return (snap ? Math.round(value) : value) + 0.5;
}

/**
 * 块中心 → 视口 y 的**唯一**投影。
 *
 * `valueToY` 必须与曲线共用同一个实时视口闭包。不能在调用方另外展开
 * `(value - min) / span`：那会漏掉 `+0.5`，并且非整数行高时会与曲线错位。
 */
export function projectParamBlockCenterY(args: {
    readonly param: string;
    readonly value: number;
    readonly heightPx: number;
    readonly valueToY: (value: number, heightPx: number) => number;
    /** pitch 默认吸附到行心；自由拖动显式中心时传 false。 */
    readonly snapToSemitone?: boolean;
}): number {
    return args.valueToY(
        paramBlockCenterValue(args.param, args.value, {
            snapToSemitone: args.snapToSemitone,
        }),
        Math.max(1, args.heightPx),
    );
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
