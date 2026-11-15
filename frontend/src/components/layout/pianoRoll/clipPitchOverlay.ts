import { CLIP_PITCH_OVERLAY_COLOR_COUNT } from "./colors";
import { toFrame } from "./paramBlockProjection";

/**
 * 参数编辑器 · Clip 源音高参考线（纯派生模型）。
 *
 * 数据源是后端 clip_pitch_data 推送并存入 session.clipPitchCurves 的
 * per-clip 曲线。它与根轨道合成的 pv.orig 不是一回事：
 *
 * - pv.orig 是单值合成线：重叠 Clip、音高参考块都会让不同来源在同一帧竞争，
 *   最终只留下一个值；
 * - 本模型保留每个 Clip 自己的曲线，用于重合、音高参考块覆盖和原始 F0 对照。
 *
 * 【帧基准：一律是工程帧，不是连续秒】
 * 后端 emit 时曲线第 i 个采样对应工程帧 `round(clip.startSec * 1000 / fp) + i`
 * （与 assemble_pitch_orig_from_cache 写入 pv.orig 的位置、与参数块投影
 * toFrame() 完全同源）。本模型因此只保存 `startFrame`，绝不保存
 * `curveStartSec` —— 按连续秒落笔会让每条 Clip 各偏一个亚帧相位（≤2.5 ms），
 * 在放大到 0.5 s 视口时是 8~9 px 的可见错位，用户会以为是自己操作错了。
 *
 * 【颜色编码 Clip 身份，不跟随 Clip 的装饰色】
 * `clip.color` 的默认值就是 emerald，用户不改色就全是同一种绿。因此这里只产出
 * `colorIndex`（稳定哈希 + 重合冲突消解），具体色值由 colors.ts 按主题解析 ——
 * 模型不碰主题，绘制侧不碰哈希。
 *
 * 【只在"真的没有数据"处切段】
 * 切段（`runs`）的唯一理由是 `midi <= 0`：该帧确实没有检测/参考值。重合深度
 * 不参与几何 —— 旧实现按"重合带边界"切段，而边界是连续秒、采样点是量化帧，
 * 切点几乎必然落在两个采样之间：绘制层把每段画成独立折线，于是重合处出现
 * **一帧宽的洞**；若切出的段只有 1 个采样，还会被整段丢掉，曲线看起来晚一帧
 * 开始。两种症状都出现在同一个 Clip 起点上，用户读到的是"两条曲线都在这里断了一小段"。
 *
 * 本文件不碰 React / DOM / GL；只负责把 Clip 几何、曲线、焦点态变成可绘制的
 * 描述符，供 sceneBuilders / render.ts 共用。
 */

/** ClipInfo 中本模块需要的字段（保持结构类型，便于测试）。 */
export interface ClipPitchOverlayClip {
    readonly id: string;
    readonly trackId: string;
    readonly startSec: number;
    readonly lengthSec: number;
    readonly muted?: boolean;
    readonly color?: string;
    readonly sourcePath?: string;
    readonly midiNoteData?: readonly unknown[];
}

export interface ClipPitchCurveData {
    readonly curveStartSec: number;
    readonly midiCurve: readonly number[];
    readonly framePeriodMs: number;
}

/**
 * 一段**连续有声**帧。
 *
 * 只有 `midi <= 0`（或非有限值）才会把曲线切成两段：那是数据本身不存在的地方，
 * 连过去就是一条贯穿静音的假斜线。除此之外不得有任何切分理由 —— 每一刀都会在
 * 绘制层变成"两条独立折线"，两段之间不会连线。
 */
export interface ClipPitchOverlayRun {
    /** 曲线数组中的起始下标（含）。 */
    readonly startIndex: number;
    /** 曲线数组中的结束下标（不含）。 */
    readonly endIndexExclusive: number;
}

export interface ClipPitchOverlay {
    readonly clipId: string;
    /** 曲线第 0 个采样所在的工程帧（= round(clip.startSec * 1000 / fp)）。 */
    readonly startFrame: number;
    /** 该 Clip 在工程帧网格上的帧数（= round(clip.lengthSec * 1000 / fp)）。 */
    readonly frameCount: number;
    readonly framePeriodMs: number;
    readonly midiCurve: readonly number[];
    readonly kind: "sourceF0" | "midiReference";
    /** 身份色序号（0..CLIP_PITCH_OVERLAY_COLOR_COUNT-1），与重合邻居必不相同。 */
    readonly colorIndex: number;
    /** 是否是当前焦点的 Clip（时间轴上被选中的那一个）。 */
    readonly focused: boolean;
    readonly zOrder: number;
    /** 连续有声段；段之间是真实的静音/无数据帧，绝不连线。 */
    readonly runs: readonly ClipPitchOverlayRun[];
}

export interface ClipPitchOverlayVisual {
    readonly alpha: number;
    readonly lineWidthPx: number;
    readonly dash: readonly [number, number] | null;
}

/** 通过有效性校验、真正会被绘制的 Clip。 */
interface EligibleClip {
    readonly clip: ClipPitchOverlayClip;
    readonly zOrder: number;
    readonly curve: ClipPitchCurveData;
    readonly startFrame: number;
    readonly frameCount: number;
}

function clipStart(clip: ClipPitchOverlayClip): number {
    return Math.max(0, Number.isFinite(clip.startSec) ? clip.startSec : 0);
}

function clipLength(clip: ClipPitchOverlayClip): number {
    return Number.isFinite(clip.lengthSec) ? Math.max(0, clip.lengthSec) : 0;
}

/**
 * FNV-1a 32 位哈希。
 *
 * 只用于"给 Clip 一个稳定的初始色号"：同一个 clipId 在任何会话、任何工程顺序
 * 下都得到同一个值，因此增删别的 Clip 不会让某条曲线换色。
 */
function hashClipId(text: string): number {
    let hash = 0x811c9dc5;
    for (let i = 0; i < text.length; i += 1) {
        hash ^= text.charCodeAt(i);
        hash = Math.imul(hash, 0x01000193) >>> 0;
    }
    return hash >>> 0;
}

/**
 * 给每个可绘制 Clip 分配身份色号。
 *
 * 【为什么不能只用 hash】撞色本身无害（不重合的两个 Clip 同色不影响阅读），
 * **但重合的 Clip 撞色就等于没修**。所以先按工程帧区间建重合邻接图，再按
 * `(startFrame, clipId)` 稳定排序做贪心着色：从 `hash % N` 开始，与已着色的
 * 邻居冲突就循环 +1。
 *
 * 结果只取决于「集合 + 区间拓扑」，与遍历顺序无关 —— 滚动、缩放、增删无关
 * Clip 都不会让颜色跳变。
 */
function assignColorIndices(items: readonly EligibleClip[]): Map<string, number> {
    const order = [...items].sort(
        (a, b) => a.startFrame - b.startFrame || a.clip.id.localeCompare(b.clip.id),
    );
    const neighbors = new Map<string, string[]>();
    for (const item of order) neighbors.set(item.clip.id, []);
    for (let i = 0; i < order.length; i += 1) {
        const a = order[i]!;
        for (let j = i + 1; j < order.length; j += 1) {
            const b = order[j]!;
            // 已按起点排序：b 的起点越过 a 的终点，后面的更不可能相交。
            if (b.startFrame >= a.startFrame + a.frameCount) break;
            if (a.startFrame < b.startFrame + b.frameCount) {
                neighbors.get(a.clip.id)!.push(b.clip.id);
                neighbors.get(b.clip.id)!.push(a.clip.id);
            }
        }
    }

    const assigned = new Map<string, number>();
    for (const item of order) {
        const used = new Set<number>();
        for (const neighborId of neighbors.get(item.clip.id) ?? []) {
            const taken = assigned.get(neighborId);
            if (taken !== undefined) used.add(taken);
        }
        let index = hashClipId(item.clip.id) % CLIP_PITCH_OVERLAY_COLOR_COUNT;
        for (let step = 0; step < CLIP_PITCH_OVERLAY_COLOR_COUNT && used.has(index); step += 1) {
            index = (index + 1) % CLIP_PITCH_OVERLAY_COLOR_COUNT;
        }
        // 调色板用完（重合深度 ≥ 色数）时退回哈希值：宁可撞色也不死循环。
        assigned.set(item.clip.id, index);
    }
    return assigned;
}

/**
 * ClipPitchCurves → per-clip 参考线。
 *
 * 过滤规则与后端 emit 契约对齐（全部按**工程帧**判断，与绘制位置同一把尺子）：
 * - `round(curveStartSec·1000/fp)` 必须等于 `round(clip.startSec·1000/fp)`；
 *   移动不足半帧时网格没变，曲线仍然有效；超过半帧立即作废；
 * - 曲线帧数必须覆盖 Clip 的帧数，否则整条丢弃（不拉伸、不补零）；
 * - 帧周期必须与工程一致，否则曲线会整体缩放（宁可少画不画错）；
 * - 有声段按 midi <= 0 / 非有限值断开；**除此之外不切段**（见 `runs` 说明）。
 */
export function buildClipPitchOverlays(args: {
    readonly clips: readonly ClipPitchOverlayClip[];
    readonly curves: Readonly<Record<string, ClipPitchCurveData>>;
    readonly groupTrackIds: ReadonlySet<string>;
    /** 当前焦点的 Clip（时间轴上的单选）；null = 没有焦点。 */
    readonly focusClipId: string | null;
    /**
     * 工程的帧周期（毫秒），来自参数视图。已知时必须与曲线自带的帧周期一致，
     * 否则曲线会整体缩放。缺省（参数视图还没取到）时不做这项校验。
     */
    readonly projectFramePeriodMs?: number;
}): ClipPitchOverlay[] {
    const { clips, curves, groupTrackIds, focusClipId, projectFramePeriodMs } = args;

    const eligible: EligibleClip[] = [];
    clips.forEach((clip, zOrder) => {
        if (!groupTrackIds.has(clip.trackId) || clip.muted === true) return;
        const curve = curves[clip.id];
        if (curve === undefined || !Number.isFinite(curve.curveStartSec)) return;
        const framePeriodMs = curve.framePeriodMs;
        if (!Number.isFinite(framePeriodMs) || framePeriodMs <= 0) return;
        // 曲线与工程不在同一个帧周期上 ⇒ 画出来必然整体缩放，宁可整条不画。
        if (
            projectFramePeriodMs != null &&
            Number.isFinite(projectFramePeriodMs) &&
            Math.abs(framePeriodMs - projectFramePeriodMs) > 0.1
        ) {
            return;
        }
        const startFrame = toFrame(clipStart(clip), framePeriodMs);
        const frameCount = toFrame(clipLength(clip), framePeriodMs);
        if (frameCount < 2) return;
        // 旧缓存属于别的几何：起点落在另一个工程帧上就整条丢弃。
        if (toFrame(curve.curveStartSec, framePeriodMs) !== startFrame) return;
        // 曲线必须覆盖整个 Clip；短了说明是分割 / 拉伸前的旧数据。
        if (curve.midiCurve.length < frameCount) return;
        eligible.push({ clip, zOrder, curve, startFrame, frameCount });
    });

    if (eligible.length === 0) return [];
    const colorIndices = assignColorIndices(eligible);

    const overlays: ClipPitchOverlay[] = [];
    for (const item of eligible) {
        const { clip, zOrder, curve, startFrame, frameCount } = item;
        const runs: ClipPitchOverlayRun[] = [];
        let runStart = -1;
        const flush = (endExclusive: number) => {
            if (runStart >= 0 && endExclusive - runStart >= 2) {
                runs.push({ startIndex: runStart, endIndexExclusive: endExclusive });
            }
            runStart = -1;
        };
        for (let index = 0; index < frameCount; index += 1) {
            const value = curve.midiCurve[index];
            const voiced = value != null && Number.isFinite(value) && value > 0;
            if (!voiced) {
                flush(index);
                continue;
            }
            if (runStart < 0) runStart = index;
        }
        flush(frameCount);
        if (runs.length === 0) continue;

        overlays.push({
            clipId: clip.id,
            startFrame,
            frameCount,
            framePeriodMs: curve.framePeriodMs,
            midiCurve: curve.midiCurve,
            kind:
                clip.midiNoteData !== undefined && clip.midiNoteData.length > 0
                    ? "midiReference"
                    : "sourceF0",
            colorIndex: colorIndices.get(clip.id) ?? 0,
            focused: focusClipId === clip.id,
            zOrder,
            runs,
        });
    }

    // 低层先画、高层后画、焦点 Clip 最后画；否则焦点线会被相邻 Clip 盖住。
    overlays.sort(
        (a, b) =>
            Number(a.focused) - Number(b.focused) ||
            a.zOrder - b.zOrder ||
            a.clipId.localeCompare(b.clipId),
    );
    return overlays;
}

/**
 * 一条 Clip 参考线图层的视觉参数。
 *
 * 【alpha 只有这一层】色值 alpha 恒为 1，透明度全在这里决定 —— 旧实现把 0.85
 * 写进色值再乘一次，两层相乘既难推理也难调。
 *
 * 【这是一层**参考线**，不是主线】参数编辑器的强度顺序必须单调：
 * `pv.edit`（0.92 / 2.6）> `pv.orig`（0.55 / 1.8）> 本图层（≤0.50 / ≤1.35）。
 * 上一版焦点用 0.95 / 1.8 —— 与 `pv.orig` 同宽、比它还亮，用户的原话是
 * "分不清主次"。这里把上限定成**折算亮度不超过 `pv.orig`**（见
 * clipPitchOverlay.test.ts 的"视觉层级"用例），焦点只保留"比别的 Clip 稍亮稍粗"这一档。
 *
 * 【为什么不再按重合深度提亮】身份色已经回答"这条线属于哪个 Clip"，重合时
 * 两条不同颜色的线同时出现本身就是信号；再叠一层明暗，同一条线会在同一段里
 * 忽明忽暗，既和焦点高亮抢同一个视觉通道，也是上一轮"曲线在重合处断开"的
 * 直接成因（切段点必然落在两个采样之间）。
 *
 * @param overlay 参考线描述符。
 * @returns 视觉参数（alpha / 线宽 / 虚线图案）。
 */
export function resolveClipPitchOverlayVisual(overlay: ClipPitchOverlay): ClipPitchOverlayVisual {
    const reference = overlay.kind === "midiReference";
    const alpha = reference ? (overlay.focused ? 0.38 : 0.24) : overlay.focused ? 0.5 : 0.34;
    return {
        alpha,
        lineWidthPx: reference ? (overlay.focused ? 1.1 : 0.9) : overlay.focused ? 1.35 : 1,
        dash: reference ? [2, 3] : null,
    };
}
