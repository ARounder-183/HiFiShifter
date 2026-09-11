/**
 * 时间轴渲染内核 · 命中测试
 *
 * 【主要内容】
 * 把「内容坐标下的指针位置」换算为工程时间，并判定命中的轨道与 clip 分区
 * （header / body）。
 *
 * 【作用】
 * 新内核的轨道区**没有任何 DOM 内容层**，命中必须由几何计算得出。本模块是
 * 「看到的 = 可点的」这条不变式的落点：几何常量与绘制端同源（`rowHeight`、
 * `pxPerSec`、`CLIP_HEADER_HEIGHT`），因此不存在 DOM 命中区与 canvas 视觉漂移。
 *
 * 【与其他模块的关系】
 * - 上游：内核宿主在手势开始时调用（见 `host/timelineKernelHost`）。
 * - 下游：手势状态机据命中结果进入「选中 / 拖拽 / seek」分支。
 * - 独立性：纯函数，无 DOM / React 依赖，可直接单测。
 *
 * 【设计约束】
 * 1. 轨道内 clip 按 `startSec` **升序**（调用方保证），本模块用二分查找定位
 *    候选 clip——轨道内 clip 数可达数百，线性扫描会进入手势热路径。
 * 2. 命中判定使用**半开区间** `[startSec, startSec + lengthSec)`：相邻紧贴的
 *    两个 clip 在交界处只命中右侧那个，与旧实现的 DOM 层叠顺序一致。
 */

/** 命中测试所需的 clip 最小字段集。 */
export interface HitTestClip {
    readonly id: string;
    readonly trackId: string;
    readonly startSec: number;
    readonly lengthSec: number;
}

/** 命中测试所需的轨道最小字段集（顺序即纵向排列顺序）。 */
export interface HitTestTrack {
    readonly id: string;
}

/** clip 内的命中分区。 */
export type ClipHitRegion = "header" | "body";

/** 命中结果。 */
export type HitResult =
    | {
          /** 命中空白（轨道内无 clip，或落在轨道区之外）。 */
          readonly kind: "empty";
          /** 指针处的工程时间（秒，已钳制到 >= 0）。 */
          readonly sec: number;
          /** 命中的轨道 id；落在轨道区之外时为 null。 */
          readonly trackId: string | null;
          /** 命中的轨道行下标；落在轨道区之外时为 -1。 */
          readonly trackIndex: number;
      }
    | {
          /** 命中 clip。 */
          readonly kind: "clip";
          readonly clip: HitTestClip;
          readonly region: ClipHitRegion;
          readonly sec: number;
          readonly trackIndex: number;
      };

/** 命中测试参数。 */
export interface HitTestArgs {
    /** 指针的内容坐标 x（CSS px，= scrollLeft + 视口内偏移）。 */
    readonly contentX: number;
    /** 指针的内容坐标 y（CSS px，= scrollTop + 视口内偏移）。 */
    readonly contentY: number;
    readonly pxPerSec: number;
    readonly rowHeight: number;
    /** 轨道列表（顺序 = 纵向排列顺序）。 */
    readonly tracks: readonly HitTestTrack[];
    /** 按轨道 id 分桶的 clip（每桶内按 startSec 升序）。 */
    readonly clipsByTrack: ReadonlyMap<string, readonly HitTestClip[]>;
    /** clip header 高度（CSS px），用于区分 header / body 分区。 */
    readonly headerHeightPx: number;
}

/**
 * 在轨道内二分查找覆盖 `sec` 的 clip。
 *
 * 规则：取「最后一个 startSec <= sec」的 clip，再校验 sec 是否落在其长度内
 * （半开区间，见文件头约束 2）。
 *
 * @param list 该轨道的 clip（按 startSec 升序）。
 * @param sec 目标时间（秒）。
 * @returns 命中的 clip；无命中时为 null。
 */
function findClipAt(list: readonly HitTestClip[], sec: number): HitTestClip | null {
    if (list.length === 0) return null;
    let low = 0;
    let high = list.length - 1;
    let candidate: HitTestClip | null = null;
    while (low <= high) {
        const mid = (low + high) >> 1;
        const item = list[mid];
        if (item.startSec <= sec) {
            candidate = item;
            low = mid + 1;
        } else {
            high = mid - 1;
        }
    }
    if (candidate === null) return null;
    return sec < candidate.startSec + candidate.lengthSec ? candidate : null;
}

/**
 * 执行命中测试。
 *
 * 流程：内容坐标 → 工程时间 → 轨道行 → 轨道内二分 → 分区判定。
 *
 * 特殊说明：落在轨道区之外（`contentY < 0` 或超出最后一行）时返回 `empty` 且
 * `trackId = null`，调用方据此区分「轨道间空白」与「工程空白」（前者不 seek）。
 *
 * @param args 命中参数。
 * @returns 命中结果。
 */
export function hitTest(args: HitTestArgs): HitResult {
    const safePxPerSec = Number.isFinite(args.pxPerSec) ? Math.max(1e-9, args.pxPerSec) : 1e-9;
    const sec = Number.isFinite(args.contentX) ? Math.max(0, args.contentX / safePxPerSec) : 0;
    const rowHeight = Number.isFinite(args.rowHeight) ? Math.max(1, args.rowHeight) : 1;
    const trackIndex = Number.isFinite(args.contentY)
        ? Math.floor(Math.max(0, args.contentY) / rowHeight)
        : 0;

    if (trackIndex >= args.tracks.length) {
        return { kind: "empty", sec, trackId: null, trackIndex: -1 };
    }
    const track = args.tracks[trackIndex];
    const list = args.clipsByTrack.get(track.id) ?? [];
    const clip = findClipAt(list, sec);
    if (clip === null) {
        return { kind: "empty", sec, trackId: track.id, trackIndex };
    }

    const localY = Math.max(0, args.contentY) - trackIndex * rowHeight;
    const headerHeightPx = Number.isFinite(args.headerHeightPx)
        ? Math.max(0, args.headerHeightPx)
        : 0;
    return {
        kind: "clip",
        clip,
        region: localY < headerHeightPx ? "header" : "body",
        sec,
        trackIndex,
    };
}
