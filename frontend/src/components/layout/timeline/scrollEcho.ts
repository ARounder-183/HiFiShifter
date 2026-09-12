/**
 * 时间轴内核 · 轨道头滚动回灌判定。
 *
 * 【主要内容】
 * 判定「轨道头（`TrackList`）容器报来的 `scrollTop`」是否应当回灌给内核。
 *
 * 【作用：为什么需要它——这是纵向"卡卡的、像被吸附"的根因】
 * 内核模式下轨道头容器只是**被动镜像**：宿主 `syncDom` 每帧写
 * `trackList.scrollTop = view.scrollTop`（未量化）。这次写入会触发原生 `scroll`
 * 事件，而事件不带来源。该事件的链路是：
 *
 * ```
 * syncDom 写 trackList.scrollTop（未量化）
 *   → 原生 scroll 事件（TrackList.tsx 的 onScroll）
 *   → 读回**量化、且滞后一帧**的 el.scrollTop
 *   → onScrollTopChange → TimelinePanel → host.setScrollTop(滞后值)
 * ```
 *
 * 实测（1920×1200，40 步拖竖向滚动条）：`setScrollTop` 被调 19 次，**19 次全部**
 * 来自上述 `onScroll` 链路，每次 `delta` 都是负数（−9.1 … −9.5）——内核正前进却被
 * 每次拉回约 9px；入参是量化值（9.5 / 28 / 46.5 / 65.5），而内核此刻在
 * 18.65 / 37.3 / 55.95。逐帧往复即用户报告的"卡卡的、像被吸附"。
 *
 * 【为什么不能简单删掉这条链路】
 * 轨道头容器**确实**承载一类真实用户输入：焦点在轨道头内的控件上时，浏览器原生
 * 的 **scroll-into-view**（Tab / 方向键切换焦点）会改变 `el.scrollTop`。实测：
 * 聚焦最后一个控件 → `domTop` 0 → 308；PageDown → 132；End → 361。
 * 这些不经过任何 JS 转发，只能靠原生 `scroll` 事件到达。因此必须**区分来源**，
 * 而不是一刀切忽略。
 *
 * 【判据：与「上次镜像写入值」比，而不是与「内核当前值」比】
 * 现有实现拿事件值和**内核当前值**比（`Math.abs(host.getViewport().scrollTop -
 * scrollTop) < 0.5`），这在拖拽时必然失效：内核每帧都在前进，事件报来的是**上一帧**
 * 镜像的值，两者相差约 9px，于是回声被当成用户输入收下。
 *
 * 正确的比较基准是**宿主上一次写进容器的值**。回声事件报的正是那个值（浏览器按
 * 设备像素量化，通常逐值相等），而真实输入（scroll-into-view）会把容器带到**别的**
 * 值上。这与横向轴的解（`onUserScrollLeft` 的来源标记）同一思路：**判来源，不判
 * 与当前真值的距离**。
 *
 * 【与其他模块的关系】
 * - 上游：`timelineKernelHost` 提供"上一次镜像写入值"（其 `syncDom` 的去重变量）。
 * - 消费者：`TimelinePanel.handleTrackListScrollTopChange` 据此决定是否回灌内核。
 * - 独立性：纯函数，不依赖 DOM / React，可直接单测。
 */

/** 回灌判定入参。 */
export interface TrackListEchoArgs {
    /**
     * 宿主上一次镜像回写轨道头时写下的值。
     *
     * 特殊说明：从未写过时为 `NaN`（内核尚在起步），此时一律判"不是回声"——
     * 宁可多采纳一次真实输入，也不要在首帧前把用户操作吞掉。
     */
    readonly mirroredScrollTop: number;
    /** 事件报来的容器当前 `scrollTop`。 */
    readonly nativeScrollTop: number;
    /** 视为"同一个值"的容差（CSS px）。 */
    readonly tolerancePx?: number;
}

/**
 * 容差默认值（CSS px）。
 *
 * 取 0.5：与宿主 `syncDom` 的写入去重容差同量级。
 *
 * 【为什么必须是 0.5 而不是更小——量化误差是常规情形，不是罕见情形】
 * 浏览器把原生 `scrollTop` 按设备像素量化，**写→读**的误差有界但普遍存在，且量级
 * 随 dpr 变化（均为实测）：
 * - dpr 1 → 最大 0.5（如写 20.5 读回 21；写 10.37 读回 10.5）
 * - dpr 2 → 最大 0.25（如写 540.694 读回 540.5；写 124.4 读回 124.5）
 * - dpr 3 → 约 0.167
 *
 * 因此"回声事件值 == 上次写入值"只在部分 dpr 下逐值成立，容差承担的是**吸收常规
 * 量化误差**的职责。推论：容差不能再小（dpr 1 时误差正好触到 0.5），也不宜放大
 * （真实输入会被误判为回声而吞掉）。
 *
 * 特殊说明：0.5 是 dpr 1 的**恰好上界**，`<=` 的闭区间语义在这里是承重的——改成
 * `<` 会让 dpr 1 边界情形漏判。`scrollEcho.test.ts` 用实测的三个 dpr 值钉住了
 * 两侧边界。
 */
const DEFAULT_TOLERANCE_PX = 0.5;

/**
 * 判定轨道头报来的 `scrollTop` 是否为镜像回声。
 *
 * 流程：从未回写过（NaN）→ 非回声；否则与上次镜像写入值比较，在容差内判为回声。
 *
 * 特殊说明 1：**取反写法是刻意的**。`Math.abs(x - NaN) <= eps` 恒为 false，
 * 正着写会把"从未回写"误判成回声，从而吞掉首帧前的真实滚动。
 *
 * 特殊说明 2：本函数只回答"是否回声"，**不做钳制**。钳制由 `ScrollKernel` 统一
 * 负责（单一职责），这里再夹一次会出现两份上限来源。
 *
 * @param args 见 `TrackListEchoArgs`。
 * @returns 是镜像回声时为 true（调用方**不得**据此回灌内核）。
 */
export function isTrackListMirrorEcho(args: TrackListEchoArgs): boolean {
    const tolerance = Number.isFinite(args.tolerancePx)
        ? Math.max(0, args.tolerancePx as number)
        : DEFAULT_TOLERANCE_PX;
    const mirrored = args.mirroredScrollTop;
    const native = args.nativeScrollTop;
    if (!Number.isFinite(mirrored) || !Number.isFinite(native)) return false;
    return Math.abs(native - mirrored) <= tolerance;
}
