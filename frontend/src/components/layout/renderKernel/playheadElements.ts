/**
 * 播放头元素写入器 —— 把**同一个视口左缘**写进播放头的竖线与倒三角。
 *
 * 【它解决的三个真实缺陷】
 *
 * 1. **元素重建后失联**（"倒三角停在工程起始处不动"的成因）。逐帧写 `style` 必须
 *    去重，但只用「位置是否变化」当键会漏掉一类情况：元素在播放头**静止**时被重建
 *    （停靠重排搬动 DOM、视图/标尺子树重挂载）。新元素没有 `left`，而位置与上次写入
 *    相同 ⇒ 判定为"不必写" ⇒ 元素停在静态位置（左缘 = 工程起始处）**永远不动**，
 *    直到播放头再次移动。因此去重键是「**元素身份** + 位置」。
 *
 * 2. **一个元素被去重、另一个跟着失联**。倒三角的写入曾经嵌在竖线的去重分支里，
 *    竖线被跳过时三角也一起被跳过。这里每个元素各占一个槽位（`slot`），互不影响。
 *
 * 3. **写入口径分散**。标尺线写 `style.left`、轨道区主体线写 `style.transform`，
 *    两处各自换算过一次。本模块只接收**已算好的视口左缘**，换算仍由
 *    `playheadLineLeftPx` 唯一负责（见 `playheadLine.test.ts`）。
 *
 * 特殊说明：面板挂载期的一次性兜底写不归本写入器管辖 —— 宿主首帧的槽位缓存是空的，
 * 必然无条件纠正一次，因此不存在"兜底值被当成已写值而不再修正"的情形。
 */

/** 位置去重阈值（CSS px）：亚像素抖动不产生视觉变化，不必写。 */
const POSITION_EPSILON_PX = 0.01;

/** 播放头元素写入器。 */
export interface PlayheadElementWriter {
    /** 写 `style.left`（标尺竖线 / 倒三角）。 */
    writeLeft(slot: string, element: HTMLElement | null, leftPx: number): void;
    /** 写 `style.transform = translateX()`（轨道区 / 编辑器主体竖线）。 */
    writeTranslateX(slot: string, element: HTMLElement | null, leftPx: number): void;
    /** 清空全部槽位缓存（宿主重建后调用，使下一帧无条件写一次）。 */
    reset(): void;
}

/**
 * 创建写入器。
 *
 * @returns 写入器（无状态外泄：槽位缓存只在本闭包内）。
 */
export function createPlayheadElementWriter(): PlayheadElementWriter {
    const slots = new Map<string, { element: HTMLElement; leftPx: number }>();

    const write = (
        slot: string,
        element: HTMLElement | null,
        leftPx: number,
        apply: (target: HTMLElement, x: number) => void,
    ): void => {
        // 非有限值（数据未就绪 / 视口退化）绝不落地：写进去会得到 `NaNpx`，
        // 元素直接失去定位，比"保持上一次位置"更糟。
        if (!Number.isFinite(leftPx)) return;
        if (element === null) {
            // 元素暂不可用（未挂载 / 已被移出文档）：丢掉槽位，等它回来时无条件写一次。
            slots.delete(slot);
            return;
        }
        const previous = slots.get(slot);
        if (
            previous !== undefined &&
            previous.element === element &&
            Math.abs(previous.leftPx - leftPx) <= POSITION_EPSILON_PX
        ) {
            return;
        }
        slots.set(slot, { element, leftPx });
        apply(element, leftPx);
    };

    return {
        writeLeft(slot, element, leftPx) {
            write(slot, element, leftPx, (target, x) => {
                target.style.left = `${x}px`;
            });
        },
        writeTranslateX(slot, element, leftPx) {
            write(slot, element, leftPx, (target, x) => {
                target.style.transform = `translateX(${x}px)`;
            });
        },
        reset() {
            slots.clear();
        },
    };
}
