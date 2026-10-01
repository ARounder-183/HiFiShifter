/**
 * 颤音预览画布共用的绘制原语。
 *
 * 【为什么单独成文件】预览由两块画布组成 —— 上方的"原参数线轮廓"条与下方的"颤音
 * 偏移"主图 —— 它们共享同一条时间轴，因此断口处理必须**逐字一致**：两边对"哪些
 * 帧没有值"的判断若不同，同一个时刻就会一边断开、一边连线，看起来像数据错位。
 *
 * 这里只放与业务无关的几何/路径工具，不碰任何标尺与配色。
 */

/**
 * 把 `[0, count)` 按"该点是否有值"切成若干连续段。
 *
 * 【为什么需要】音高曲线里的「未检测」帧以 `NaN` 给出（见 `vibratoPitch.ts` 的音符段
 * 判定）。不切段的话，Canvas 会把断口两侧直接连起来 / 填起来 —— "这里没有数据"
 * 就被画成了"这里有一条线"，正是要避免的误读。
 */
export function finiteRuns(values: readonly number[], count: number): Array<[number, number]> {
    const runs: Array<[number, number]> = [];
    let start = -1;
    for (let i = 0; i < count; i += 1) {
        const ok = Number.isFinite(values[Math.min(i, values.length - 1)]);
        if (ok) {
            if (start < 0) start = i;
        } else if (start >= 0) {
            runs.push([start, i - 1]);
            start = -1;
        }
    }
    if (start >= 0) runs.push([start, count - 1]);
    return runs;
}

/**
 * 折线：遇到无值的采样点**抬笔**，下一个有效点重新起笔。
 *
 * 直接 `lineTo(NaN)` 在 Canvas2D 里等价于"跳过这一点"，断口两侧仍会被连成一条
 * 直线 —— 那正好把"这里没有数据"画成了"这里有一条线"。显式抬笔才能让断口真的断开。
 *
 * 【为什么收成一个自由函数而不是各自实现】主图与轮廓条都要这一套；两份实现里只要
 * 有一份漏了有限性判断，断口就会在那一块画错，而这类缺陷不会抛错。
 */
export function strokeFinitePolyline(
    ctx: CanvasRenderingContext2D,
    values: readonly number[],
    xAt: (index: number) => number,
    yAt: (value: number) => number,
): void {
    ctx.beginPath();
    let pen = false;
    for (let i = 0; i < values.length; i += 1) {
        const value = values[i];
        if (!Number.isFinite(value)) {
            pen = false;
            continue;
        }
        const x = xAt(i);
        const y = yAt(value);
        if (pen) ctx.lineTo(x, y);
        else {
            ctx.moveTo(x, y);
            pen = true;
        }
    }
    ctx.stroke();
}
