/**
 * 裁切 / 延伸的**源窗口换算**（唯一定义处）。
 *
 * ## 要解决的问题
 *
 * 拖 Clip 边缘延伸 / 裁短时，用户直觉的规则**与是否倒放无关**：
 *
 * > 拖**右**边 ⇒ **左**边的内容固定；拖**左**边 ⇒ **右**边的内容固定。
 *
 * 但"时间轴的左端 / 右端分别播放哪个源位置"在倒放时是**镜像**的。后端对此有唯一
 * 权威模型（`backend/src-tauri/src/state.rs:121` `clip_playback_window_sec`）：
 *
 * ```text
 * 正放：win = [source_start, source_start + len·rate)   升序消费，输出不翻转
 * 倒放：win = [source_end − len·rate, source_end)       升序消费，输出整体翻转
 * ```
 *
 * 配合"升序消费后再整体翻转"的约定（`state.rs:108-119` 的方向契约、`:155-159` 的
 * 前导静音同侧），得到：
 *
 * | | 时间轴**左**端播放 | 时间轴**右**端播放 |
 * |---|---|---|
 * | 正放 | `source_start` | `source_start + len·rate` |
 * | 倒放 | `source_end`   | `source_end − len·rate`   |
 *
 * 倒放下"存储的 `sourceStartSec` 就是**右端**消费位置"这一点还被后端规范化直接钉死：
 * `state.rs:238-253` `normalize_nonloop_source_window` 对倒放写
 * `source_start := source_end − len·rate`。
 *
 * ## 本模块的规则
 *
 * 倒放时"时间轴某一端"与"源字段"的对应关系整个调换（左端 = `source_end`、右端 =
 * `source_start`），所以**既要取反位移、也要调换被写的字段**：
 *
 * ```text
 * δsrc = δ · rate · (reversed ? −1 : 1)
 *
 *              正放            倒放
 * 拖左缘（右端固定） source_start += δsrc   source_end += δsrc
 * 拖右缘（左端固定） source_end   += δsrc   source_start += δsrc
 * ```
 *
 * 等价说法：**被改的字段永远是"被拖的那条时间轴边缘"所对应的源字段**，而位移的符号
 * 在倒放时取反（因为源域方向与时间轴方向相反）。
 *
 * 代数校验（倒放，`se=10, len=4, rate=1` ⇒ `win=[6,10]`；左端播 10、右端播 6）：
 *
 * - 右缘 +1s ⇒ `len=5`，要求左端仍播 10 ⇒ `se` 不动、`win=[5,10]` ⇒
 *   `ss: 6→5 = ss₀ + δsrc`（`δsrc = −1`）✓
 * - 左缘 +1s（缩短）⇒ `start+1, len=3`，要求右端仍播 6 ⇒ `ss` 不动、`win=[6,9]` ⇒
 *   `se: 10→9 = se₀ + δsrc`（`δsrc = −1`）✓
 *
 * 另一种等价写法是"固定端字段 + 新长度"：两个方向的窗口关系恒为 `se = ss + len·rate`
 *（正放倒放同一式），因此**只要知道哪一端固定，另一端由该式推出**。本模块选择位移式，
 * 因为它只依赖 `deltaSec`，不要求传入的源窗口已被规范化。
 *
 * ## Loop
 *
 * Loop clip 的 `sourceStartSec / sourceEndSec` 是**回绕相位锚点**（周期 D 见
 * `state.rs:296-307`），不是消费窗口，两条边因此不对称：
 *
 * - **右缘**：只改长度。相位锚点决定"时间轴**左端**播放媒体的哪个位置"，动它会把
 *   左端一起带走（违反"左端固定"）。多出的长度由回绕内容填充。
 * - **左缘**：必须改锚点 —— 左端内容本就该随拖拽移动，而右端由"锚点 + 长度"
 *   共同决定，按 `δ·rate` 反向调整锚点即可保持右端不变（与非 Loop 同一条公式）。
 *   锚点取模环绕到 `[0, D)` 保持规范（消费端本就按 `floor_mod` 解释，环绕只是防止
 *   多次拖拽后数值无界漂移）。
 *
 * 【曾经的缺陷】把 Loop 一律当作"只改长度"（右缘的规则）会让**左缘拖拽完全不写
 * 锚点** ⇒ 右端内容跟着长度变化跑掉（实测：拖左缘 1s 后右端源位置 4 → 5）。
 * 旧实现 `useEditDrag` 对此有显式分支（`loop && limitedDelta < 0` 的环绕锚点回退、
 * `loop && reversed` 的锚点推进），内核迁移时丢失。
 *
 * ## 为什么必须收口到一处
 *
 * 本函数诞生于一个真实缺陷：单 clip 与多选两条路径各自手写了这段换算，**两份都漏了
 * `reversed`**，于是倒放 Clip 拖右缘改的是左端、拖左缘改的是右端 —— 固定的一侧整个
 * 颠倒。四条分支（边 × 倒放）塌缩成"取反 + 两条边"后，方向只出现一次，结构上不再有
 * 第二处可以写错。
 */

/** 裁切的边。 */
export type TrimEdge = "left" | "right";

export interface TrimSourceWindowArgs {
    /** 被拖拽的边。 */
    readonly edge: TrimEdge;
    /** Clip 是否倒放。 */
    readonly reversed: boolean;
    /** Clip 是否启用 Loop（源字段是回绕锚点 ⇒ 不改）。 */
    readonly loopEnabled: boolean;
    /**
     * 时间轴位移（秒）。
     *
     * 语义与 `resolveTrimEdge` 的 `deltaSec` 一致：**拖右缘为正 = 变长**；
     * **拖左缘为负 = 变长**（`resolveTrimEdge` 的 `deltaSec` 在左缘是
     * `nextStart − startSec`，向左拖为负）。
     */
    readonly deltaSec: number;
    /**
     * **组合**消费速率（源秒 / 时间轴秒）：
     * `clipPlaybackRate × activeTake.playbackRate`，与 `state.rs:121` 的
     * `span = length × clip.playback_rate` 同一口径。
     *
     * 【为什么不能传 clip 级倍率】Take 速率 ≠ 1 时源位移会被算错（正放倒放都错）。
     */
    readonly rate: number;
    /**
     * Loop 的回绕周期 D（秒）；`0` = 未知。
     *
     * 仅用于把 Loop 的相位锚点取模环绕到 `[0, D)`（防多次拖拽后数值无界漂移）。
     * 非 Loop 不使用。
     */
    readonly mediaDurationSec: number;
    readonly sourceStartSec: number;
    readonly sourceEndSec: number;
}

export interface TrimSourceWindow {
    readonly sourceStartSec: number;
    readonly sourceEndSec: number;
}

/** 速率净化：非有限 / 过小按 1 处理（与后端 `pr_valid` 同口径）。 */
function sanitizeRate(rate: number): number {
    return Number.isFinite(rate) && rate > 1e-6 ? rate : 1;
}

/**
 * 把边缘拖拽位移换算为新的源窗口。
 *
 * @returns 新的 `{sourceStartSec, sourceEndSec}`；**Loop 的右缘或输入非法时返回
 *   `null`**（调用方只改 `lengthSec`，跳过源窗口写入）。
 */
export function resolveTrimSourceWindow(
    args: TrimSourceWindowArgs,
): TrimSourceWindow | null {
    if (!Number.isFinite(args.deltaSec)) return null;

    const rate = sanitizeRate(args.rate);
    // 源域方向与时间轴方向在倒放时相反（见文件头）。
    const sourceDelta = args.deltaSec * rate * (args.reversed ? -1 : 1);

    const sourceStartSec = Number.isFinite(args.sourceStartSec) ? args.sourceStartSec : 0;
    const sourceEndSec = Number.isFinite(args.sourceEndSec) ? args.sourceEndSec : 0;

    // Loop 的**右缘**：相位锚点决定时间轴左端播放位置，右缘移动只应改长度。
    // （非 Loop 的右缘相反：必须移动"右缘对应的源端点"才能固定左端。）
    if (args.loopEnabled && args.edge === "right") return null;

    // Loop 的锚点取模环绕到 [0, D)，保持字段规范、防止多次拖拽后无界漂移。
    const wrap = (value: number): number =>
        args.loopEnabled && args.mediaDurationSec > 1e-9
            ? ((value % args.mediaDurationSec) + args.mediaDurationSec) % args.mediaDurationSec
            : value;

    // 被改的字段 = **被拖的那条时间轴边缘**所对应的源字段：
    //   正放：左缘 ↔ sourceStart、右缘 ↔ sourceEnd
    //   倒放：左缘 ↔ sourceEnd、右缘 ↔ sourceStart（镜像）
    // 另一端保持逐值不变（这正是"固定的是另一边"）。
    if (args.edge === "left") {
        return args.reversed
            ? { sourceStartSec, sourceEndSec: wrap(sourceEndSec + sourceDelta) }
            : { sourceStartSec: wrap(sourceStartSec + sourceDelta), sourceEndSec };
    }
    return args.reversed
        ? { sourceStartSec: wrap(sourceStartSec + sourceDelta), sourceEndSec }
        : { sourceStartSec, sourceEndSec: wrap(sourceEndSec + sourceDelta) };
}
