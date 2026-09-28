/**
 * 时间轴 · clip 前置重叠计算
 *
 * 【主要内容】
 * 由一组 clip 计算每个 clip 在"自身左侧前导区"的重叠时长（秒），以及判定 clip
 * 渲染顺序的比较函数。
 *
 * 【作用】
 * 重叠区需要做等权可视化混合，避免后绘制的 clip 完全盖住前一个 —— 因此必须知道
 * 每个 clip 左侧被前序 clip 覆盖了多长。渲染顺序（`startSec` 升序、同起点按 id
 * 字典序）决定哪些 clip 算"前序"，故两个函数必须放在一起。
 *
 * 【为什么独立成模块】
 * 原本定义在 `TrackLane.tsx`（旧的轨道 DOM 组件）。旧渲染路径移除后，`TrackLane`
 * 组件被删除，但本函数**仍被使用**：`TimelineWaveformSurface`（波形层，内核视图
 * `TimelineKernelView` 自行挂载，是内核模式下唯一挂载点）导入它，把
 * `leadingOverlapSecByClipId` 交给 `waveform/sceneBuilder` 计算重叠区的等权混合。
 * 因此从组件文件里拆出，避免"删组件顺带删掉活代码"。
 *
 * 【性能约束】本函数在**竖直滚动每跨一行**都会因 rows 重建而重跑（见
 * `TimelineKernelView.handleVisibleRowsChange`），因此必须是 O(n log n)
 * 而不是 O(n²)，且比较器不能走 `localeCompare`。详见两个函数各自的说明。
 *
 * 【与其他模块的关系】
 * - 上游：`TimelineWaveformSurface` 传入 `ClipInfo[]`。
 * - 下游：纯计算，返回 `clipId -> 重叠秒数`，无副作用、不依赖 DOM / React。
 */
import type { ClipInfo } from "../../../features/session/sessionTypes";

/**
 * clip 渲染顺序比较：先按起始时间升序，起始时间相同再按 id 字典序。
 *
 * 【为什么需要】前导重叠的计算结果**依赖遍历顺序**：只有排在当前 clip 之前的
 * clip 才算它的"前序"，才会计入它的左侧重叠。若顺序不稳定，同一组 clip 在两次
 * 渲染中可能得到不同的重叠结果，重叠区颜色随之闪烁。
 *
 * 【为什么不用 `localeCompare`】`localeCompare` 每次比较都要走 Intl 排序规则
 * （多数实现会构造或查询 `Collator`），而本比较器在一次排序里会被调用
 * O(n log n) 次。这里的需求只是「一个确定、可复现的全序」，用 `<` / `>`
 * 比较字符串得到的 UTF-16 码元序同样是全序，而且比 locale 序**更可复现**
 * —— locale 序会随宿主语言环境变化，反而破坏"同一工程两次渲染结果一致"。
 *
 * 【特殊说明】`startSec` 用 `?? 0` 兜底缺字段；起始时间之差小于 `1e-9` 视为
 * 同起点（浮点误差不应改变顺序判定），此时回退到 id 比较 —— 码元序是全序，
 * 保证任意输入下顺序唯一且可复现。
 *
 * @param a 参与比较的 clip
 * @param b 参与比较的 clip
 * @returns 负数表示 a 在前，正数表示 b 在前，0 表示两者等价
 */
function compareClipRenderOrder(a: ClipInfo, b: ClipInfo): number {
    const d = (a.startSec ?? 0) - (b.startSec ?? 0);
    if (Math.abs(d) > 1e-9) return d;
    const idA = String(a.id);
    const idB = String(b.id);
    if (idA < idB) return -1;
    if (idA > idB) return 1;
    return 0;
}

/**
 * 计算每个 clip 在“自身左侧前导区”的重叠时长（秒）。
 *
 * 该前导重叠区对应“该 clip 在当前渲染顺序中位于上层”的区域，
 * 用于在重叠区做等权可视化混合，避免后绘制 clip 完全盖住前一个 clip。
 *
 * 【流程】① 按 `compareClipRenderOrder` 排序得到渲染顺序（不改动入参）；
 * ② 顺序遍历，对每个 clip 回头看它的全部前序 clip，取"前序末端"的最大值
 * （末端 = 起点 + 长度，并与当前 clip 末端取 min，重叠不可能超出自身）；
 * ③ 减去当前 clip 起点即得左侧前导重叠，负数钳到 0。
 *
 * 【为什么可以单遍扫描（O(n log n) → O(n)，排序主导）】
 * 步骤 ② 的定义是「对所有前序 j 取 `max_j min(clipEnd, end_j)`」。排序后
 * 「全部前序」恰好等于「已遍历过的 clip」，而 `min(clipEnd, ·)` 对第二个参数
 * **单调**，于是
 *
 *     max_j min(clipEnd, end_j)  ==  min(clipEnd, max_j end_j)
 *
 * 因此只需维护「已见 clip 的最大末端」这一个标量，无需对每个 clip 回扫全部
 * 前序。此前是 O(n²)：40 clip/轨 × 10 轨约 1.6 万次比较，且每次都发生在竖直
 * 滚动跨行触发的 rows 重建里。等价性有随机对拍用例（对同一输入与旧 O(n²)
 * 参照实现比对）。
 *
 * 【特殊说明】判定重叠时用 `clipStart + 1e-9` 做容差：相邻首尾相接的两个 clip
 * （如 a 到 5、b 从 5 开始）理论重叠为 0，但浮点加减可能得到 1e-16 的"假重叠"，
 * 会让 b 左侧凭空出现一条混合色带。容差把这种数值噪声归零。
 *
 * @param clips 同一轨道（或同一渲染层）内的 clip 列表，顺序任意（只读，不被修改）
 * @returns `clipId -> 前导重叠秒数`（无重叠者为 0）
 */
export function computeLeadingOverlapSecByClipId(
    clips: readonly ClipInfo[],
): Record<string, number> {
    const sorted = [...clips].sort(compareClipRenderOrder);
    const leadingOverlapSecByClipId: Record<string, number> = {};

    let maxEndSoFar = Number.NEGATIVE_INFINITY;
    for (const clip of sorted) {
        const clipStart = clip.startSec;
        const clipEnd = clip.startSec + clip.lengthSec;
        // 单遍扫描：见上方推导。maxEndSoFar 为 -Infinity（首个 clip）时
        // min() 得到 -Infinity，判定为「无重叠」，与旧实现的初始值语义一致。
        const leadingOverlapEnd = Math.min(clipEnd, maxEndSoFar);
        leadingOverlapSecByClipId[clip.id] =
            leadingOverlapEnd > clipStart + 1e-9 ? leadingOverlapEnd - clipStart : 0;
        if (clipEnd > maxEndSoFar) maxEndSoFar = clipEnd;
    }

    return leadingOverlapSecByClipId;
}
