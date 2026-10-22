/**
 * 片段**item 层**的增益 —— 时间轴旋钮编辑、header 标签展示的那一层。
 *
 * 【为什么要一个函数】REAPER 的 item 增益与 take 增益是两个旋钮。渲染、几何、标签
 * 若能各自决定读哪一层，迟早有一处读错（表现是"标签显示 0 dB，声音却是 −6 dB"）。
 * 统一从这里取，读哪一层就只有一个答案。
 *
 * 【用在哪些地方】凡是把 `ClipInfo` 投影成"视图片段"（命中块、渲染块、泳道块）的
 * 构造点：把 item 增益写进视图的 `gain` 字段，下游的样式解析与标签就自动一致——
 * 不需要给四层视图类型各加一个新字段。
 *
 * `clip_gain` 缺省（旧载荷）按 1.0；0 是合法的静音值，不能被当成"缺省"。
 */
export function clipItemGain(clip: { clip_gain?: number } | null | undefined): number {
    const value = clip?.clip_gain;
    return typeof value === "number" && Number.isFinite(value) ? Math.max(0, value) : 1;
}
