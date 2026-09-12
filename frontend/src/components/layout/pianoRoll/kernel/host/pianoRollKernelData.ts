/**
 * 参数编辑器内核 · 宿主数据镜像
 *
 * 【主要内容】
 * 声明宿主每帧需要从 React 侧读取的数据形状：工程时长、缩放，以及当前参数的
 * **值域视口**（`min` / `max` / `span`）。
 *
 * 【作用】
 * 宿主是长生命周期的命令式运行时对象，若直接持有 React state 或 Redux store，
 * 数据更新会迫使宿主重建（丢失滚动位置与手势状态）。数据镜像把「回调用什么」
 * 与「React 何时重渲染」解耦：面板每次 render 更新镜像字段，宿主在帧提交时现读。
 *
 * 【为什么值域只有 min/max/span 而没有 center】
 * `center` 的真值是内核的像素 `scrollTop`（唯一事实源）。若镜像里再放一份
 * `center`，就出现两个真值源，滚动条拖拽与面板写入会互相覆盖。这里只提供换算
 * 参数（值域边界与跨度），`center` 一律经 `verticalValueScroll` 双向换算得出。
 *
 * 【与其他模块的关系】
 * - 上游：`PianoRollPanel` 构造镜像对象并每次 render 更新其字段。
 * - 下游：`pianoRollKernelHost` 在帧提交与值域换算时读取。
 * - 独立性：纯类型 + 无逻辑，不依赖 DOM / React。
 */

/** 值域边界与跨度（与 `verticalValueScroll` 的入参同形）。 */
export interface PianoRollValueDomain {
    /** 该参数值域下界。 */
    readonly min: number;
    /** 该参数值域上界。 */
    readonly max: number;
    /** 视口当前可见的值跨度。 */
    readonly span: number;
}

/** 宿主每帧读取的数据镜像。 */
export interface PianoRollKernelData {
    /**
     * 工程总时长（秒），用于算内容宽度与横向滚动上限。
     *
     * 特殊说明：应与面板 `contentWidth` 同源（面板已含 `Math.max(1, …)` 保护），
     * 两边不一致会让滚动条 thumb 比例与实际可滚范围分叉。
     */
    readonly projectSec: number;
    /**
     * 当前选中参数的值域（供竖向「值域 ↔ 像素」换算）。
     *
     * 特殊说明：参数切换 / 值域变化时面板只需更新本字段；竖向滚动位置由内核持有，
     * 不会因为本字段变化而复位（与旧实现一致：切参数后滚动条位置按新值域重算，
     * 由面板调用 `host.setValueCenter` 显式提交）。
     */
    readonly valueDomain: PianoRollValueDomain;
}
