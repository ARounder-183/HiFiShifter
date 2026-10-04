/**
 * 刻度窗口的共享常量与缓冲公式。
 *
 * 【为什么单独成模块】这几个值被四处引用：
 * - **生成刻度**：`buildTimelineTicks` 按窗口加缓冲生成；
 * - **取刻度窗口**：`tickAxis.createTickAxis` 按量化锚点构造轴；
 * - **标尺切片**：`TimeRulerMarks` 从刻度数组里切出可见部分；
 * - **内核滚动提交**：`timelineKernelHost` 决定多久把水平位置提回 React。
 *
 * 此前它们是四个各自写死的字面量/公式，彼此之间的**约束关系**——"提交死区必须
 * 被窗口缓冲吸收，否则可见范围会有一部分没有刻度"——没有任何地方表达，也没有
 * 测试锁定。任何一处被改动（例如把提交步长调大以省重渲染）都会让标尺在滚动或
 * 缩放后露出空白段，而症状只在特定缩放值下出现、滚动一下又消失，极难归因。
 *
 * 收敛到一处之后，约束成为结构性的：内核的提交步长与窗口缓冲引用同一个
 * `TICK_WINDOW_LAG_PX`，`buildTimelineTicks.windowing.test.ts` 再把
 * "滞后 ≤ 缓冲" 钉成不变量。
 *
 * 【依赖约定】本模块**不导入任何其它模块**，因此可被上述四处任意引用而不产生
 * 循环依赖（`buildTimelineTicks` ↔ `tickAxis` 之间存在既有的相互引用关系）。
 */

/**
 * 刻度窗口锚点的量化步长（CSS px）。
 *
 * 滚动位置按它向下量化成锚点，取刻度时把视口宽加宽一个步长，从而保证
 * "锚点 ≤ 真实滚动位置 < 锚点 + 步长"、覆盖区间必然包含真实视口。
 * 量化让滚动期间刻度数组与标尺子树不必每帧重算。
 */
export const TICK_WINDOW_STEP_PX = 256;

/**
 * 内核向 React 提交水平位置的量化步长（CSS px），也就是刻度窗口必须吸收的
 * **滞后上界**。
 *
 * 【为什么必须有上界】内核每帧用**实时** `scrollLeft` 写标尺内容层的 transform，
 * 而标尺里"有哪些刻度"由 React 按**它自己的** `scrollLeft` state 生成。两者之间
 * 隔着一个量化提交（为避免滚动帧进 React）。因此 React 侧的位置最多滞后内核真值
 * 一整个步长 —— 窗口缓冲必须覆盖这段滞后，否则视口一端会出现没有刻度的空白段。
 *
 * 内核的提交步长与窗口缓冲都引用本常量，二者不可能再漂移。
 */
export const TICK_WINDOW_LAG_PX = 256;

/**
 * 刻度窗口的单侧缓冲（CSS px）—— **生成与切片共用同一公式**。
 *
 * 此前 `buildTimelineTicks` 与 `TimeRulerMarks` 各自写死 `max(320, vw * 0.5)`，
 * 两处公式一旦分叉（例如只改一处），切片窗口就会比生成窗口更宽，切出**不存在**
 * 的刻度区间，同样表现为标尺露白。
 *
 * 下界取 `TICK_WINDOW_LAG_PX + 64`：既要严格大于滞后上界（否则滞后最大时窗口
 * 盖不住视口），也留一点余量吸收设备像素吸附（`rulerLayerTranslatePx`）带来的
 * 亚像素误差。
 *
 * @param viewportWidthPx 视口宽度（CSS px）。非法值按 0 处理（退化为下界）。
 */
export function tickWindowBufferPx(viewportWidthPx: number): number {
    const width = Number.isFinite(viewportWidthPx) && viewportWidthPx > 0 ? viewportWidthPx : 0;
    return Math.max(TICK_WINDOW_LAG_PX + 64, width * 0.5);
}
