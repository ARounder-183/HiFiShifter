/**
 * 空白区 seek 手势的阶段语义（纯函数，供内核与面板共用）。
 *
 * 【要修的问题】播放中对时间轴**空白位置**按下（或按住拖拽）时，播放光标会
 * "闪回"：旧实现把按下当成一次立即提交的 seek（写 `playheadSec` + 打后端），
 * 而拖动中间帧又只写乐观值、不打后端 —— 于是前端 `playheadSec` 与 30Hz 播放
 * 轮询（`syncPlaybackState.fulfilled` 用引擎实际播放位置覆写它）反复争夺同一个
 * 字段。视觉插值对"非轮询写入"走硬复位、对"新鲜轮询采样"走 1x 外推，两种写入
 * 交替到达时表现为光标在指针位置与引擎播放位置之间来回跳（约 33ms 一个周期）。
 *
 * 【目标语义】播放状态下：
 * - 按下：**不动**播放光标（只保留"空白点击"的选中语义）；
 * - 拖拽：**不动**播放光标；
 * - 松手：把播放光标跳到松手位置并**延续播放**（后端 seek 不改变 is_playing）。
 *
 * 非播放状态保持既有行为：按下即跳转、拖拽逐帧预览、松手提交。
 *
 * 【为什么把阶段显式化】内核此前只回传一个 `commit` 布尔（按下与松手都是
 * true），面板无法区分"按下"与"松手"，也就无法只对其中一个放行播放头写入。
 * 阶段是本模块的输入契约；策略表放在这里，回归测试（`seekGesture.test.ts`）
 * 无需启动内核或 React 即可钉死它。
 */

/** seek 手势的阶段：按下 / 拖拽中间帧 / 松手收尾。 */
export type SeekGesturePhase = "press" | "move" | "release";

/** 播放头的写入方式。 */
export type SeekPlayheadWrite = "none" | "preview" | "commit";

export interface SeekGesturePlan {
    /**
     * 是否执行「空白点击」的选中语义（清空 clip 选中 + 按设置切换当前轨道）。
     *
     * 只在**按下**时执行一次：松手不再重复（旧实现按下与松手都走提交分支，
     * 同一语义被执行两次）。
     */
    readonly blankClickSemantics: boolean;
    /**
     * 播放头写入方式：
     * - `"none"`：完全不写（播放中按下 / 拖拽）；
     * - `"preview"`：只写前端乐观值，按 rAF 节流，不打后端；
     * - `"commit"`：写乐观值并提交后端 seek。
     */
    readonly playhead: SeekPlayheadWrite;
}

/**
 * 解析本次 seek 事件的执行计划。
 *
 * @param phase seek 手势阶段。
 * @param isPlaying 当前是否播放中。调用方应传 store 的**实时**值
 *   （`store.getState().session.runtime.isPlaying`），而不是 effect 提交后
 *   才刷新的镜像。
 * @returns 执行计划；详见 `SeekGesturePlan`。
 */
export function planSeekGesture(phase: SeekGesturePhase, isPlaying: boolean): SeekGesturePlan {
    if (phase === "press") {
        return {
            blankClickSemantics: true,
            // 播放中按下不跳转：否则立即写下的乐观值会被随后的播放轮询拽回引擎
            // 位置（闪回的根因），且松手前音频已因后端 seek 跳走。
            playhead: isPlaying ? "none" : "commit",
        };
    }
    if (phase === "move") {
        return {
            blankClickSemantics: false,
            playhead: isPlaying ? "none" : "preview",
        };
    }
    // release：无论是否播放都提交落点。播放中提交后引擎 seek 到该处、传输层
    // 保持播放（后端 `set_transport` 不改 is_playing），即"延续播放状态"。
    return { blankClickSemantics: false, playhead: "commit" };
}
