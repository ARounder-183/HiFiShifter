/**
 * fadeLineClickGesture — 淡变包络线的单击/双击消歧（时间窗 + 目标键）。
 *
 * ## 为什么不用 PointerEvent.detail
 * UI Events 规范只对 mouse 事件序列保证 detail=点击计数；pointerdown 的
 * detail 在部分 WebView（Tauri 使用的 WebView2 等）中恒为 0，导致基于
 * `detail >= 2` 的双击检测失效。这里改用**时间窗 + 目标键**的确定性判定：
 * 同一目标键在 DOUBLE_MS 内的第二次按下即视为双击。
 *
 * 注意：不再做 seek 延迟 —— 播放头交互已降为最低优先级（pointer-events-none），
 * 且包络线单击的寻址落点收敛到淡化区内侧边缘，因此第一击的寻址不会与
 * 双击重置冲突。
 */

const DOUBLE_MS = 320;

let lastDown: { key: string; time: number } | null = null;

export type FadeLineClickPhase = "first" | "double";

/** 记录一次包络线按下；返回本次按下属于双击的第二击还是首次单击。 */
export function noteFadeLinePointerDown(key: string): FadeLineClickPhase {
    const now = performance.now();
    const prev = lastDown;
    lastDown = { key, time: now };
    if (prev && prev.key === key && now - prev.time <= DOUBLE_MS) {
        return "double";
    }
    return "first";
}
