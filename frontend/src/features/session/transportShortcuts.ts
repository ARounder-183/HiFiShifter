/**
 * 传输类快捷键的语义裁决（播放 / 暂停 / 停止）。
 *
 * 【为什么抽成纯函数】这两个快捷键的语义必须与 DAW 惯例严格对齐，而它
 * 曾经被一次"防重复触发"的修复误伤 —— `playback.stop` 在空闲时的分支被
 * 整段删除（提交 77553e61），于是默认绑定 `Enter`（标签即「播放 / 停止」）
 * 在未播放时变成静默 no-op，用户读到的手册语义（`回车`：播放/停止）与
 * 实际行为分叉。裁决表放在这里，回归测试（`transportShortcuts.test.ts`）
 * 就能把这张表钉死，不必启动整个 App 组件。
 *
 * 【语义】（与 REAPER / VEGAS 等 DAW 一致，也是本项目的既定定义）
 * - `playback.toggle`（默认 Space）= **播放 / 暂停**：
 *   播放中按 → 暂停，播放光标停在**当前播放位置**；空闲时按 → 从光标起播。
 * - `playback.stop`（默认 Enter）= **播放 / 停止**：
 *   播放中按 → 停止，播放光标回到**本次起播位置**（`playbackAnchorSec`）；
 *   空闲时按 → 从光标起播（**这一条曾是缺失的分支**）。
 *
 * 「暂停」与「停止」的唯一区别就是光标落点：前者留在原地，后者回到起播点。
 */

/** 传输快捷键触发的动作（由 App 的快捷键处理器翻译为对应的 Redux thunk）。 */
export type TransportShortcutCommand = "play" | "pause" | "stop";

/** 参与语义裁决的传输快捷键。 */
export type TransportShortcutActionId = "playback.toggle" | "playback.stop";

/**
 * 解析传输快捷键在当前播放态下应执行的动作。
 *
 * @param actionId 命中的快捷键动作。
 * @param isPlaying 当前是否处于播放中（调用方应传 store 的**实时**值，
 *   而不是 effect 提交后才刷新的镜像——连打时镜像会滞后一拍）。
 * @returns `"play"` = 从当前播放头起播；`"pause"` = 停止但留在当前位置；
 *   `"stop"` = 停止并回到本次起播位置。
 */
export function resolveTransportShortcutCommand(
    actionId: TransportShortcutActionId,
    isPlaying: boolean,
): TransportShortcutCommand {
    if (isPlaying) {
        return actionId === "playback.stop" ? "stop" : "pause";
    }
    return "play";
}
