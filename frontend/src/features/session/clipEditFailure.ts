/**
 * 片段编辑失败分类：区分"后端正在自行收敛"与"需要用户处置"。
 *
 * 【为什么需要分类】此前 `setClipStateRemote` / `setClipsStateBulkRemote` 的
 * `.rejected` 一律把状态置成 `"Clip edit rejected"`（"音频块编辑被拒绝"），而任何
 * 失败模式都走同一条路 —— 包括**瞬时**的"宿主还在追上几何/媒体"（后端会自己重试并
 * 收敛，不需要用户做任何事）。结果用户每做一次编辑都被报一次错，即使编辑其实生效了。
 *
 * 【为什么按文案匹配而不是后端给 code】这些字符串由后端在宿主回调边界产出，是既有的
 * 稳定契约（且都有对应测试钉住）。加一个 `kind` 字段会同时改协议、webview 与前端三处，
 * 收益只是省掉一次子串匹配。**但方向是明确的**：日后若后端加上机器可读的 `kind`，
 * 这个函数就是唯一需要改的地方。
 *
 * 【默认必须偏"永久"】未知文案判为 `permanent`：宁可多报一次可见错误，也不能把一个
 * 真正需要处置的失败静默掉。
 */
export type ClipEditFailureKind = "transient" | "permanent";

/**
 * 瞬时失败的文案标记（后端会自行收敛，前端不提示）：
 * - 宿主尚未把几何/媒体/身份回流给插件（`…has not caught up` / `is not ready`）；
 * - 插件自己的写回触发的重载窗口（`Conflict: host …`，见会话侧 `editing_session`）；
 * - 重试预算耗尽（`budget exceeded`）——继续提示也无法让用户做任何事；
 * - 编辑租约在编辑期间被撤销（`lease revoked`），下一次操作会重新取得。
 */
const TRANSIENT_MARKERS: readonly string[] = [
    // 单复数都覆盖：`clip … has not caught up` / `both clips … have not caught up`。
    "not caught up",
    "is not ready",
    "not ready",
    "Conflict: host",
    "refresh required",
    "budget exceeded",
    "lease revoked",
    "host PCM unavailable",
    "host model not ready",
];

/** 把一次片段编辑失败归类为瞬时或永久。空/未知文案一律按永久处理。 */
export function classifyClipEditFailure(message: unknown): ClipEditFailureKind {
    const text = typeof message === "string" ? message.trim() : "";
    if (!text) {
        return "permanent";
    }
    return TRANSIENT_MARKERS.some((marker) => text.includes(marker)) ? "transient" : "permanent";
}
