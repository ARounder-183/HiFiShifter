/**
 * 片段编辑失败分类：区分"后端正在自行收敛"与"需要用户处置"。
 *
 * 【为什么需要分类】此前 `setClipStateRemote` / `setClipsStateBulkRemote` 的
 * `.rejected` 一律把状态置成 `"Clip edit rejected"`（"音频块编辑被拒绝"），而任何
 * 失败模式都走同一条路 —— 包括**瞬时**的"宿主还在追上几何/媒体"（后端会自己重试并
 * 收敛，不需要用户做任何事）。结果用户每做一次编辑都被报一次错，即使编辑其实生效了。
 *
 * 【为什么按文案匹配而不是后端给 code】这些字符串由后端在宿主回调边界产出，是既有的
 * 稳定契约（且都有对应测试钉住）。
 *
 * 【为什么入参是一个结构而不是一个字符串（这是一个真实缺陷的根因）】此前调用方传的是
 * `action.error.message` —— 而那是 Tauri/插件的**通用包装文案**
 * （`Backend invoke failed: plugin:set_clip_state`），后端原文在 `cause` 里，
 * `kind` / `code` 在 RTK 的 `miniSerializeError`（只复制 name/message/stack/code
 * 四个字符串属性）里被丢掉。于是白名单**永远匹配不到任何一条**，所有失败都被判成
 * 永久 —— 用户看到的正是"编辑明明生效了，却报音频块编辑被拒绝"。
 * 现在入参带上后端原文（`detail`）与协议分类（`kind`），白名单才真正生效。
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
    // 【为什么加这一条】后端的提交闸在"宿主在我编辑期间动过"时会报
    // `Conflict: workspace …`（此前是 `Conflict: workspace scope or curves changed`，
    // 现在是 `Conflict: workspace edits could not be rebased`）。两者都是**可重基**的
    // 瞬时状态：后端自己会换基线重放，用户不需要做任何事。此前它落在白名单外，
    // 于是每次"移动 item 后顺手调一下参数"都闪一次"音频块编辑被拒绝" —— 而编辑
    // 其实已经生效。
    "Conflict: workspace",
    "could not be rebased",
    "refresh required",
    "budget exceeded",
    "lease revoked",
    "host PCM unavailable",
    "host model not ready",
];

/** 一次片段编辑失败的**可分类信息**（由 thunk 的 `rejectWithValue` 提供）。 */
export interface ClipEditFailureInput {
    /** 插件协议给出的分类：`terminal` = 会话已结束。 */
    kind?: string;
    /** 通用包装文案（`Backend invoke failed: …`），拿不到后端原文时的兜底。 */
    message?: string;
    /** 后端原文（沿 `cause` 链取到的最内层 message）。 */
    detail?: string;
}

/**
 * 把一次片段编辑失败归类为瞬时或永久。空/未知文案一律按永久处理。
 *
 * 判据顺序：
 * 1. `kind === "terminal"` ⇒ 永久（会话已结束，重试无意义）；
 * 2. 有后端原文就按白名单判 —— 白名单本来就是为这些原文写的；
 * 3. 没有原文（旧后端 / 非插件形态）退回包装文案，仍然匹配不到 ⇒ 永久（保守）。
 *
 * 【为什么不直接按 `kind === "transient"` 静默】协议的 `kind` 只回答"会话是否还能
 * 继续"，**不是**"这条错误该不该给用户看"。`invalid host channel mode` 这类校验失败
 * 同样被判成 transient，按 kind 静默就会把"你的输入不被支持"一起吞掉 —— 比误报更糟。
 */
export function classifyClipEditFailure(input: unknown): ClipEditFailureKind {
    if (typeof input === "string") {
        // 兼容旧调用点：纯字符串按文案判（等价于"没有 kind、没有 detail"）。
        return classifyByText(input);
    }
    if (!input || typeof input !== "object") {
        return "permanent";
    }
    const { kind, message, detail } = input as ClipEditFailureInput;
    if (kind === "terminal") {
        return "permanent";
    }
    return classifyByText(detail ?? message ?? "");
}

function classifyByText(value: unknown): ClipEditFailureKind {
    const text = typeof value === "string" ? value.trim() : "";
    if (!text) {
        return "permanent";
    }
    return TRANSIENT_MARKERS.some((marker) => text.includes(marker)) ? "transient" : "permanent";
}
