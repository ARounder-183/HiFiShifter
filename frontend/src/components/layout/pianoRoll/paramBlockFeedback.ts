/**
 * 操作反馈的全局状态栏通道（**单槽位**）。
 *
 * 画布内不再渲染块错误浮层；所有失败、拒绝与需要用户处置的提示都从这里进入
 * 左下角状态栏，与音高分析 / 波形分析共用同一处视线落点。
 *
 * 【为什么名字里带 paramBlock 而别的功能也在用】它是为参数块写入失败建的，但形状
 * 与"任何一次性命令做完了说一声"完全一致（一个槽位、tone、tooltip、停留时长）。
 * 音高互转等命令通过下面的 `publishStatusFeedback` 别名写同一个槽位 —— 改名要动
 * 十几个调用点与两份测试，收益只是措辞；而**两套**单槽位并排出现才是真的错。
 */
import {
    PARAM_BLOCK_FEEDBACK_HOLD_MS,
    type ParamBlockFailureFeedback,
} from "./paramBlockCommitError";

export interface ParamBlockFeedback {
    readonly text: string;
    readonly tone: "success" | "warning" | "danger" | "accent";
    readonly tooltip?: string;
    readonly holdMs?: number;
}

let current: ParamBlockFeedback | null = null;
let hideTimer: ReturnType<typeof setTimeout> | null = null;
/** 上一次真正发布的时间与内容（用于同文案去重）。 */
let lastPublishedAt = Number.NEGATIVE_INFINITY;
const listeners = new Set<() => void>();

/**
 * 前端自产 detail → i18n 键。
 *
 * `detail` 有两个来源：后端诊断串（码 / 前缀，保持英文原文，同 `errorCodeKey`
 * 的策略）与前端 throw 出的用户可读句子。后者里只有"正常使用就会遇到"的两条
 * 需要翻译 —— 逐条列在这里而不是在 throw 点拼 i18n，是为了让错误对象里的
 * detail 保持稳定的英文原文（日志、测试与前端间的字符串匹配都依赖它）。
 * 内部不变量类的 detail（expectedCurveRevision 之类）是 bug 线索，不翻译。
 */
const DETAIL_KEY: Record<string, string> = {
    "Conflict: parameter curve changed during block read":
        "status_param_block_conflict_curve_changed",
    "parameter block edit set is not loaded": "status_param_block_edit_set_not_loaded",
};

/** tooltip 文案：能映射的 detail 按当前语言取词，其余（后端码 / 诊断串）原样。 */
function tooltipText(
    tf: (key: string) => string,
    error: { code: string; detail?: string },
): string {
    if (error.detail !== undefined) {
        const key = DETAIL_KEY[error.detail];
        if (key !== undefined) return tf(key);
        return error.detail;
    }
    return error.code;
}

/** 同文案同 tone 在这个窗口内重复发布会被丢掉（连点两次同一命令不该刷两条）。 */
const DUPLICATE_FEEDBACK_WINDOW_MS = 500;

function emit(): void {
    for (const listener of listeners) listener();
}

export function publishParamBlockFeedback(next: ParamBlockFeedback | null): void {
    /*
     * 【单槽去重】这里只有一个槽位，后发布的会覆盖先发布的。连续两次同一命令
     * 失败（例如连按两次分割）会产生两条一模一样的提示，视觉上像"没反应"——
     * 短暂窗口内的重复直接丢弃。
     */
    if (next !== null && current !== null) {
        if (next.text === current.text && next.tone === current.tone) {
            if (performance.now() - lastPublishedAt < DUPLICATE_FEEDBACK_WINDOW_MS) return;
        }
    }
    if (hideTimer !== null) clearTimeout(hideTimer);
    hideTimer = null;
    current = next;
    lastPublishedAt = performance.now();
    emit();
    if (next?.holdMs && next.holdMs > 0) {
        hideTimer = setTimeout(() => {
            current = null;
            hideTimer = null;
            emit();
        }, next.holdMs);
    }
}

/**
 * 把一次**写入失败**翻译成状态栏反馈。
 *
 * 【为什么收在这里】四类失败（rejected / conflict / transport / silent）各有自己
 * 的文案、tone 与停留时长；散在面板里会让"同一类失败两种措辞"再次出现。
 * 旧实现把四类压成一句 danger，用户无法判断"我的编辑到底有没有生效"。
 *
 * @param tf i18n 取词函数（面板注入，保持本模块与 i18n 解耦）。
 */
export function publishParamBlockWriteFailure(args: {
    readonly failure: ParamBlockFailureFeedback;
    readonly error: { readonly code: string; readonly detail?: string };
    readonly tf: (key: string) => string;
}): void {
    const { failure, error, tf } = args;
    if (failure === "silent") return;
    if (failure === "rejected") {
        publishParamBlockFeedback({
            text: tf("status_param_block_edit_rejected"),
            tone: "warning",
            tooltip: tooltipText(tf, error),
            holdMs: PARAM_BLOCK_FEEDBACK_HOLD_MS.rejected,
        });
        return;
    }
    if (failure === "conflict") {
        publishParamBlockFeedback({
            text: tf("status_param_block_conflict_resynced"),
            tone: "warning",
            tooltip: tooltipText(tf, error),
            holdMs: PARAM_BLOCK_FEEDBACK_HOLD_MS.conflict,
        });
        return;
    }
    publishParamBlockFeedback({
        text: tf("status_param_block_write_failed"),
        tone: "danger",
        tooltip: tooltipText(tf, error),
        holdMs: PARAM_BLOCK_FEEDBACK_HOLD_MS.transport,
    });
}

/**
 * 同一个槽位的通用名字（参数块之外的调用方用它，见模块头）。
 *
 * 不是包装函数：多一层转发只会让"这是同一个槽位"变得不明显。
 */
export const publishStatusFeedback = publishParamBlockFeedback;

export const paramBlockFeedbackStore = {
    subscribe(listener: () => void): () => void {
        listeners.add(listener);
        return () => listeners.delete(listener);
    },
    getSnapshot(): ParamBlockFeedback | null {
        return current;
    },
};
