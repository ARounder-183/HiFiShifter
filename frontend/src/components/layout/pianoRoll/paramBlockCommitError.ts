/**
 * 参数块写入失败的**分类**。
 *
 * 【为什么需要】旧实现把所有失败压成一句「参数块操作失败」，并且靠
 * `message.includes(...)` 猜原因。于是：CAS 冲突（"世界变了"）与非法编辑集
 * （"操作非法"）与 IPC 断开（"环境坏了"）在用户眼里是同一件事，也都不可重试。
 *
 * 【分类协议】后端的两类前缀是**协议**，不是文案：
 * - `Conflict: ` —— CAS 失败。世界变了，**可以**刷新基准后重试；
 * - `Invalid: ` —— 校验失败。操作非法，重试只会再失败一次。
 * 其余一律视为传输层问题（IPC 断开 / 序列化失败），最多对账后重试一次。
 *
 * 【它不做什么】不决定文案、不发提示、不重试 —— 只回答"这是什么失败"。
 */
import { backendDetail } from "../../../services/invoke";

export type ParamBlockFailureKind = "conflict" | "rejected" | "transport" | "cancelled";

/** `conflict` 的细分：决定重试用哪种重基方式。 */
export type ParamBlockConflictCode = "edit_set" | "curve" | "geometry";

export class ParamBlockCommitError extends Error {
    readonly kind: ParamBlockFailureKind;
    readonly code: string;
    readonly detail?: string;

    constructor(args: {
        readonly kind: ParamBlockFailureKind;
        readonly code: string;
        readonly detail?: string;
    }) {
        super(args.detail ?? args.code);
        this.name = "ParamBlockCommitError";
        this.kind = args.kind;
        this.code = args.code;
        this.detail = args.detail;
    }
}

const CONFLICT_PREFIX = "Conflict: ";
const INVALID_PREFIX = "Invalid: ";

/** 从后端原文里取出 `Invalid:` 后面的短语，作为稳定的 code。 */
function rejectedCode(detail: string): string {
    const phrase = detail.slice(INVALID_PREFIX.length).trim();
    if (phrase.length === 0) return "invalid";
    return phrase.replace(/\s+/g, "_").toLowerCase();
}

/**
 * 错误 → 分类。
 *
 * 【为什么按前缀而不是 includes】validate.rs 的约定是"错误字符串统一以
 * `Invalid: ` 开头"，replace 的 CAS 统一以 `Conflict: ` 开头。按前缀分类是
 * **协议**，按 includes 分类是**猜文案**。
 */
export function classifyParamBlockFailure(error: unknown): ParamBlockCommitError {
    if (error instanceof ParamBlockCommitError) return error;
    const detail = backendDetail(error) ?? (error instanceof Error ? error.message : String(error));

    if (detail.startsWith(INVALID_PREFIX)) {
        return new ParamBlockCommitError({
            kind: "rejected",
            code: rejectedCode(detail),
            detail,
        });
    }
    if (detail.startsWith(CONFLICT_PREFIX)) {
        const rest = detail.slice(CONFLICT_PREFIX.length);
        if (rest.includes("parameter block edits changed")) {
            return new ParamBlockCommitError({ kind: "conflict", code: "edit_set", detail });
        }
        if (rest.includes("parameter curve changed")) {
            return new ParamBlockCommitError({ kind: "conflict", code: "curve", detail });
        }
        if (rest.includes("parameter geometry changed")) {
            return new ParamBlockCommitError({ kind: "conflict", code: "geometry", detail });
        }
        return new ParamBlockCommitError({ kind: "conflict", code: "unknown", detail });
    }
    // 取消是**预期**结果（手势被新手势取代 / epoch 失效），不是错误。
    if (detail.toLowerCase().includes("cancelled")) {
        return new ParamBlockCommitError({ kind: "cancelled", code: "cancelled", detail });
    }
    return new ParamBlockCommitError({ kind: "transport", code: "transport", detail });
}

/**
 * 提交队列**可以重基后重试**的失败 —— 所有 CAS 冲突。
 *
 * 【为什么从「只有 edit_set」放宽到「所有 conflict」】旧判据的前提是「补丁是死的，
 * 重放同一份补丁只可能再失败」。现在写队列把**读基准**也搬进了队列槽
 * （`ParamBlockWritePlan.prepare`），重试等于「重新读一遍世界、按同一个意图重算
 * 补丁」—— 这正是 `curve` / `geometry` 冲突的正确修复方式。
 *
 * 【为什么仍然排除 transport】传输失败时上一次写入**可能已经落地**，重试会把
 * 一次用户操作变成两次写入（历史里出现两步）。
 *
 * 【为什么排除 rejected / cancelled】前者是确定性的（再跑一次还是非法），后者是
 * 预期内的作废（手势已被取代），重试都只会让用户多等一次。
 */
export function canRebaseParamBlockCommit(error: ParamBlockCommitError): boolean {
    return error.kind === "conflict";
}

/**
 * 用户可见反馈的**停留时长**（毫秒）。
 *
 * 【为什么收成常量】旧实现让每个调用点自己写 `holdMs`：同一类失败在不同场景停留
 * 3s / 5s / 8s / 15s 不等，用户无法形成预期。三类各一个数字，理由见行内注释。
 */
export const PARAM_BLOCK_FEEDBACK_HOLD_MS = {
    /** 校验拒绝：需要读 tooltip 才知道改什么，给足时间。 */
    rejected: 6_000,
    /** 冲突已自动重同步：告知性，不要求处置。 */
    conflict: 6_000,
    /** 写入失败：可能丢编辑，必须留到用户看见并处置。 */
    transport: 15_000,
} as const;
/** 用户可见反馈的类别（由面板映射到具体文案与 tone）。 */
export type ParamBlockFailureFeedback = "rejected" | "conflict" | "transport" | "silent";

/**
 * 失败 → 用户可见类别。
 *
 * 【cancelled 在类型上就是 silent】它不是错误：手势被新手势取代、读窗口不完整、
 * 外部历史在提交途中变了 —— 这些情况下「本次手势作废」是**正确结果**，弹一条
 * danger 会让用户以为编辑丢了。
 */
export function paramBlockFailureFeedback(error: ParamBlockCommitError): ParamBlockFailureFeedback {
    if (error.kind === "cancelled") return "silent";
    return error.kind;
}
