// ARA 命令通过既有调用门面传输，宿主 token 与管道信息不进入前端。
import { invoke } from "../../services/invoke";
import type { MessageKey } from "../../i18n/messages";

export interface AraInstance {
    instance_id: string;
    name: string;
    pid: number;
}
export interface AraResult {
    ok: boolean;
    error?: string;
    instance_id?: string;
    revision?: number;
    model_revision?: number;
}
export const araApi = {
    list: () => invoke<AraInstance[]>("ara_list_instances"),
    connect: (instanceId: string, force: boolean) =>
        invoke<AraResult>("ara_connect", instanceId, force),
    refresh: (force: boolean) => invoke<AraResult>("ara_refresh", force),
    submit: () => invoke<AraResult>("ara_submit"),
    disconnect: () => invoke<AraResult>("ara_disconnect"),
};

/** 保留后端 Conflict 和脏工程诊断，不展示调用包装器的通用错误。 */
export function araError(error: unknown): string {
    if (error instanceof Error)
        return error.cause !== undefined ? araError(error.cause) : error.message;
    return String(error);
}

/**
 * 门禁拒绝的分类前缀；其后是**语言无关**的字段路径列表，例如
 * `ara_host_fields: clips[0].start_sec: 0.0 -> 0.25`。
 */
const ARA_HOST_FIELDS_PREFIX = "ara_host_fields:";

/** 宿主时间线已变（clip 集合不同），需要重新连接。 */
const ARA_RECONNECT_REQUIRED = "ara_reconnect_required";

/**
 * ARA 错误的**展示**文案。
 *
 * 【为什么分类留在后端、文案留在前端】后端只返回语言无关的分类与字段路径，
 * 文案按 catalog 本地化 —— 与 `history_op_*` 同一原则：后端不持有 UI 文案。
 *
 * 【为什么必须指名字段】旧实现无论实际漂移的是哪一项，都固定指控
 * "几何 / 增益 / 名字"，用户无法据以行动。现在报出 `clips[2].start_sec` 这样的
 * 具体路径，用户才知道该去 REAPER 改什么。
 *
 * 其余错误（含 Conflict 与 `dirty_project:`）原样透传，由调用方按既有分支处理。
 */
export function araErrorText(
    error: unknown,
    t: (key: MessageKey) => string,
    tVars: (key: MessageKey, vars: Record<string, string | number>) => string,
): string {
    const raw = araError(error);
    if (raw.startsWith(ARA_HOST_FIELDS_PREFIX)) {
        const fields = raw.slice(ARA_HOST_FIELDS_PREFIX.length).trim();
        return fields
            ? tVars("ara_submit_blocked_host_fields", { fields })
            : t("ara_submit_blocked");
    }
    if (raw.startsWith(ARA_RECONNECT_REQUIRED)) return t("ara_submit_reconnect_required");
    return raw;
}
