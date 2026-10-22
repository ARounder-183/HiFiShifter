// 宿主链接的前端门面。
//
// 【它取代了什么】旧的 "ARA 宿主连接" 面向一个**活着的实例**：`ara_list_instances`
// 扫心跳目录、`ara_connect` 走两次往返、会话只在内存里、断开即失去一切。现在宿主导出的
// 是一份磁盘上的**工作区目录**，没有在线/离线之分，只有"打开"和"提交"。因此这里
// 既没有实例选择，也没有心跳与令牌。
//
// 【为什么与插件侧的导出分两组】独立 App 走 Tauri 命令（`host_link_list/open/…`），
// 插件窗口走插件自己的命令通道（`host_link_export/status/release`，与
// `plugin_get_apply_state` 同一条）。两者目标不同、返回形状也不同，放在一起会让
// "这个命令在哪种形态下可用"要靠读者推断（与 `captureApi` 的取舍一致）。
import { invoke } from "../../services/invoke";
import type { MessageKey } from "../../i18n/messages";

/** 工作区的来源：ARA 插件实例、Capture 录制、或宿主侧扩展桥接。 */
export type HostLinkSourceKind = "ara" | "capture" | "bridge";

/** 工作区音频的完整度读数。 */
export interface HostLinkAudio {
    complete: number;
    partial: number;
    extern: number;
}

/** 列表里的一个工作区目录；读不出来时只带 `unreadable` 与目录名。 */
export interface HostLinkItem {
    directory: string;
    /** 目录读不出来时的原因（此时其余字段缺失）。 */
    unreadable?: string;
    identity?: string;
    hostLabel?: string;
    sourceKind?: HostLinkSourceKind;
    createdAt?: number;
    clips?: number;
    tracks?: number;
    audio?: HostLinkAudio;
    losses?: number;
    acceptsCommit?: boolean;
}

/** `host_link_list` 的返回：扫描根目录 + 其中的工作区。 */
export interface HostLinkList {
    root: string;
    items: HostLinkItem[];
}

/** 宿主对一次提交的回答（`LinkStatus` 的前端形状）。 */
export interface HostLinkSubmitStatus {
    ok: boolean;
    /** 协议级错误码（见 `hifishifter_host_link::error_code`）。 */
    code?: number;
    error?: string;
    atUnixMs: number;
}

/** 打开 / 刷新 / 提交后的会话摘要。 */
export interface HostLinkSession {
    attached: boolean;
    directory: string;
    identity: string;
    hostLabel: string;
    sourceKind: HostLinkSourceKind;
    createdAt: number;
    clips: number;
    tracks: number;
    audio: HostLinkAudio;
    acceptsCommit: boolean;
    /** `true` = 宿主拥有几何，本地几何改动无法提交。 */
    gatesGeometry: boolean;
    /** 两个修订计数器：编辑代次与宿主模型代次。 */
    edits: number;
    model: number;
    /** 逐条降级的摘要（"跳过了什么、为什么"）。 */
    skipped: string;
    skippedCount: number;
    skippedDetail: string[];
    /** 宿主清单自报的损失。 */
    losses: string[];
    unsupportedPrecision: boolean;
    /** 仅 open / refresh 返回：完整时间轴载荷。 */
    timeline?: unknown;
    /** 仅 refresh 返回：重新导入后保留下来的条数。 */
    carriedParams?: number;
    carriedTracks?: number;
    /** 仅 submit 返回。 */
    applied?: boolean;
    pending?: boolean;
    status?: HostLinkSubmitStatus | null;
}

export const hostLinkApi = {
    /** 列出可打开的工作区；`directory` 缺省时用后端默认根目录。 */
    list: (directory?: string) => invoke<HostLinkList>("host_link_list", directory),
    open: (directory: string, force: boolean) =>
        invoke<HostLinkSession>("host_link_open", directory, force),
    refresh: (force: boolean) => invoke<HostLinkSession>("host_link_refresh", force),
    submit: () => invoke<HostLinkSession>("host_link_submit"),
    release: () => invoke<{ ok: true; attached: false }>("host_link_release"),
};

/** 插件侧导出工作区的状态阶段。 */
export type HostLinkExportPhase = "running" | "failed" | "ready";

/** 插件侧 `host_link_status` / `host_link_export` 的负载。 */
export interface HostLinkExportStatus {
    attached: boolean;
    phase?: HostLinkExportPhase;
    /** `phase === "failed"` 时的原因。 */
    error?: string;
    identity?: string;
    hostLabel?: string;
    sourceKind?: string;
    directory?: string;
    exportedAt?: number;
    clips?: number;
    tracks?: number;
    audio?: HostLinkAudio;
    losses?: number;
    acceptsCommit?: boolean;
    diagnostics?: unknown;
    /** 宿主对最近一次回写的回答。 */
    lastEdit?: HostLinkSubmitStatus & { edits: number; model: number };
}

export const hostLinkPluginApi = {
    /** 开始一次后台导出；立即返回（结果由 `status` 轮询）。 */
    export: () => invoke<HostLinkExportStatus>("host_link_export"),
    status: () => invoke<HostLinkExportStatus>("host_link_status"),
    release: () => invoke<HostLinkExportStatus>("host_link_release"),
};

/**
 * 取出后端错误的可读文本。
 *
 * 【为什么要剥 `cause`】`invoke` 把底层错误包在 `BackendInvokeError.cause` 里，
 * 真正的分类前缀在更内层。与旧的 `araError` 同一套解包口径。
 */
export function hostLinkError(error: unknown): string {
    if (error instanceof Error)
        return error.cause !== undefined ? hostLinkError(error.cause) : error.message;
    return String(error);
}

/**
 * 门禁拒绝的分类前缀；其后是**语言无关**的字段路径列表，例如
 * `ara_host_fields: clips[2].start_sec: 0.0 -> 0.25`。
 */
const HOST_FIELDS_PREFIX = "ara_host_fields:";

/** 本地工程有未保存改动（调用方应确认后用 `force: true` 重试）。 */
const DIRTY_PROJECT_PREFIX = "dirty_project:";

/** 另一个窗口已经打开这个工作区。 */
const ALREADY_OPEN_PREFIX = "this workspace is already open";

/** 没有打开任何工作区。 */
const NOT_OPEN = "no host workspace is open";

/** 宿主不接受提交。 */
const NO_EDITS = "this host does not accept edits";

/** 后端是否报告"本地工程有未保存改动"。调用方据此进入确认态，而不是把错误摊给用户。 */
export function isDirtyProjectError(error: unknown): boolean {
    return hostLinkError(error).startsWith(DIRTY_PROJECT_PREFIX);
}

/**
 * 宿主链接错误的**展示**文案。
 *
 * 【为什么分类留在后端、文案留在前端】后端只返回语言无关的分类与字段路径，
 * 文案按 catalog 本地化 —— 与 `history_op_*` 同一原则：后端不持有 UI 文案。
 *
 * 【为什么必须指名字段】几何门禁拒绝时，`ara_host_fields:` 后面跟着具体字段路径
 * （`clips[2].start_sec`）；只说"宿主时间线变了"用户无从下手。
 *
 * 其余错误原样透传，由调用方按既有分支处理。
 */
export function hostLinkErrorText(
    error: unknown,
    t: (key: MessageKey) => string,
    tVars: (key: MessageKey, vars: Record<string, string | number>) => string,
): string {
    const raw = hostLinkError(error);
    if (raw.startsWith(HOST_FIELDS_PREFIX)) {
        const fields = raw.slice(HOST_FIELDS_PREFIX.length).trim();
        return fields
            ? tVars("host_link_submit_blocked_host_fields", { fields })
            : t("host_link_submit_blocked");
    }
    if (raw.startsWith(DIRTY_PROJECT_PREFIX)) return t("host_link_dirty_project");
    if (raw.startsWith(ALREADY_OPEN_PREFIX)) return t("host_link_already_open");
    if (raw.startsWith(NOT_OPEN)) return t("host_link_not_open");
    if (raw.startsWith(NO_EDITS)) return t("host_link_commit_refused");
    return raw;
}

/**
 * 宿主对一次提交的回答 → 展示文案。
 *
 * 【为什么优先看字段路径再看码】`GEOMETRY_CHANGED`（13）同时带码与 `error`，而
 * `error` 里的字段列表才是用户能据以行动的东西。因此在码之前先试字段路径；
 * 命中了就展示字段，没命中才按码给一句泛化的说明。
 */
export function hostLinkStatusText(
    status: HostLinkSubmitStatus,
    t: (key: MessageKey) => string,
    tVars: (key: MessageKey, vars: Record<string, string | number>) => string,
): string {
    if (status.error) {
        const localised = hostLinkErrorText(status.error, t, tVars);
        if (localised !== status.error) return localised;
    }
    switch (status.code) {
        case 10:
            return t("host_link_code_unsupported_format");
        case 11:
            return t("host_link_code_model_not_ready");
        case 12:
            return t("host_link_code_conflict");
        case 13:
            return t("host_link_submit_blocked");
        case 14:
            return tVars("host_link_code_invalid_edit", { error: status.error ?? "" });
        case 15:
            return t("host_link_code_link_mismatch");
        case 1:
        case 2:
        case 3:
        case 4:
        case 5:
        case 6:
        case 7:
        case 8:
        case 9:
            return t("host_link_code_document_closed");
        default:
            return status.error ?? t("host_link_submit_failed");
    }
}
