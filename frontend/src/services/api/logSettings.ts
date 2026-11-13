/**
 * logSettingsApi — 帮助菜单里的日志详细程度（唯一入口）。
 *
 * 只有两个命令：读视图、写档位。清理历史日志 / 日志尾部预览都需要独立的确认框或
 * 查看面板（=额外入口），按需求明确不做，因此这里不注册也不导出它们。
 * 返回的只读字段（files / totalBytes / recentChanges ...）只供菜单状态行与诊断包使用。
 */
import { invoke } from "../invoke";

export type LogVerbosity = "off" | "low" | "medium" | "high";

export type LogPolicySource = "default" | "persisted" | "env" | "cli" | "temporary";

export interface LogSettingsView {
    ok: boolean;
    error?: string;
    verbosity: LogVerbosity;
    persistedVerbosity: LogVerbosity;
    source: LogPolicySource;
    overridden: boolean;
    overrideHint?: string | null;
    fileEnabled: boolean;
    fileWritable: boolean;
    loggingHealthy: boolean;
    hostLoggerOwned: boolean;
    lastError?: string | null;
    logDir?: string | null;
    logFile?: string | null;
    sessionId?: string;
    files: { name: string; bytes: number; modifiedAt: string }[];
    totalBytes: number;
    maxFileBytes: number;
    maxFiles: number;
    temporaryUntil?: string;
    policyVersion: 1;
    recentChanges: { at: string; from: LogVerbosity; to: LogVerbosity; source: string }[];
}

/** 只读命令：「无」档下也可用，否则用户无法把档位改回来。 */
export function getLogSettings(): Promise<LogSettingsView> {
    return invoke("get_log_settings");
}

/** 写档位：后端做持久化 + 热切换 + 自证行 + 广播事件。 */
export function setLogSettings(verbosity: LogVerbosity): Promise<LogSettingsView> {
    return invoke("set_log_settings", verbosity);
}
