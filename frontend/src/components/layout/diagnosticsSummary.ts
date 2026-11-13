/**
 * 「复制诊断摘要」的纯数据模型。
 *
 * 【为什么单独抽出】MenuBar.tsx 太大、没有测试宿主；而这段逻辑是用户反复
 * 踩到的路径：复制失败时既看不到摘要本身，又只知道一句猜出来的原因。
 * 抽成纯函数后，"摘要里有什么""失败该给用户什么"都有可执行的回归门禁。
 *
 * 它**不调用** `open_log_folder`：那条命令会创建目录并打开资源管理器，
 * 复制一段文字不该产生这些副作用（在「无」档下更不该新建日志目录）。
 */
import type { LogSettingsView } from "../../services/api/logSettings";

export interface AboutInfo {
    version?: string;
    commit?: string | null;
    commitShort?: string | null;
    dirty?: boolean;
}

export interface RuntimeInfo {
    device?: string;
}

export function diagnosticsSummary(
    about: AboutInfo | null,
    runtime: RuntimeInfo | null,
    settings: LogSettingsView | null,
): string {
    const logPath = settings?.logFile || settings?.logDir;
    return [
        `HiFiShifter ${about?.version ?? "unknown"}`,
        `commit: ${about?.commit ?? about?.commitShort ?? "unknown"}${
            about?.dirty ? " (dirty)" : ""
        }`,
        `host: ${runtime?.device ?? "unknown"}`,
        `log verbosity: ${settings?.verbosity ?? "unknown"}`,
        `logging: ${settings?.loggingHealthy === false ? "unhealthy" : "enabled"}`,
        `log: ${logPath ?? (settings?.fileEnabled === false ? "disabled" : "unavailable")}`,
    ].join("\n");
}
